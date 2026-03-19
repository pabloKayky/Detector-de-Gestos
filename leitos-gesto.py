"""
=============================================================================
RASTREADOR DE GESTOS COM VISÃO COMPUTACIONAL
=============================================================================
Descrição:
    Captura o vídeo da webcam em tempo real, detecta a mão do usuário e
    classifica o gesto atual usando pontos de referência anatômicos (landmarks)
    fornecidos pelo MediaPipe. Ao lado da câmera, exibe uma imagem ilustrativa
    correspondente ao gesto detectado.

Tecnologias:
    - OpenCV    → captura de vídeo e exibição da janela
    - MediaPipe → detecção de mãos (21 pontos) e rosto (468 pontos)
    - NumPy     → operações com arrays de imagem
    - ctypes    → API Win32 para mover a janela (somente Windows)

Gestos reconhecidos:
    JOINHA, APONTANDO, MAO_ABERTA, PUNHO, PAZ,
    HANG_LOOSE, DEDO_MEIO, ROCK, PENSANDO, NEUTRO

Controles:
    Q ou ESC → encerra o programa
    Clicar e arrastar a janela → move a janela pela tela
=============================================================================
"""

import cv2        # OpenCV — captura de câmera e renderização
import mediapipe as mp  # MediaPipe — detecção de mãos e rosto
import numpy as np      # NumPy — manipulação de arrays de imagem
import os               # os — para montar caminhos de arquivo
import time             # time — para controle de tempo no log do terminal


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminho absoluto da pasta onde este script está salvo.
# Garante que as imagens sejam encontradas independentemente de onde
# o script é executado (resolve o erro "LOAD IMAGE FAILED").
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Nome da janela exibida pelo OpenCV.
# Precisa ser idêntico em todas as chamadas pois o Win32 localiza
# a janela pelo título para permitir o arraste.
NOME_JANELA = 'Rastreador de Gestos'


# =============================================================================
# ESTADO PARA ARRASTAR A JANELA
# Variáveis globais compartilhadas entre os eventos do callback do mouse.
# =============================================================================

_posicao_inicial_mouse  = None   # Posição (x, y) do mouse na tela quando clicou
_posicao_inicial_janela = None   # Posição (x, y) da janela nesse mesmo instante
_arrastando             = False  # True enquanto o botão esquerdo estiver pressionado


# =============================================================================
# ESTADO DO LOG NO TERMINAL
# Evita imprimir o mesmo gesto centenas de vezes por segundo.
# =============================================================================

_ultimo_gesto_logado = None  # Último gesto que foi impresso no terminal
_ultimo_tempo_log    = 0     # Timestamp (segundos) do último log realizado
INTERVALO_LOG        = 0.5   # Tempo mínimo (segundos) entre logs do mesmo gesto


# =============================================================================
# DICIONÁRIO: código interno do gesto → nome amigável exibido na tela/terminal
# =============================================================================

NOMES_GESTOS = {
    "JOINHA":    "JOINHA",
    "APONTANDO": "APONTANDO",
    "NEUTRO":    "NEUTRO",
    "PENSANDO":  "PENSANDO",
    "ROCK":      "ROCK",
    "PUNHO":     "PUNHO",
    "HANG_LOOSE": "HANG LOOSE",
    "PAZ":       "PAZ E AMOR",
    "MAO_ABERTA": "MAO ABERTA",
    "DEDO_MEIO": "DEDO DO MEIO",
}


# =============================================================================
# DICIONÁRIO: código interno do gesto → arquivo de imagem correspondente
# Todos os arquivos devem estar na mesma pasta deste script.
# =============================================================================

IMAGENS_GESTOS = {
    "JOINHA":    "joinha.jpg",
    "APONTANDO": "apontando.jpg",
    "NEUTRO":    "neutral.jpg",
    "PENSANDO":  "pensando.jpg",
    "ROCK":      "rock.jpg",
    "PUNHO":     "punho.jpg",
    "HANG_LOOSE": "Hang loose.jpg",
    "PAZ":       "paz.jpg",
    "MAO_ABERTA": "aberta.jpg",
    "DEDO_MEIO": "meio.jpg",
}


# =============================================================================
# INICIALIZAÇÃO DO MEDIAPIPE
# =============================================================================

# Utilitário de desenho: responsável por desenhar os pontos e conexões da mão
mp_desenho = mp.solutions.drawing_utils

# Módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands

# Módulo de malha facial do MediaPipe (usado para localizar o nariz)
mp_rosto = mp.solutions.face_mesh

# Instância do detector de mãos:
#   static_image_mode=False    → modo vídeo contínuo (mais eficiente que processar foto a foto)
#   max_num_hands=1            → rastreia no máximo 1 mão por vez
#   min_detection_confidence   → confiança mínima para INICIAR o rastreamento (70%)
#   min_tracking_confidence    → confiança mínima para MANTER o rastreamento entre frames (60%)
detector_maos = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# Instância do detector de rosto:
#   max_num_faces=1  → detecta apenas 1 rosto por vez
#   confidências 50% → equilibra velocidade e precisão para detecção facial
detector_rosto = mp_rosto.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def callback_mouse(evento, x, y, flags, parametro):
    """
    Função chamada automaticamente pelo OpenCV a cada evento do mouse na janela.
    Implementa o arraste da janela clicando e segurando (somente Windows).

    Fluxo:
        CLIQUE    → salva posição atual do mouse e da janela como referência
        MOVIMENTO → calcula deslocamento e reposiciona a janela via API Win32
        SOLTAR    → encerra o estado de arraste

    Parâmetros (fornecidos automaticamente pelo OpenCV):
        evento    → tipo do evento (clique, movimento, soltar)
        x, y      → posição do mouse DENTRO da janela (não usada aqui)
        flags     → teclas modificadoras pressionadas (não usada aqui)
        parametro → dado extra passado ao registrar o callback (não usado aqui)
    """
    global _posicao_inicial_mouse, _posicao_inicial_janela, _arrastando
    import ctypes  # Importado aqui para não causar erro em sistemas não-Windows

    if evento == cv2.EVENT_LBUTTONDOWN:
        # ── Botão esquerdo pressionado: inicia o arraste ──────────────────────
        _arrastando = True

        # Captura a posição ABSOLUTA do cursor na tela (coordenadas do monitor inteiro)
        ponto = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(ponto))
        _posicao_inicial_mouse = (ponto.x, ponto.y)

        # Captura a posição atual da janela pesquisando pelo seu título
        handle = ctypes.windll.user32.FindWindowW(None, NOME_JANELA)
        retangulo = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(handle, ctypes.byref(retangulo))
        _posicao_inicial_janela = (retangulo.left, retangulo.top)

    elif evento == cv2.EVENT_MOUSEMOVE and _arrastando:
        # ── Mouse em movimento com botão pressionado: arrasta a janela ────────
        ponto = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(ponto))

        # Deslocamento = posição atual do mouse - posição no momento do clique
        delta_x = ponto.x - _posicao_inicial_mouse[0]
        delta_y = ponto.y - _posicao_inicial_mouse[1]

        # Nova posição da janela = posição original + deslocamento
        nova_x = _posicao_inicial_janela[0] + delta_x
        nova_y = _posicao_inicial_janela[1] + delta_y

        # Move a janela via Win32; flag 0x0001 (SWP_NOSIZE) preserva o tamanho
        handle = ctypes.windll.user32.FindWindowW(None, NOME_JANELA)
        ctypes.windll.user32.SetWindowPos(handle, None, nova_x, nova_y, 0, 0, 0x0001)

    elif evento == cv2.EVENT_LBUTTONUP:
        # ── Botão solto: encerra o arraste ────────────────────────────────────
        _arrastando = False


def logar_gesto(gesto):
    """
    Imprime o gesto detectado no terminal somente quando ele MUDA.
    Evita spam de centenas de linhas idênticas por segundo.

    Parâmetros:
        gesto (str): código interno do gesto (ex: "JOINHA", "PAZ")
    """
    global _ultimo_gesto_logado, _ultimo_tempo_log
    agora = time.time()

    if gesto != _ultimo_gesto_logado:
        # Formata o horário atual como HH:MM:SS
        horario = time.strftime("%H:%M:%S")
        # Busca o nome amigável em PT-BR; se não encontrar, usa o código mesmo
        nome = NOMES_GESTOS.get(gesto, gesto)
        print(f"[{horario}] Gesto detectado: {nome}")
        _ultimo_gesto_logado = gesto
        _ultimo_tempo_log    = agora


def buscar_camera_c920():
    """
    Localiza automaticamente a câmera Logitech C920 entre os dispositivos
    disponíveis, usando dois métodos em sequência:

    Método 1 — por nome (requer pygrabber):
        Lista todos os dispositivos de vídeo e procura por "C920" ou
        "HD Pro Webcam" no nome do dispositivo.

    Método 2 — por resolução (fallback):
        Testa os índices 0 a 5 e verifica qual câmera aceita 1920x1080.
        Evita capturar a webcam integrada do notebook (geralmente índice 0).

    Retorna:
        int → índice da câmera encontrada (padrão: 0 se nenhuma for identificada)
    """

    # ── Método 1: busca pelo nome do dispositivo via pygrabber ────────────────
    try:
        from pygrabber.dshow_graph import FilterGraph
        grafo     = FilterGraph()
        dispositivos = grafo.get_input_devices()

        print("Câmeras detectadas:")
        for i, nome in enumerate(dispositivos):
            print(f"  [{i}] {nome}")

        for i, nome in enumerate(dispositivos):
            if "C920" in nome or "c920" in nome.lower() or "HD Pro Webcam" in nome:
                print(f"Logitech C920 encontrada pelo nome no índice {i}: '{nome}'")
                return i

    except ImportError:
        print("pygrabber não instalado — usando detecção por resolução.")
    except Exception as erro:
        print(f"Erro ao listar dispositivos: {erro}")

    # ── Método 2: fallback — testa resolução de cada câmera ──────────────────
    indice_reserva = None  # Guarda câmera HD externa caso Full HD não seja encontrada

    for indice in range(6):  # Testa índices de câmera de 0 até 5
        captura_teste = cv2.VideoCapture(indice, cv2.CAP_DSHOW)

        if not captura_teste.isOpened():
            captura_teste.release()
            continue

        sucesso, _ = captura_teste.read()
        if sucesso:
            # Tenta forçar Full HD e verifica se a câmera aceita
            captura_teste.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            captura_teste.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            largura = captura_teste.get(cv2.CAP_PROP_FRAME_WIDTH)
            altura  = captura_teste.get(cv2.CAP_PROP_FRAME_HEIGHT)
            captura_teste.release()

            print(f"  Índice {indice}: {int(largura)}x{int(altura)}")

            if largura >= 1920 and altura >= 1080:
                # Câmera Full HD confirmada → muito provavelmente é a C920
                print(f"Câmera Full HD encontrada no índice {indice} — assumindo C920.")
                return indice
            elif largura >= 1280 and indice_reserva is None and indice != 0:
                # Câmera HD externa (não a integrada do notebook) como reserva
                indice_reserva = indice
        else:
            captura_teste.release()

    if indice_reserva is not None:
        print(f"C920 não confirmada; usando câmera HD no índice {indice_reserva}.")
        return indice_reserva

    print("C920 não encontrada. Usando índice 0.")
    return 0


def ler_imagem_unicode(caminho):
    """
    Lê um arquivo de imagem suportando caminhos com acentos, espaços e
    outros caracteres especiais — limitação do cv2.imread() padrão no Windows.

    Estratégia:
        Abre o arquivo em modo binário → converte bytes para array NumPy →
        decodifica como imagem colorida BGR com cv2.imdecode().

    Parâmetros:
        caminho (str): caminho completo até o arquivo de imagem

    Retorna:
        numpy.ndarray → imagem BGR carregada
        None          → se o arquivo não existir ou ocorrer algum erro
    """
    try:
        with open(caminho, 'rb') as arquivo:
            bytes_imagem = np.frombuffer(arquivo.read(), dtype=np.uint8)
        return cv2.imdecode(bytes_imagem, cv2.IMREAD_COLOR)
    except Exception as erro:
        print(f"Erro ao abrir arquivo: {caminho} — {erro}")
        return None


def carregar_e_redimensionar_imagem(nome_arquivo, altura_alvo):
    """
    Carrega a imagem ilustrativa do gesto e a redimensiona proporcionalmente
    para ter exatamente a mesma altura do frame da câmera.

    Isso garante que, ao exibir câmera + imagem do gesto lado a lado,
    ambas fiquem perfeitamente alinhadas na vertical.

    Parâmetros:
        nome_arquivo (str): nome do arquivo (ex: "joinha.jpg")
        altura_alvo  (int): altura desejada em pixels (= altura do frame da câmera)

    Retorna:
        numpy.ndarray → imagem redimensionada
        None          → se o arquivo não for encontrado
    """
    caminho_completo = os.path.join(PASTA_SCRIPT, nome_arquivo)
    imagem = ler_imagem_unicode(caminho_completo)

    if imagem is None:
        print(f"Erro: não foi possível carregar '{nome_arquivo}'. Caminho: {caminho_completo}")
        return None

    # Calcula a proporção mantendo o aspecto original (sem distorção)
    proporcao    = altura_alvo / imagem.shape[0]
    largura_nova = int(imagem.shape[1] * proporcao)

    return cv2.resize(imagem, (largura_nova, altura_alvo))


def classificar_gesto(landmarks_mao):
    """
    Analisa os 21 pontos de referência (landmarks) da mão detectados pelo
    MediaPipe e retorna o código do gesto correspondente.

    ─── CONCEITO FUNDAMENTAL ────────────────────────────────────────────────
    O MediaPipe usa coordenadas Y NORMALIZADAS (0.0 a 1.0), onde:
        • valor MENOR = posição mais ALTA na tela
        • valor MAIOR = posição mais BAIXA na tela

    Para saber se um dedo está levantado, comparamos:
        PONTA do dedo (TIP)  vs.  segunda articulação do dedo (PIP)
        Se TIP.y < PIP.y  →  dedo apontado para cima (levantado)
        Se TIP.y > PIP.y  →  dedo dobrado para baixo (fechado)

    ─── LANDMARKS UTILIZADOS ────────────────────────────────────────────────
        THUMB_TIP          → ponta do polegar
        INDEX_FINGER_TIP   → ponta do indicador
        MIDDLE_FINGER_TIP  → ponta do dedo médio
        RING_FINGER_TIP    → ponta do anelar
        PINKY_TIP          → ponta do mínimo

        MIDDLE_FINGER_PIP  → articulação do médio  (referência geral da palma)
        INDEX_FINGER_PIP   → articulação do indicador
        RING_FINGER_PIP    → articulação do anelar
        PINKY_PIP          → articulação do mínimo
    ─────────────────────────────────────────────────────────────────────────

    Parâmetros:
        landmarks_mao → objeto de landmarks retornado pelo MediaPipe Hands

    Retorna:
        str → código do gesto detectado (ex: "JOINHA", "PAZ", "NEUTRO")
    """

    # ── Coordenadas Y das pontas de cada dedo ─────────────────────────────
    y_polegar   = landmarks_mao.landmark[mp_maos.HandLandmark.THUMB_TIP].y
    y_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].y
    y_medio     = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].y
    y_anelar    = landmarks_mao.landmark[mp_maos.HandLandmark.RING_FINGER_TIP].y
    y_minimo    = landmarks_mao.landmark[mp_maos.HandLandmark.PINKY_TIP].y

    # Articulação do médio usada como referência geral da "altura da palma"
    y_articulacao_medio = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_PIP].y

    # ── JOINHA ────────────────────────────────────────────────────────────
    # Polegar acima da palma E todos os outros 4 dedos fechados abaixo da palma
    polegar_levantado    = y_polegar < y_articulacao_medio
    quatro_dedos_fechados = (
        y_indicador > y_articulacao_medio and
        y_medio     > y_articulacao_medio and
        y_anelar    > y_articulacao_medio and
        y_minimo    > y_articulacao_medio
    )
    if polegar_levantado and quatro_dedos_fechados:
        return "JOINHA"

    # ── APONTANDO ─────────────────────────────────────────────────────────
    # Somente o indicador levantado; polegar, médio, anelar e mínimo fechados
    indicador_levantado = y_indicador < y_articulacao_medio
    outros_fechados     = (
        y_medio  > y_articulacao_medio and
        y_anelar > y_articulacao_medio and
        y_minimo > y_articulacao_medio
    )
    polegar_fechado = y_polegar > y_articulacao_medio

    if indicador_levantado and outros_fechados and polegar_fechado:
        return "APONTANDO"

    # ── Articulações individuais (usadas nos gestos abaixo) ───────────────
    y_articulacao_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_PIP].y
    y_articulacao_anelar    = landmarks_mao.landmark[mp_maos.HandLandmark.RING_FINGER_PIP].y
    y_articulacao_minimo    = landmarks_mao.landmark[mp_maos.HandLandmark.PINKY_PIP].y

    # ── MAO ABERTA ────────────────────────────────────────────────────────
    # Todos os 4 dedos levantados acima de suas próprias articulações
    todos_levantados = (
        y_indicador < y_articulacao_indicador and
        y_medio     < y_articulacao_medio     and
        y_anelar    < y_articulacao_anelar    and
        y_minimo    < y_articulacao_minimo
    )
    if todos_levantados:
        return "MAO_ABERTA"

    # ── PUNHO ─────────────────────────────────────────────────────────────
    # Todos os dedos fechados abaixo de suas articulações E polegar fechado
    todos_fechados = (
        y_indicador > y_articulacao_indicador and
        y_medio     > y_articulacao_medio     and
        y_anelar    > y_articulacao_anelar    and
        y_minimo    > y_articulacao_minimo    and
        y_polegar   > y_articulacao_medio
    )
    if todos_fechados:
        return "PUNHO"

    # ── PAZ E AMOR ────────────────────────────────────────────────────────
    # Indicador e médio levantados; anelar, mínimo e polegar fechados
    gesto_paz = (
        y_indicador < y_articulacao_indicador and
        y_medio     < y_articulacao_medio     and
        y_anelar    > y_articulacao_anelar    and
        y_minimo    > y_articulacao_minimo    and
        y_polegar   > y_articulacao_medio
    )
    if gesto_paz:
        return "PAZ"

    # ── HANG LOOSE ────────────────────────────────────────────────────────
    # Polegar e mínimo levantados; indicador, médio e anelar fechados
    gesto_hang_loose = (
        y_polegar   < y_articulacao_medio     and   # polegar levantado
        y_minimo    < y_articulacao_minimo    and   # mínimo levantado
        y_indicador > y_articulacao_indicador and   # indicador fechado
        y_medio     > y_articulacao_medio     and   # médio fechado
        y_anelar    > y_articulacao_anelar          # anelar fechado
    )
    if gesto_hang_loose:
        return "HANG_LOOSE"

    # ── DEDO DO MEIO ──────────────────────────────────────────────────────
    # Somente o dedo médio levantado; todos os outros fechados
    gesto_dedo_meio = (
        y_medio     < y_articulacao_medio     and   # médio levantado
        y_indicador > y_articulacao_indicador and   # indicador fechado
        y_anelar    > y_articulacao_anelar    and   # anelar fechado
        y_minimo    > y_articulacao_minimo    and   # mínimo fechado
        y_polegar   > y_articulacao_medio           # polegar fechado
    )
    if gesto_dedo_meio:
        return "DEDO_MEIO"

    # ── ROCK ──────────────────────────────────────────────────────────────
    # Indicador e mínimo levantados; médio, anelar e polegar fechados
    # (articulações relidas aqui para deixar o bloco explícito e legível)
    y_articulacao_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_PIP].y
    y_articulacao_minimo    = landmarks_mao.landmark[mp_maos.HandLandmark.PINKY_PIP].y

    gesto_rock = (
        y_indicador < y_articulacao_indicador and   # indicador levantado
        y_minimo    < y_articulacao_minimo    and   # mínimo levantado
        y_medio     > y_articulacao_medio     and   # médio fechado
        y_anelar    > y_articulacao_medio     and   # anelar fechado
        y_polegar   > y_articulacao_medio           # polegar fechado
    )
    if gesto_rock:
        return "ROCK"

    # ── NEUTRO ────────────────────────────────────────────────────────────
    # Nenhuma das condições acima foi satisfeita
    return "NEUTRO"


def verificar_gesto_pensando(landmarks_mao, landmarks_rosto, largura_frame, altura_frame):
    """
    Verifica se o usuário está fazendo o gesto de "pensando":
    ponta do indicador próxima ao nariz com o dedo médio fechado.

    Como funciona:
        1. Converte a posição normalizada da ponta do indicador para pixels.
        2. Converte a posição normalizada da ponta do nariz (landmark 4) para pixels.
        3. Calcula a distância euclidiana entre os dois pontos.
        4. Se distância < 50px E médio fechado → gesto de "pensando" detectado.

    Parâmetros:
        landmarks_mao    → landmarks da mão (MediaPipe Hands)
        landmarks_rosto  → landmarks do rosto (MediaPipe FaceMesh)
        largura_frame    → largura do frame em pixels (para desnormalizar)
        altura_frame     → altura do frame em pixels (para desnormalizar)

    Retorna:
        True  → gesto de "pensando" detectado
        False → gesto não detectado ou dados ausentes
    """
    if not landmarks_mao or not landmarks_rosto:
        return False  # Sem mão ou rosto detectados, não há como verificar

    # ── Posição da ponta do indicador em pixels ───────────────────────────
    # As coordenadas do MediaPipe são normalizadas (0.0 a 1.0);
    # multiplicar pela dimensão do frame converte para pixels reais.
    ponta_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP]
    x_indicador = int(ponta_indicador.x * largura_frame)
    y_indicador = int(ponta_indicador.y * altura_frame)

    # ── Posição da ponta do nariz em pixels ───────────────────────────────
    # Landmark 4 do FaceMesh corresponde à ponta do nariz
    ponta_nariz = landmarks_rosto.landmark[4]
    x_nariz = int(ponta_nariz.x * largura_frame)
    y_nariz = int(ponta_nariz.y * altura_frame)

    # ── Distância euclidiana entre indicador e nariz ──────────────────────
    # Fórmula: √( (x2-x1)² + (y2-y1)² )
    distancia = np.sqrt((x_indicador - x_nariz) ** 2 + (y_indicador - y_nariz) ** 2)

    DISTANCIA_MAXIMA = 50  # Limiar em pixels: abaixo disso = "próximo ao nariz"

    # ── Verifica se o dedo médio está fechado ─────────────────────────────
    # Necessário para diferenciar de simplesmente apontar para o nariz
    y_articulacao_medio = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_PIP].y
    y_ponta_medio       = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].y
    medio_fechado = y_ponta_medio > y_articulacao_medio

    if distancia < DISTANCIA_MAXIMA and medio_fechado:
        return True

    return False


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

# Encontra e inicializa a câmera
INDICE_CAMERA = buscar_camera_c920()
captura = cv2.VideoCapture(INDICE_CAMERA, cv2.CAP_DSHOW)
# CAP_DSHOW = backend DirectShow do Windows, mais estável que o padrão

# Tenta forçar resolução Full HD (1920x1080) na câmera
# Se a câmera não suportar, o OpenCV usará a maior resolução disponível
captura.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print(f"Rastreador de Gestos iniciado (câmera índice {INDICE_CAMERA}). Pressione 'q' para sair.")
print("Dica: clique e arraste a janela para movê-la pela tela.\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL — processa frame a frame até o usuário pressionar Q ou ESC
# ─────────────────────────────────────────────────────────────────────────────
while captura.isOpened():

    # Captura um frame da câmera
    # sucesso = False se a câmera parar de responder
    sucesso, frame = captura.read()
    if not sucesso:
        break

    # Espelha o frame horizontalmente (como olhar para um espelho)
    # Torna o movimento mais intuitivo para o usuário
    frame = cv2.flip(frame, 1)

    # Extrai as dimensões do frame capturado
    altura_frame, largura_frame, _ = frame.shape

    # Converte o frame de BGR (padrão OpenCV) para RGB (padrão MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Processa o frame com os detectores do MediaPipe ───────────────────
    resultado_maos  = detector_maos.process(frame_rgb)
    resultado_rosto = detector_rosto.process(frame_rgb)

    # Inicializa o gesto como NEUTRO antes de classificar
    gesto_atual = "NEUTRO"

    # Variáveis que receberão os landmarks detectados (ou ficarão None)
    landmarks_mao_detectada   = None
    landmarks_rosto_detectado = None

    # Extrai os landmarks da primeira mão encontrada (se houver)
    if resultado_maos.multi_hand_landmarks:
        landmarks_mao_detectada = resultado_maos.multi_hand_landmarks[0]

    # Extrai os landmarks do primeiro rosto encontrado (se houver)
    if resultado_rosto.multi_face_landmarks:
        landmarks_rosto_detectado = resultado_rosto.multi_face_landmarks[0]

    # ── Classifica o gesto (apenas se uma mão foi detectada) ──────────────
    if landmarks_mao_detectada:

        # Verifica "PENSANDO" primeiro pois depende também do rosto
        if landmarks_rosto_detectado:
            if verificar_gesto_pensando(
                landmarks_mao_detectada,
                landmarks_rosto_detectado,
                largura_frame,
                altura_frame
            ):
                gesto_atual = "PENSANDO"

        # Se não foi "pensando", classifica pelos outros gestos
        if gesto_atual == "NEUTRO":
            gesto_atual = classificar_gesto(landmarks_mao_detectada)

        # Desenha os pontos e conexões da mão sobre o frame
        mp_desenho.draw_landmarks(
            frame,
            landmarks_mao_detectada,
            mp_maos.HAND_CONNECTIONS,
            mp_desenho.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  # pontos: roxo
            mp_desenho.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2), # linhas: rosa
        )

    # ── Carrega e exibe a imagem ilustrativa do gesto ─────────────────────
    imagem_gesto = carregar_e_redimensionar_imagem(
        IMAGENS_GESTOS[gesto_atual], altura_frame
    )

    if imagem_gesto is not None:
        # Junta o frame da câmera com a imagem do gesto lado a lado (horizontal)
        frame_saida = np.concatenate((frame, imagem_gesto), axis=1)

        # Escreve o nome do gesto em verde no painel direito (sobre a imagem)
        cv2.putText(
            frame_saida,
            f"Gesto: {NOMES_GESTOS.get(gesto_atual, gesto_atual)}",
            (largura_frame + 10, 30),  # posição: início do painel direito + margem
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,          # escala do texto
            (0, 255, 0),  # cor verde em BGR
            2,            # espessura do texto
        )
    else:
        # Fallback: exibe só o frame da câmera com aviso de erro em vermelho
        frame_saida = frame
        cv2.putText(
            frame_saida,
            "ERRO: Imagem nao encontrada!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # vermelho em BGR
            2,
        )

    # ── Redimensiona o frame final para caber na tela ─────────────────────
    # O frame concatenado (câmera + imagem) pode ultrapassar a largura da tela;
    # limitamos a 1280px mantendo a proporção original.
    LARGURA_MAXIMA = 1280
    altura_saida, largura_saida = frame_saida.shape[:2]

    if largura_saida > LARGURA_MAXIMA:
        escala       = LARGURA_MAXIMA / largura_saida
        frame_exibir = cv2.resize(
            frame_saida,
            (LARGURA_MAXIMA, int(altura_saida * escala))
        )
    else:
        frame_exibir = frame_saida

    # Exibe o frame final na janela
    cv2.imshow(NOME_JANELA, frame_exibir)

    # Registra o callback de mouse — pode ser chamado a cada frame sem problema,
    # pois o OpenCV ignora registros repetidos para o mesmo callback
    cv2.setMouseCallback(NOME_JANELA, callback_mouse)

    # Registra o gesto no terminal (somente quando muda)
    logar_gesto(gesto_atual)

    # Aguarda 5ms por uma tecla; encerra se for Q (ord=113) ou ESC (ord=27)
    tecla = cv2.waitKey(5)
    if tecla == ord('q') or tecla == 27:
        break


# =============================================================================
# FINALIZAÇÃO — libera todos os recursos ao encerrar o programa
# =============================================================================

detector_maos.close()    # Encerra o detector de mãos do MediaPipe
detector_rosto.close()   # Encerra o detector de rosto do MediaPipe
captura.release()        # Libera a câmera para outros programas
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV
