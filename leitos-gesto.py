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

import cv2
import mediapipe as mp
import numpy as np
import os
import time


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
NOME_JANELA  = 'Rastreador de Gestos'


# =============================================================================
# ESTADO PARA ARRASTAR A JANELA
# =============================================================================

_posicao_inicial_mouse  = None
_posicao_inicial_janela = None
_arrastando             = False


# =============================================================================
# ESTADO DO LOG NO TERMINAL
# =============================================================================

_ultimo_gesto_logado = None
_ultimo_tempo_log    = 0
INTERVALO_LOG        = 0.5


# =============================================================================
# DICIONÁRIO: código interno → nome amigável
# =============================================================================

NOMES_GESTOS = {
    "JOINHA":     "JOINHA",
    "APONTANDO":  "APONTANDO",
    "NEUTRO":     "NEUTRO",
    "PENSANDO":   "PENSANDO",
    "ROCK":       "ROCK",
    "PUNHO":      "PUNHO",
    "HANG_LOOSE": "HANG LOOSE",
    "PAZ":        "PAZ E AMOR",
    "MAO_ABERTA": "MAO ABERTA",
    "DEDO_MEIO":  "DEDO DO MEIO",
}


# =============================================================================
# DICIONÁRIO: código interno → arquivo de imagem
# =============================================================================

IMAGENS_GESTOS = {
    "JOINHA":     "joinha.jpg",
    "APONTANDO":  "apontando.jpg",
    "NEUTRO":     "neutral.jpg",
    "PENSANDO":   "pensando.jpg",
    "ROCK":       "rock.jpg",
    "PUNHO":      "punho.jpg",
    "HANG_LOOSE": "Hang loose.jpg",
    "PAZ":        "paz.jpg",
    "MAO_ABERTA": "aberta.jpg",
    "DEDO_MEIO":  "meio.jpg",
}


# =============================================================================
# INICIALIZAÇÃO DO MEDIAPIPE
# =============================================================================

mp_desenho = mp.solutions.drawing_utils
mp_maos    = mp.solutions.hands
mp_rosto   = mp.solutions.face_mesh

detector_maos = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

detector_rosto = mp_rosto.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def listar_cameras_disponiveis():
    """
    Testa índices de 0 a 9 com múltiplos backends do OpenCV e retorna
    as câmeras que conseguiram capturar pelo menos um frame com sucesso.

    Tenta obter os nomes reais dos dispositivos via pygrabber.
    Se não estiver instalado, usa nomes genéricos ("Camera 0", etc.).

    Retorna:
        list of (int, str, int, str) → [(índice, nome, backend_id, nome_backend)]
    """
    # Tenta obter nomes reais via pygrabber
    nomes = {}
    try:
        from pygrabber.dshow_graph import FilterGraph
        dispositivos = FilterGraph().get_input_devices()
        for i, nome in enumerate(dispositivos):
            nomes[i] = nome
        print("Dispositivos detectados pelo sistema:")
        for i, nome in nomes.items():
            print(f"  [{i}] {nome}")
        print()
    except Exception:
        pass

    # Backends testados em ordem de preferência
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF,  "Media Foundation"),
        (cv2.CAP_ANY,   "Auto"),
    ]

    cameras = []
    vistos  = set()

    for indice in range(10):
        for backend_id, nome_backend in backends:
            cap = cv2.VideoCapture(indice, backend_id)
            if not cap.isOpened():
                cap.release()
                continue

            sucesso, frame = cap.read()
            cap.release()

            if sucesso and frame is not None and frame.size > 0:
                if indice not in vistos:
                    vistos.add(indice)
                    nome = nomes.get(indice, f"Camera {indice}")
                    cameras.append((indice, nome, backend_id, nome_backend))
                break  # backend funcionou, passa para o próximo índice

    return cameras


def abrir_camera(indice, backend_id):
    """
    Abre a câmera no índice e backend especificados, configura 720p e
    descarta os primeiros frames para estabilizar a imagem.

    Retorna:
        cv2.VideoCapture configurada e pronta para uso
    """
    cap = cv2.VideoCapture(indice, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Descarta frames iniciais — alguns drivers retornam preto nos primeiros frames
    for _ in range(10):
        cap.read()

    return cap


def selecionar_camera():
    """
    Lista as câmeras disponíveis no terminal e pede ao usuário que escolha.
    Valida a entrada e retorna (índice, backend_id).
    """
    print("=" * 55)
    print("  RASTREADOR DE GESTOS — Seleção de Câmera")
    print("=" * 55)
    print("Detectando câmeras disponíveis (aguarde)...\n")

    cameras = listar_cameras_disponiveis()

    if not cameras:
        print("Nenhuma câmera encontrada! Verifique as conexões.")
        raise SystemExit(1)

    print("Câmeras disponíveis:")
    for indice, nome, _, nome_backend in cameras:
        print(f"  [{indice}] {nome}  (backend: {nome_backend})")

    print()
    indices_validos = [str(c[0]) for c in cameras]

    while True:
        escolha = input(f"Digite o número da câmera {indices_validos}: ").strip()
        if escolha in indices_validos:
            indice_escolhido = int(escolha)
            cam = next(c for c in cameras if c[0] == indice_escolhido)
            _, nome_escolhida, backend_id, nome_backend = cam
            print(f"\nUsando: [{indice_escolhido}] {nome_escolhida} ({nome_backend})\n")
            return indice_escolhido, backend_id
        else:
            print(f"Opção inválida. Escolha: {indices_validos}")


def callback_mouse(evento, x, y, flags, parametro):
    """
    Implementa o arraste da janela clicando e segurando (somente Windows).
    """
    global _posicao_inicial_mouse, _posicao_inicial_janela, _arrastando
    import ctypes

    if evento == cv2.EVENT_LBUTTONDOWN:
        _arrastando = True
        ponto = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(ponto))
        _posicao_inicial_mouse = (ponto.x, ponto.y)

        handle    = ctypes.windll.user32.FindWindowW(None, NOME_JANELA)
        retangulo = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(handle, ctypes.byref(retangulo))
        _posicao_inicial_janela = (retangulo.left, retangulo.top)

    elif evento == cv2.EVENT_MOUSEMOVE and _arrastando:
        ponto = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(ponto))

        delta_x = ponto.x - _posicao_inicial_mouse[0]
        delta_y = ponto.y - _posicao_inicial_mouse[1]
        nova_x  = _posicao_inicial_janela[0] + delta_x
        nova_y  = _posicao_inicial_janela[1] + delta_y

        handle = ctypes.windll.user32.FindWindowW(None, NOME_JANELA)
        ctypes.windll.user32.SetWindowPos(handle, None, nova_x, nova_y, 0, 0, 0x0001)

    elif evento == cv2.EVENT_LBUTTONUP:
        _arrastando = False


def logar_gesto(gesto):
    """
    Imprime o gesto no terminal somente quando ele muda.
    """
    global _ultimo_gesto_logado, _ultimo_tempo_log
    agora = time.time()

    if gesto != _ultimo_gesto_logado:
        horario = time.strftime("%H:%M:%S")
        nome    = NOMES_GESTOS.get(gesto, gesto)
        print(f"[{horario}] Gesto detectado: {nome}")
        _ultimo_gesto_logado = gesto
        _ultimo_tempo_log    = agora


def ler_imagem_unicode(caminho):
    """
    Lê uma imagem suportando caminhos com acentos e caracteres especiais.
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
    Carrega a imagem do gesto e a redimensiona para a altura do frame da câmera.
    """
    caminho_completo = os.path.join(PASTA_SCRIPT, nome_arquivo)
    imagem = ler_imagem_unicode(caminho_completo)

    if imagem is None:
        print(f"Erro: não foi possível carregar '{nome_arquivo}'.")
        return None

    proporcao    = altura_alvo / imagem.shape[0]
    largura_nova = int(imagem.shape[1] * proporcao)
    return cv2.resize(imagem, (largura_nova, altura_alvo))


def classificar_gesto(landmarks_mao):
    """
    Analisa os 21 landmarks da mão e retorna o código do gesto.

    Coordenadas Y do MediaPipe são normalizadas (0.0 a 1.0):
        valor menor = posição mais alta na tela
        valor maior = posição mais baixa na tela

    Dedo levantado: TIP.y < PIP.y
    Dedo fechado:   TIP.y > PIP.y
    """

    y_polegar   = landmarks_mao.landmark[mp_maos.HandLandmark.THUMB_TIP].y
    y_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].y
    y_medio     = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].y
    y_anelar    = landmarks_mao.landmark[mp_maos.HandLandmark.RING_FINGER_TIP].y
    y_minimo    = landmarks_mao.landmark[mp_maos.HandLandmark.PINKY_TIP].y

    y_art_medio = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_PIP].y

    # Articulações individuais
    y_art_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_PIP].y
    y_art_anelar    = landmarks_mao.landmark[mp_maos.HandLandmark.RING_FINGER_PIP].y
    y_art_minimo    = landmarks_mao.landmark[mp_maos.HandLandmark.PINKY_PIP].y

    # ── JOINHA ────────────────────────────────────────────────────────────
    if (y_polegar   < y_art_medio and
            y_indicador > y_art_medio and
            y_medio     > y_art_medio and
            y_anelar    > y_art_medio and
            y_minimo    > y_art_medio):
        return "JOINHA"

    # ── APONTANDO ─────────────────────────────────────────────────────────
    if (y_indicador < y_art_medio and
            y_medio   > y_art_medio and
            y_anelar  > y_art_medio and
            y_minimo  > y_art_medio and
            y_polegar > y_art_medio):
        return "APONTANDO"

    # ── MAO ABERTA ────────────────────────────────────────────────────────
    if (y_indicador < y_art_indicador and
            y_medio  < y_art_medio and
            y_anelar < y_art_anelar and
            y_minimo < y_art_minimo):
        return "MAO_ABERTA"

    # ── PUNHO ─────────────────────────────────────────────────────────────
    if (y_indicador > y_art_indicador and
            y_medio   > y_art_medio and
            y_anelar  > y_art_anelar and
            y_minimo  > y_art_minimo and
            y_polegar > y_art_medio):
        return "PUNHO"

    # ── PAZ E AMOR ────────────────────────────────────────────────────────
    if (y_indicador < y_art_indicador and
            y_medio   < y_art_medio and
            y_anelar  > y_art_anelar and
            y_minimo  > y_art_minimo and
            y_polegar > y_art_medio):
        return "PAZ"

    # ── HANG LOOSE ────────────────────────────────────────────────────────
    if (y_polegar   < y_art_medio and
            y_minimo    < y_art_minimo and
            y_indicador > y_art_indicador and
            y_medio     > y_art_medio and
            y_anelar    > y_art_anelar):
        return "HANG_LOOSE"

    # ── DEDO DO MEIO ──────────────────────────────────────────────────────
    if (y_medio     < y_art_medio and
            y_indicador > y_art_indicador and
            y_anelar    > y_art_anelar and
            y_minimo    > y_art_minimo and
            y_polegar   > y_art_medio):
        return "DEDO_MEIO"

    # ── ROCK ──────────────────────────────────────────────────────────────
    if (y_indicador < y_art_indicador and
            y_minimo  < y_art_minimo and
            y_medio   > y_art_medio and
            y_anelar  > y_art_medio and
            y_polegar > y_art_medio):
        return "ROCK"

    return "NEUTRO"


def verificar_gesto_pensando(landmarks_mao, landmarks_rosto, largura_frame, altura_frame):
    """
    Verifica se a ponta do indicador está próxima ao nariz com o médio fechado.
    """
    if not landmarks_mao or not landmarks_rosto:
        return False

    ponta_indicador = landmarks_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP]
    x_indicador = int(ponta_indicador.x * largura_frame)
    y_indicador = int(ponta_indicador.y * altura_frame)

    ponta_nariz = landmarks_rosto.landmark[4]  # landmark 4 = ponta do nariz
    x_nariz = int(ponta_nariz.x * largura_frame)
    y_nariz = int(ponta_nariz.y * altura_frame)

    distancia = np.sqrt((x_indicador - x_nariz) ** 2 + (y_indicador - y_nariz) ** 2)

    y_art_medio   = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_PIP].y
    y_ponta_medio = landmarks_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].y
    medio_fechado = y_ponta_medio > y_art_medio

    return distancia < 50 and medio_fechado


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

# Seleção interativa de câmera no terminal
INDICE_CAMERA, BACKEND_CAMERA = selecionar_camera()
captura = abrir_camera(INDICE_CAMERA, BACKEND_CAMERA)

if not captura.isOpened():
    print(f"ERRO: Não foi possível abrir a câmera {INDICE_CAMERA}.")
    raise SystemExit(1)

print(f"Rastreador de Gestos iniciado (câmera {INDICE_CAMERA}). Pressione 'q' para sair.")
print("Dica: clique e arraste a janela para movê-la pela tela.\n")


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

while captura.isOpened():

    sucesso, frame = captura.read()

    # Frame vazio: câmera pode estar inicializando, tenta de novo
    if not sucesso or frame is None or frame.size == 0:
        print("Aviso: frame vazio. Tentando novamente...")
        time.sleep(0.05)
        continue

    # Espelha o frame (como olhar para um espelho)
    frame = cv2.flip(frame, 1)

    altura_frame, largura_frame, _ = frame.shape

    # Converte BGR → RGB para o MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado_maos  = detector_maos.process(frame_rgb)
    resultado_rosto = detector_rosto.process(frame_rgb)

    gesto_atual               = "NEUTRO"
    landmarks_mao_detectada   = None
    landmarks_rosto_detectado = None

    if resultado_maos.multi_hand_landmarks:
        landmarks_mao_detectada = resultado_maos.multi_hand_landmarks[0]

    if resultado_rosto.multi_face_landmarks:
        landmarks_rosto_detectado = resultado_rosto.multi_face_landmarks[0]

    # ── Classifica o gesto ────────────────────────────────────────────────
    if landmarks_mao_detectada:

        if landmarks_rosto_detectado:
            if verificar_gesto_pensando(
                landmarks_mao_detectada,
                landmarks_rosto_detectado,
                largura_frame,
                altura_frame
            ):
                gesto_atual = "PENSANDO"

        if gesto_atual == "NEUTRO":
            gesto_atual = classificar_gesto(landmarks_mao_detectada)

        mp_desenho.draw_landmarks(
            frame,
            landmarks_mao_detectada,
            mp_maos.HAND_CONNECTIONS,
            mp_desenho.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
            mp_desenho.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        )

    # ── Exibe imagem ilustrativa do gesto ─────────────────────────────────
    imagem_gesto = carregar_e_redimensionar_imagem(
        IMAGENS_GESTOS[gesto_atual], altura_frame
    )

    if imagem_gesto is not None:
        frame_saida = np.concatenate((frame, imagem_gesto), axis=1)
        cv2.putText(
            frame_saida,
            f"Gesto: {NOMES_GESTOS.get(gesto_atual, gesto_atual)}",
            (largura_frame + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        frame_saida = frame
        cv2.putText(
            frame_saida,
            "ERRO: Imagem nao encontrada!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # ── Redimensiona para caber na tela (máx. 1280px de largura) ──────────
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

    cv2.imshow(NOME_JANELA, frame_exibir)
    cv2.setMouseCallback(NOME_JANELA, callback_mouse)

    logar_gesto(gesto_atual)

    tecla = cv2.waitKey(5)
    if tecla == ord('q') or tecla == 27:
        break


# =============================================================================
# FINALIZAÇÃO
# =============================================================================

detector_maos.close()
detector_rosto.close()
captura.release()
cv2.destroyAllWindows()
