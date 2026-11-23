# A23- Confirmação na nuvem (Azure Custom Vision) + GPS + envio ao Google Drive
#
# Fluxo:
#   A11 salva o frame -> enviar_async() -> fila -> worker -> analisar_imagem()
#   analisar_imagem(): Azure detecta -> desenha caixa + faixa de texto na imagem
#   -> grava em Fila_Envio -> rclone move para o Drive (e apaga local se subiu)
import os
import queue
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import pynmea2
import serial
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
ENDPOINT = ""
# Use variável de ambiente (e gere uma chave nova no portal do Azure):
#   export AZURE_PREDICTION_KEY="sua_chave"
PREDICTION_KEY = os.environ.get("AZURE_PREDICTION_KEY", "")
PROJECT_ID = ""
PUBLISHED_NAME = "ModeloDetector3.0"

LIMIAR_NUVEM = 0.80      # confiança mínima para considerar buraco confirmado
LIMIAR_DESENHO = 0.75    # caixas com probabilidade menor que isso não são desenhadas

# "gdrive" = nome do remote criado no `rclone config`; "Buracos" = pasta no Drive
DRIVE_DESTINO = "Meu_Drive:Buracos"

BASE_DIR = Path(__file__).resolve().parent
PASTA_ENVIO = BASE_DIR / "Fila_Envio"   # só fica aqui o que ainda não subiu
PASTA_ENVIO.mkdir(exist_ok=True)

FONTE = cv2.FONT_HERSHEY_SIMPLEX

predictor = CustomVisionPredictionClient(
    ENDPOINT, ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
)

# ----------------------------------------------------------------------------
# GPS em segundo plano (mantém só a última posição válida)
# ----------------------------------------------------------------------------
_posicao = (None, None)
_ultima_atualizacao_gps = 0.0
_lock = threading.Lock()

GPS_PORTA = "/dev/serial0"
GPS_BAUDRATE = 9600
GPS_IDADE_MAXIMA = 3.0


def _loop_gps():
    global _posicao, _ultima_atualizacao_gps

    while True:
        try:
            print(f"[GPS] Tentando conectar em {GPS_PORTA}...", flush=True)

            with serial.Serial(
                GPS_PORTA,
                baudrate=GPS_BAUDRATE,
                timeout=1
            ) as ser:

                print("[GPS] Porta serial aberta.", flush=True)

                while True:
                    linha = ser.readline().decode(
                        "ascii", errors="ignore"
                    ).strip()

                    if not linha:
                        continue

                    if linha.startswith(("$GPRMC", "$GNRMC")):
                        try:
                            msg = pynmea2.parse(linha)

                            if msg.status == "A":
                                latitude = round(msg.latitude, 6)
                                longitude = round(msg.longitude, 6)

                                with _lock:
                                    _posicao = (latitude, longitude)
                                    _ultima_atualizacao_gps = time.monotonic()

                        except pynmea2.ParseError:
                            print("[GPS] Erro ao interpretar NMEA.", flush=True)

        except Exception as e:
            print(f"[GPS] Erro de comunicacao: {e}", flush=True)
            time.sleep(2)


def capturar_coordenadas_gps():
    with _lock:
        idade = time.monotonic() - _ultima_atualizacao_gps

        if idade > GPS_IDADE_MAXIMA:
            return None, None

        return _posicao


threading.Thread(target=_loop_gps, daemon=True).start()

# ----------------------------------------------------------------------------
# Desenho: caixa(s) + faixa de texto dentro da própria imagem
# ----------------------------------------------------------------------------
def desenhar_resultado(imagem_bgr, predicoes, top, linhas_texto):
    """Desenha as caixas e acrescenta uma faixa de texto embaixo da imagem.

    Retorna uma nova imagem (BGR). As coordenadas do Azure vêm normalizadas (0 a 1).
    Obs.: cv2.putText não desenha acentos, por isso o texto da faixa é sem acento.
    """
    img = imagem_bgr.copy()
    h, w = img.shape[:2]

    for p in predicoes:
        if p.probability < LIMIAR_DESENHO:
            continue
        b = p.bounding_box
        x1, y1 = int(b.left * w), int(b.top * h)
        x2, y2 = int((b.left + b.width) * w), int((b.top + b.height) * h)

        cor = (0, 0, 255) if p is top else (0, 200, 255)  # BGR: vermelho / laranja
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)

        rotulo = f"{p.tag_name} {p.probability * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(rotulo, FONTE, 0.5, 1)
        y_txt = max(y1, th + 8)  # evita o rótulo sair pelo topo
        cv2.rectangle(img, (x1, y_txt - th - 8), (x1 + tw + 6, y_txt), cor, -1)
        cv2.putText(img, rotulo, (x1 + 3, y_txt - 4), FONTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Faixa escura embaixo com os dados (tudo fica em um único arquivo de imagem)
    altura_linha = 24
    altura_faixa = altura_linha * len(linhas_texto) + 10
    img = cv2.copyMakeBorder(img, 0, altura_faixa, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    for i, texto in enumerate(linhas_texto):
        y = h + 8 + altura_linha * (i + 1) - 6
        cv2.putText(img, texto, (8, y), FONTE, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ----------------------------------------------------------------------------
# Envio ao Google Drive (rclone) e limpeza
# ----------------------------------------------------------------------------
def enviar_para_drive():
    """Envia os arquivos finalizados para o Google Drive."""

    try:
        r = subprocess.run(
            ["rclone", "move", str(PASTA_ENVIO), DRIVE_DESTINO],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if r.returncode == 0:
            print("[DRIVE] Operacao concluida.", flush=True)

            if r.stdout.strip():
                print(r.stdout.strip(), flush=True)

        else:
            print(
                f"[DRIVE] Falha: {r.stderr.strip()[:500]}",
                flush=True
            )

    except FileNotFoundError:
        print("[DRIVE] rclone nao instalado.", flush=True)

    except subprocess.TimeoutExpired:
        print(
            "[DRIVE] Tempo esgotado. Arquivos permanecem na fila.",
            flush=True
        )


def limpar_antigos(pasta, horas=1, padrao="*.jpg"):
    """Apaga arquivos mais velhos que `horas` (sobras de execuções interrompidas)."""
    limite = time.time() - horas * 3600
    for f in Path(pasta).glob(padrao):
        try:
            if f.stat().st_mtime < limite:
                f.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Análise na nuvem
# ----------------------------------------------------------------------------
def analisar_imagem(caminho_imagem, latitude=None, longitude=None):
    caminho = Path(caminho_imagem)
    try:
        image_data = caminho.read_bytes()
        results = predictor.detect_image(PROJECT_ID, PUBLISHED_NAME, image_data)

        print("\n--- Resultados Azure Custom Vision (Detecção de Objetos) ---", flush=True)
        for p in results.predictions:
            b = p.bounding_box
            print(
                f"{p.tag_name}: {p.probability * 100:.2f}% "
                f"[Caixa: x={b.left:.2f}, y={b.top:.2f}, w={b.width:.2f}, h={b.height:.2f}]",
                flush=True,
            )

        if not results.predictions:
            return None

        top = max(results.predictions, key=lambda p: p.probability)
        if top.probability < LIMIAR_NUVEM:
            return None

        agora = datetime.now()
        timestamp = agora.strftime("%Y%m%d_%H%M%S")
        lat_txt = f"{latitude:.6f}" if latitude is not None else "semGPS"
        lon_txt = f"{longitude:.6f}" if longitude is not None else "semGPS"

        linhas = [
            f"{top.tag_name} - {top.probability * 100:.1f}%",
            f"Data/hora: {agora.strftime('%d/%m/%Y %H:%M:%S')}",
            f"Latitude: {lat_txt}   Longitude: {lon_txt}",
        ]

        original = cv2.imread(str(caminho))
        if original is None:
            print(f"[A22] Não consegui abrir a imagem: {caminho}", flush=True)
            return None

        final = desenhar_resultado(original, results.predictions, top, linhas)

        # Os dados também vão no nome do arquivo, facilitando buscar no Drive
        tag = top.tag_name.replace(" ", "")
        nome = f"buraco_{timestamp}_{tag}_{top.probability * 100:.0f}_{lat_txt}_{lon_txt}.jpg"
        cv2.imwrite(str(PASTA_ENVIO / nome), final, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[ALERTA NUVEM] Confirmado: {nome}", flush=True)

        enviar_para_drive()
        return top.tag_name, top.probability, latitude, longitude

    finally:
        # O frame original já cumpriu seu papel; não deixa acumular no Raspberry
        try:
            caminho.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Fila em segundo plano (o loop da câmera nunca espera pela rede)
# ----------------------------------------------------------------------------
_fila = queue.Queue(maxsize=5)


def _worker():
    print("[A23] worker iniciado", flush=True)
    while True:
        caminho, lat, lon = _fila.get()
        try:
            analisar_imagem(caminho, lat, lon)
        except Exception:
            traceback.print_exc()
        finally:
            _fila.task_done()


threading.Thread(target=_worker, daemon=True).start()


def enviar_async(caminho):
    lat, lon = capturar_coordenadas_gps()  # posição do instante da captura
    try:
        _fila.put_nowait((caminho, lat, lon))
    except queue.Full:
        print("[FILA] Cheia, imagem descartada", flush=True)
        try:
            Path(caminho).unlink()
        except OSError:
            pass
