import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

import threading
import cv2
import numpy as np
import ai_edge_litert.interpreter as tflite
from A23 import enviar_async

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_SMIP_INT8.tflite"
LABELS_PATH = BASE_DIR / "labels2.txt"
SAVE_DIR = BASE_DIR / "Imagens_Salvas_filtro1"
SAVE_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30            # pedindo 15 nunca passaria de 15; para 20+ é preciso pedir 30
CAMERA_FORMATO = "YUYV"    # [NOVO] "YUYV" (estável no seu setup) ou "MJPG" (deu "Corrupt JPEG data")
MOSTRAR_JANELA = True      # [NOVO] False = sem imshow (para testar se a janela pesa); sair com Ctrl+C

WINDOW_NAME = "Deteccao - Logitech USB + TFLite"
INTERVALO_SALVAMENTO = 2.5
LIMIAR_LOCAL = 0.80
CLASSES_IGNORADAS = {"Manhole", "No Pothole"}
PERFIL_INTERVALO = 10.0
DIVISOR_ENTRADA = 1.0   # [NOVO] a cada quantos segundos imprime o tempo médio por etapa

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Modelo TFLite não encontrado: {MODEL_PATH}")
if not LABELS_PATH.is_file():
    raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {LABELS_PATH}")

with LABELS_PATH.open("r", encoding="utf-8") as arquivo:
    labels = [linha.strip() for linha in arquivo if linha.strip()]

interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_height, input_width = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]
in_scale, in_zp = input_details[0]["quantization"]
out_scale, out_zp = output_details[0]["quantization"]
print(f"[MODELO] entrada: {input_dtype.__name__} q={input_details[0]['quantization']} | "
      f"saída: {output_dtype.__name__} q={output_details[0]['quantization']}")

# Tabela 0-255 -> valor que o modelo espera (só se a entrada for quantizada)
if input_dtype != np.float32:
    info = np.iinfo(input_dtype)
    x = np.arange(256, dtype=np.float32) / DIVISOR_ENTRADA
    lut = np.clip(np.round(x / in_scale + in_zp), info.min, info.max).astype(input_dtype)
else:
    lut = None

class CameraThread:
    def __init__(self, cam):
        self.cam = cam
        self.frame = None
        self.lock = threading.Lock()
        self.novo = threading.Event()
        self.rodando = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.rodando:
            ok, f = self.cam.read()
            if ok and f is not None:
                with self.lock:
                    self.frame = f
                    self.novo.set()
            else:
                time.sleep(0.01)

    def read(self):
        if not self.novo.wait(timeout=1.0):
            return False, None
        with self.lock:
            f = self.frame
            self.novo.clear()
        return True, f

    def parar(self):
        self.rodando = False
        self.thread.join(timeout=1.0)

def travar_framerate(indice):
    """Desliga a 'prioridade de exposição automática' da Logitech.

    Com ela ligada, a câmera derruba o FPS (7.5 é um dos degraus) quando o
    ambiente está escuro, para ganhar tempo de exposição. Precisa do v4l-utils:
        sudo apt install -y v4l-utils
    O ajuste se perde quando a câmera é desplugada; por isso é refeito aqui.
    """
    dispositivo = f"/dev/video{indice}"
    for nome in ("exposure_dynamic_framerate", "exposure_auto_priority"):
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", dispositivo, f"--set-ctrl={nome}=0"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("[CAMERA] v4l2-ctl indisponível (sudo apt install v4l-utils); "
                  "o FPS pode cair no escuro.", flush=True)
            return
        if r.returncode == 0:
            print(f"[CAMERA] {nome}=0 aplicado (FPS fixo).", flush=True)
            return
    print(f"[CAMERA] Não achei o controle de FPS dinâmico; veja 'v4l2-ctl -d {dispositivo} -l'.",
          flush=True)


def fourcc_texto(valor):
    """Converte o código numérico do formato em texto (ex.: MJPG, YUYV)."""
    v = int(valor)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


# Logitech UVC: OpenCV recebe os quadros em BGR pelo driver V4L2.
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not camera.isOpened():
    camera.release()
    raise RuntimeError(
        f"Não foi possível abrir a câmera no índice {CAMERA_INDEX}. "
        "Confira /dev/video* e tente CAMERA_INDEX=1."
    )

travar_framerate(CAMERA_INDEX)

# O FOURCC precisa ser definido ANTES da resolução e do FPS.
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FORMATO))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
leitor = CameraThread(camera)


print(
    f"Logitech USB iniciada: "
    f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
    f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
    f"formato: {fourcc_texto(camera.get(cv2.CAP_PROP_FOURCC))} | "
    f"FPS pedido ao driver: {camera.get(cv2.CAP_PROP_FPS):.1f}. "
    + ("Pressione 'q' para sair." if MOSTRAR_JANELA else "Pressione Ctrl+C para sair.")
)

if MOSTRAR_JANELA:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

ultimo_salvamento = 0.0
tempo_anterior = time.time()

# [NOVO] Perfil de tempo: média por etapa a cada PERFIL_INTERVALO segundos
perfil = {"leitura": 0.0, "pre": 0.0, "ia": 0.0, "tela": 0.0}
perfil_frames = 0
perfil_inicio = time.time()

try:
    while True:
        t0 = time.time()
        ok, frame_bgr = leitor.read()
        t_camera = time.time() - t0

        if not ok or frame_bgr is None:
            print("\n[FALHA] Sem quadro da câmera; tentando novamente...", flush=True)
            time.sleep(0.2)
            continue

        # O modelo recebe RGB (como recebia da Picamera2); a webcam fornece BGR.
        t_pre0 = time.time()
        frame_small = cv2.resize(
            frame_bgr, (input_width, input_height), interpolation=cv2.INTER_AREA
        )
        if lut is None:
            input_data = np.expand_dims(frame_small, axis=0).astype(input_dtype, copy=False)
        else:
            input_data = lut[frame_small][np.newaxis]
        t_pre = time.time() - t_pre0

        t1 = time.time()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        t_inferencia = time.time() - t1

        saida = interpreter.get_tensor(output_details[0]["index"]).flatten()
        if output_dtype == np.float32:
            probabilities = saida
        else:
            probabilities = (saida.astype(np.float32) - out_zp) * out_scale
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        nome_classe = labels[class_id] if class_id < len(labels) else "Desconhecido"

        agora = time.time()
        dt = agora - tempo_anterior
        fps = 1.0 / dt if dt > 0 else 0.0
        tempo_anterior = agora

        print(
            f"Classe: {nome_classe} | Confiança: {confidence:.2%} | Loop: {fps:.1f} FPS | "
            f"Câmera: {t_camera:.3f}s | IA: {t_inferencia:.3f}s   ",
            end="\r", flush=True
        )

        if (
            nome_classe not in CLASSES_IGNORADAS
            and confidence >= LIMIAR_LOCAL
            and agora - ultimo_salvamento >= INTERVALO_SALVAMENTO
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = SAVE_DIR / f"detected_{timestamp}.jpg"
            if cv2.imwrite(str(filename), frame_bgr):
                enviar_async(str(filename))
                print(f"\n[ALERTA] Imagem salva e encaminhada: {filename}")
                ultimo_salvamento = agora

        t_tela0 = time.time()
        sair = False
        if MOSTRAR_JANELA:
            cv2.imshow(WINDOW_NAME, frame_bgr)
            sair = (cv2.waitKey(1) & 0xFF) == ord("q")
        t_tela = time.time() - t_tela0

        # [NOVO] acumula o perfil e imprime a média a cada PERFIL_INTERVALO
        perfil["leitura"] += t_camera
        perfil["pre"] += t_pre
        perfil["ia"] += t_inferencia
        perfil["tela"] += t_tela
        perfil_frames += 1
        agora2 = time.time()
        if agora2 - perfil_inicio >= PERFIL_INTERVALO:
            n = perfil_frames
            ms = {k: 1000.0 * v / n for k, v in perfil.items()}
            total_ms = 1000.0 * (agora2 - perfil_inicio) / n
            outros = total_ms - sum(ms.values())   # salvar imagem, print, threads em segundo plano
            print(
                f"\n[PERFIL] {n / (agora2 - perfil_inicio):.1f} FPS | "
                f"leitura {ms['leitura']:.0f} ms | pre-proc {ms['pre']:.0f} ms | "
                f"IA {ms['ia']:.0f} ms | tela {ms['tela']:.0f} ms | outros {outros:.0f} ms",
                flush=True,
            )
            perfil = {k: 0.0 for k in perfil}
            perfil_frames = 0
            perfil_inicio = agora2

        if sair:
            break

except KeyboardInterrupt:
    pass
finally:
    leitor.parar()
    cv2.destroyAllWindows()
    print("\nCâmera USB encerrada.")