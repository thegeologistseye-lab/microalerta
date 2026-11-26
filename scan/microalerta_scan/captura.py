"""Captura del disco en una cuadrícula de campos.

El programa mueve la platina con los dos motores de las perillas, saca en cada campo una foto
con luz azul (_F, Rojo de Nilo) y otra con luz blanca (_W), y guarda la posición de cada campo
en campos.csv, que `scan --posiciones` usa para contar una sola vez las partículas de los solapes.

Coordenadas: x_mm, y_mm = punto del disco que queda en el centro de la imagen, con el (0, 0) en
el centro del disco y los ejes de la imagen (x a la derecha, y hacia abajo). La relación con los
pasos del motor se fija en captura.json (pasos por mm y signo de cada eje) y se mide con
`calibrar-platina`.

Juego de la perilla y los engranajes: el programa llega a cada campo siempre avanzando en +x y +y.
Si tiene que retroceder, se pasa `juego_mm` más atrás y vuelve a avanzar.
"""
import csv
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2


# ------------------------------------------------------------------ configuración
@dataclass
class ConfigCaptura:
    diametro_mm: float = 21.0          # diámetro efectivo del depósito (círculo interior del empaque)
    margen_mm: float = 0.3             # se fotografía un poco más allá del borde del círculo
    campo_ancho_mm: float = 2.85       # campo de la cámara con el objetivo 4× y la lente 0,5×
    campo_alto_mm: float = 2.14
    solape: float = 0.15               # fracción del campo que se repite entre vecinos (10–20 %)
    um_por_px: float = 1.1             # escala de la cámara (se mide con el micrómetro)
    pasos_mm_x: float = 6144 / 80      # 2.048 pasos × 3 (engranajes) por 80 mm de recorrido
    pasos_mm_y: float = 6144 / 35
    signo_x: int = 1                   # +1 si al avanzar el motor el centro de la foto avanza sobre el disco
                                       # (hacia la derecha de la imagen en x, hacia abajo en y); si no, −1
    signo_y: int = 1
    juego_mm: float = 0.3              # sobrepaso para llegar siempre desde el mismo lado
    espera_s: float = 0.6              # tiempo para que la platina deje de vibrar antes de la foto
    modos: list = field(default_factory=lambda: ["F", "W"])
    luces: str = "arduino"             # arduino (el programa cambia la luz) o manual (dos pasadas)
    brillo_azul: int = 255             # PWM 0–255 del LED azul
    brillo_blanca: int = 120
    camara_ancho: int = 2592
    camara_alto: int = 1944
    exposicion: float = None           # valor propio de la cámara; None = no tocar
    ganancia: float = None
    descartar_cuadros: int = 4         # cuadros que se botan antes de cada foto (la cámara entrega cuadros viejos)
    # modo manual con celular
    campo_celular_mm: float = 4.5      # diámetro del campo visto por el ocular (número de campo ÷ aumento)
    separacion_celular: float = 1.05   # separación entre centros = diámetro × esto (los campos no se tocan)
    vernier_cero_x: float = 0.0        # lectura del vernier con el centro del disco en el centro del campo
    vernier_cero_y: float = 0.0
    signo_vernier_x: int = 1
    signo_vernier_y: int = 1

    @classmethod
    def cargar(cls, ruta=None, **cambios):
        datos = {}
        if ruta:
            with open(ruta, encoding="utf-8") as fh:
                datos = json.load(fh)
        validos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        validos.update({k: v for k, v in cambios.items() if v is not None})
        return cls(**validos)


# ------------------------------------------------------------------ planes
def _rect_toca_circulo(cx, cy, w, h, r):
    dx = max(abs(cx) - w / 2, 0.0)
    dy = max(abs(cy) - h / 2, 0.0)
    return dx * dx + dy * dy < r * r


def plan_cuadricula(cfg):
    """Campos con solape que cubren el círculo. Filas de arriba hacia abajo, cada fila de
    izquierda a derecha (siempre en el mismo sentido, por el juego)."""
    r = cfg.diametro_mm / 2 + cfg.margen_mm
    paso_x = cfg.campo_ancho_mm * (1 - cfg.solape)
    paso_y = cfg.campo_alto_mm * (1 - cfg.solape)
    nx = int(math.ceil((2 * r - cfg.campo_ancho_mm) / paso_x)) + 1
    ny = int(math.ceil((2 * r - cfg.campo_alto_mm) / paso_y)) + 1
    x0 = -(nx - 1) * paso_x / 2
    y0 = -(ny - 1) * paso_y / 2
    campos, n = [], 0
    for fi in range(ny):
        y = y0 + fi * paso_y
        for co in range(nx):
            x = x0 + co * paso_x
            if _rect_toca_circulo(x, y, cfg.campo_ancho_mm, cfg.campo_alto_mm, r):
                n += 1
                campos.append({"campo": f"campo{n:03d}", "fila": fi + 1, "col": co + 1,
                               "x_mm": round(x, 4), "y_mm": round(y, 4)})
    return campos


def plan_manual(cfg):
    """Grilla de campos circulares (celular por el ocular) que no se tocan, dentro del disco."""
    d = cfg.campo_celular_mm
    s = d * cfg.separacion_celular
    r_util = cfg.diametro_mm / 2 - d / 2
    k = int(math.floor(r_util / s))
    campos, n = [], 0
    for fi, j in enumerate(range(-k, k + 1)):
        for co, i in enumerate(range(-k, k + 1)):
            x, y = i * s, j * s
            if math.hypot(x, y) <= r_util + 1e-9:
                n += 1
                campos.append({"campo": f"campo{n:03d}", "fila": fi + 1, "col": co + 1,
                               "x_mm": round(x, 4), "y_mm": round(y, 4),
                               "vernier_x": round(cfg.vernier_cero_x + cfg.signo_vernier_x * x, 2),
                               "vernier_y": round(cfg.vernier_cero_y + cfg.signo_vernier_y * y, 2)})
    return campos


def cobertura_manual(cfg, campos):
    area_campos = len(campos) * math.pi * (cfg.campo_celular_mm / 2) ** 2
    return area_campos / (math.pi * (cfg.diametro_mm / 2) ** 2)


def guardar_plan(campos, ruta):
    cols = list(campos[0].keys())
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(campos)


def leer_plan(ruta):
    with open(ruta, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k in ("x_mm", "y_mm"):
            f[k] = float(f[k])
    return filas


# ------------------------------------------------------------------ platinas
class PlatinaSerial:
    """Arduino Nano con firmware/microalerta_platina. Protocolo de texto, una orden por línea:
    M x y (ir a pasos absolutos) · Z (cero aquí) · P (posición) · L azul blanca (luces 0–255)."""

    def __init__(self, puerto, baudios=115200, espera_respuesta_s=60):
        import serial  # pyserial
        self.s = serial.Serial(puerto, baudios, timeout=espera_respuesta_s)
        time.sleep(2.0)                     # el Nano se reinicia al abrir el puerto
        self.s.reset_input_buffer()
        r = self.orden("?")
        if not r.startswith("MICROALERTA"):
            raise IOError(f"El puerto {puerto} no responde como la platina de MicroAlerta: {r!r}")

    def orden(self, txt):
        self.s.write((txt + "\n").encode("ascii"))
        r = self.s.readline().decode("ascii", "replace").strip()
        if not r:
            raise IOError(f"La platina no respondió a «{txt}»")
        if r.startswith("ERR"):
            raise IOError(f"La platina respondió {r} a «{txt}»")
        return r

    def ir_a_pasos(self, px, py):
        self.orden(f"M {int(px)} {int(py)}")

    def cero(self):
        self.orden("Z")

    def luces(self, azul, blanca):
        self.orden(f"L {int(azul)} {int(blanca)}")

    def cerrar(self):
        try:
            self.luces(0, 0)
        finally:
            self.s.close()


class PlatinaSimulada:
    """Platina de mentira para la prueba: tiene juego (zona muerta) y un pequeño error al azar."""

    def __init__(self, cfg, juego_real_mm=0.08, error_mm=0.004, semilla=0):
        self.cfg = cfg
        self.juego = juego_real_mm
        self.error = error_mm
        self.rng = np.random.default_rng(semilla)
        self.cmd = np.zeros(2)               # mm según los pasos enviados
        self.real = np.zeros(2)              # mm donde quedó de verdad
        self.luz = (0, 0)
        self.ruido = np.zeros(2)

    def ir_a_pasos(self, px, py):
        nuevo = np.array([px / self.cfg.pasos_mm_x, py / self.cfg.pasos_mm_y])
        for k in range(2):
            # modelo de juego: la platina sigue al motor solo cuando este empuja fuera de la zona muerta
            if nuevo[k] > self.real[k] + self.juego / 2:
                self.real[k] = nuevo[k] - self.juego / 2
            elif nuevo[k] < self.real[k] - self.juego / 2:
                self.real[k] = nuevo[k] + self.juego / 2
        self.cmd = nuevo
        self.ruido = self.rng.normal(0, self.error, 2)

    def posicion_real_mm(self):
        """Punto del disco en el centro de la imagen (ejes de la imagen)."""
        return (self.cfg.signo_x * (self.real[0] + self.juego / 2) + self.ruido[0],
                self.cfg.signo_y * (self.real[1] + self.juego / 2) + self.ruido[1])

    def cero(self):
        self.cmd[:] = 0
        self.real[:] = -self.juego / 2       # como si se hubiera llegado avanzando
        self.ruido = np.zeros(2)

    def luces(self, azul, blanca):
        self.luz = (azul, blanca)

    def cerrar(self):
        pass


# ------------------------------------------------------------------ cámaras
class CamaraOpenCV:
    """Cámara USB de ocular (UVC). Fija resolución, exposición y ganancia si la cámara lo permite."""

    def __init__(self, indice, cfg):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(int(indice), backend)
        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir la cámara {indice}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camara_ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camara_alto)
        self.avisos = []
        if cfg.exposicion is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if os.name != "nt" else 0)
            if not self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposicion):
                self.avisos.append("La cámara no aceptó la exposición fija: fíjela en su aplicación.")
        if cfg.ganancia is not None and not self.cap.set(cv2.CAP_PROP_GAIN, cfg.ganancia):
            self.avisos.append("La cámara no aceptó la ganancia fija.")
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.descartar = cfg.descartar_cuadros

    def capturar(self):
        for _ in range(self.descartar):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok or img is None:
            raise IOError("La cámara no entregó imagen")
        return img

    def cerrar(self):
        self.cap.release()


class CamaraSimulada:
    """Fotografía la escena simulada en la posición real de la platina simulada."""

    def __init__(self, escena, platina, cfg, ancho_px, alto_px):
        self.escena, self.platina, self.cfg = escena, platina, cfg
        self.w, self.h = ancho_px, alto_px

    def capturar(self):
        x, y = self.platina.posicion_real_mm()
        modo = "F" if self.platina.luz[0] > 0 else "W"
        return self.escena.fotografiar(x, y, self.w, self.h, self.cfg.um_por_px, modo)

    def cerrar(self):
        pass


# ------------------------------------------------------------------ recorrido
class Recorrido:
    """Lleva la platina a cada campo llegando siempre con el motor avanzando (pasos crecientes), igual
    que al fijar el cero. Así el juego queda siempre del mismo lado, sea cual sea el signo de cada eje:
    si el siguiente punto está "atrás" en pasos, se pasa juego_mm más allá y vuelve avanzando."""

    def __init__(self, platina, cfg):
        self.p, self.cfg = platina, cfg
        self.actual = None                    # mm (coordenadas de la imagen) del último punto
        self.pasos = None                     # pasos del último punto

    def _pasos(self, x, y):
        return (round(self.cfg.signo_x * x * self.cfg.pasos_mm_x),
                round(self.cfg.signo_y * y * self.cfg.pasos_mm_y))

    def ir(self, x, y):
        destino = self._pasos(x, y)
        sobra = (round(self.cfg.juego_mm * self.cfg.pasos_mm_x), round(self.cfg.juego_mm * self.cfg.pasos_mm_y))
        previo = [d - s if self.pasos is None or d < self.pasos[k] else d
                  for k, (d, s) in enumerate(zip(destino, sobra))]
        if tuple(previo) != destino:
            self.p.ir_a_pasos(*previo)
        self.p.ir_a_pasos(*destino)
        self.pasos, self.actual = destino, (x, y)


def _luz(platina, cfg, modo):
    if modo == "F":
        platina.luces(cfg.brillo_azul, 0)
    elif modo == "W":
        platina.luces(0, cfg.brillo_blanca)
    else:
        platina.luces(0, 0)


def capturar(campos, platina, camara, cfg, carpeta, pedir=input, avisar=print):
    """Recorre los campos y guarda campoNNN_F.jpg / campoNNN_W.jpg, campos.csv y captura.json."""
    os.makedirs(carpeta, exist_ok=True)
    rec = Recorrido(platina, cfg)
    t0 = time.time()
    calidad = [cv2.IMWRITE_JPEG_QUALITY, 95]

    def foto(c, modo):
        img = camara.capturar()
        cv2.imwrite(os.path.join(carpeta, f"{c['campo']}_{modo}.jpg"), img, calidad)

    if cfg.luces == "arduino":
        for i, c in enumerate(campos, 1):
            rec.ir(c["x_mm"], c["y_mm"])
            time.sleep(cfg.espera_s)
            for modo in cfg.modos:
                _luz(platina, cfg, modo)
                time.sleep(0.15 if cfg.espera_s > 0 else 0)
                foto(c, modo)
            avisar(f"  {c['campo']} ({i}/{len(campos)})  x={c['x_mm']:+.2f} y={c['y_mm']:+.2f} mm")
        _luz(platina, cfg, None)
    else:
        for modo in cfg.modos:
            nombre = "AZUL (con el filtro naranjo puesto)" if modo == "F" else "BLANCA"
            pedir(f"Encienda la luz {nombre}, apague las demás y presione Enter… ")
            rec.pasos = rec.actual = None     # vuelve al inicio llegando con el motor avanzando
            for i, c in enumerate(campos, 1):
                rec.ir(c["x_mm"], c["y_mm"])
                time.sleep(cfg.espera_s)
                foto(c, modo)
                avisar(f"  {modo} {c['campo']} ({i}/{len(campos)})")
    rec.ir(0.0, 0.0)                          # deja la platina en el centro
    guardar_plan(campos, os.path.join(carpeta, "campos.csv"))
    info = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "n_campos": len(campos),
            "duracion_s": round(time.time() - t0, 1), "modo": "motores", "configuracion": asdict(cfg)}
    with open(os.path.join(carpeta, "captura.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info


# ------------------------------------------------------------------ cuadrícula (mosaico)
def armar_cuadricula(carpeta, modo="F", um_por_px=None, ancho_max_px=3000, diametro_mm=None,
                     campo_circular_mm=None, salida=None):
    """Pega las fotos de cada campo en su posición y dibuja el número de cada uno.
    Devuelve la ruta de la imagen. Sirve para revisar que se cubrió todo el disco."""
    campos = leer_plan(os.path.join(carpeta, "campos.csv"))
    info = {}
    ruta_info = os.path.join(carpeta, "captura.json")
    if os.path.exists(ruta_info):
        with open(ruta_info, encoding="utf-8") as fh:
            info = json.load(fh).get("configuracion", {})
    um_por_px = um_por_px or info.get("um_por_px")
    diametro_mm = diametro_mm or info.get("diametro_mm")
    fotos = []
    for c in campos:
        ruta = os.path.join(carpeta, f"{c['campo']}_{modo}.jpg")
        if os.path.exists(ruta):
            fotos.append((c, ruta))
    if not fotos:
        raise FileNotFoundError(f"No hay fotos _{modo} en {carpeta}")
    img0 = cv2.imread(fotos[0][1])
    h0, w0 = img0.shape[:2]
    if campo_circular_mm and not um_por_px:
        um_por_px = campo_circular_mm * 1000 / min(h0, w0)
    if not um_por_px:
        raise ValueError("Falta la escala (µm por píxel) para armar la cuadrícula")
    wmm, hmm = w0 * um_por_px / 1000, h0 * um_por_px / 1000
    xs = [c["x_mm"] for c, _ in fotos]
    ys = [c["y_mm"] for c, _ in fotos]
    xmin, xmax = min(xs) - wmm / 2, max(xs) + wmm / 2
    ymin, ymax = min(ys) - hmm / 2, max(ys) + hmm / 2
    esc = ancho_max_px / (xmax - xmin)           # px de la cuadrícula por mm
    W, H = int(math.ceil((xmax - xmin) * esc)), int(math.ceil((ymax - ymin) * esc))
    lienzo = np.zeros((H, W, 3), np.uint8)
    tw, th = max(1, int(round(wmm * esc))), max(1, int(round(hmm * esc)))
    for c, ruta in fotos:
        im = cv2.resize(cv2.imread(ruta), (tw, th), interpolation=cv2.INTER_AREA)
        x = int(round((c["x_mm"] - wmm / 2 - xmin) * esc))
        y = int(round((c["y_mm"] - hmm / 2 - ymin) * esc))
        x1, y1 = min(x + tw, W), min(y + th, H)
        lienzo[y:y1, x:x1] = im[:y1 - y, :x1 - x]
    capa = lienzo.copy()
    fs = max(0.35, tw / 260)
    for c, _ in fotos:
        x = int(round((c["x_mm"] - wmm / 2 - xmin) * esc))
        y = int(round((c["y_mm"] - hmm / 2 - ymin) * esc))
        cv2.rectangle(capa, (x, y), (x + tw - 1, y + th - 1), (255, 255, 255), 1)
        num = c["campo"].replace("campo", "")
        cv2.putText(capa, num, (x + 4, y + int(18 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(capa, num, (x + 4, y + int(18 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    if diametro_mm:
        cx, cy = int(round(-xmin * esc)), int(round(-ymin * esc))
        cv2.circle(capa, (cx, cy), int(round(diametro_mm / 2 * esc)), (0, 200, 255), 2, cv2.LINE_AA)
    lienzo = cv2.addWeighted(capa, 0.75, lienzo, 0.25, 0)
    salida = salida or os.path.join(carpeta, f"cuadricula_{modo}.jpg")
    cv2.imwrite(salida, lienzo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return salida


# ------------------------------------------------------------------ hoja para el modo manual
def hoja_manual(cfg, campos, ruta):
    """Dibujo de la grilla para imprimir: círculo del disco, campos numerados y lecturas del vernier."""
    W, H = 1240, 1754                                # A4 a 150 ppp
    hoja = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(hoja, "MicroAlerta - grilla manual (celular)", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cob = cobertura_manual(cfg, campos)
    sub = (f"Disco {cfg.diametro_mm:g} mm | campo {cfg.campo_celular_mm:g} mm | {len(campos)} campos | "
           f"cubren {cob * 100:.0f} % del disco")
    cv2.putText(hoja, sub, (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
    cx, cy, R = W // 2, 640, 440
    esc = R / (cfg.diametro_mm / 2)
    cv2.circle(hoja, (cx, cy), R, (0, 0, 0), 2, cv2.LINE_AA)
    for c in campos:
        x, y = int(cx + c["x_mm"] * esc), int(cy + c["y_mm"] * esc)
        cv2.circle(hoja, (x, y), int(cfg.campo_celular_mm / 2 * esc), (0, 120, 220), 2, cv2.LINE_AA)
        n = str(int(c["campo"].replace("campo", "")))
        (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(hoja, n, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    y = 1140
    cv2.putText(hoja, "Campo   Vernier X (mm)   Vernier Y (mm)", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    col, fila = 0, 0
    for c in campos:
        txt = f"{int(c['campo'].replace('campo', '')):>3}     {c['vernier_x']:>7.2f}        {c['vernier_y']:>7.2f}"
        cv2.putText(hoja, txt, (60 + col * 600, y + 40 + fila * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        fila += 1
        if fila >= 17:
            fila, col = 0, col + 1
    cv2.putText(hoja, "Siempre llegar a cada lectura girando la perilla en el mismo sentido.", (60, H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(ruta, hoja)
    return ruta


# ------------------------------------------------------------------ fotos del celular
def _fecha_foto(ruta):
    """Fecha de toma (EXIF DateTimeOriginal) si existe; si no, la fecha del archivo."""
    try:
        from PIL import Image
        exif = Image.open(ruta).getexif()
        v = exif.get(36867) or exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if v:
            return dt.datetime.strptime(str(v), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return os.path.getmtime(ruta)


def ordenar_fotos(carpeta_fotos, carpeta_salida, campos, modos=("F",), orden="por-campo"):
    """Copia las fotos del celular, en el orden en que se tomaron, con el nombre de cada campo.
    por-campo: F, W, F, W… · por-pasada: todas las F y después todas las W."""
    import shutil
    ext = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff")
    fotos = [os.path.join(carpeta_fotos, f) for f in os.listdir(carpeta_fotos) if f.lower().endswith(ext)]
    fotos.sort(key=lambda r: (_fecha_foto(r), os.path.basename(r)))
    esperadas = len(campos) * len(modos)
    if len(fotos) != esperadas:
        raise ValueError(f"Hay {len(fotos)} fotos y se esperaban {esperadas} ({len(campos)} campos × "
                         f"{len(modos)} luz/luces). Revise que no falte ni sobre ninguna.")
    os.makedirs(carpeta_salida, exist_ok=True)
    pares = []
    if orden == "por-campo":
        it = iter(fotos)
        for c in campos:
            for m in modos:
                pares.append((next(it), c, m))
    else:
        it = iter(fotos)
        for m in modos:
            for c in campos:
                pares.append((next(it), c, m))
    for ruta, c, m in pares:
        destino = os.path.join(carpeta_salida, f"{c['campo']}_{m}{os.path.splitext(ruta)[1].lower()}")
        shutil.copy2(ruta, destino)
    guardar_plan(campos, os.path.join(carpeta_salida, "campos.csv"))
    return pares


# ------------------------------------------------------------------ calibración de la platina
def desplazamiento_px(a, b):
    """Corrimiento (dx, dy) en píxeles de la imagen b respecto de a (correlación de fase)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ventana = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, ventana)
    return dx, dy, resp


def calibrar_platina(platina, camara, cfg, mover_mm=0.8, avisar=print):
    """Mueve cada eje y mide cuánto se corre la imagen. Entrega pasos por mm, signo, ángulo de la
    cámara respecto de la platina y juego. Necesita textura que no se repita: un disco de papel usado
    con luz blanca sirve; las rayas del micrómetro no (se repiten cada 10 µm y engañan a la medición).
    El recorrido se limita a 30 % del campo para que las dos fotos compartan la mayor parte."""
    res = {}
    platina.cero()
    time.sleep(cfg.espera_s)
    img0 = camara.capturar()
    campo_mm = min(img0.shape[:2]) * cfg.um_por_px / 1000
    mover_mm = min(mover_mm, 0.3 * campo_mm)
    for eje in ("x", "y"):
        pasos_mm = cfg.pasos_mm_x if eje == "x" else cfg.pasos_mm_y
        n = int(round(mover_mm * pasos_mm))
        ida = (n, 0) if eje == "x" else (0, n)
        # llegar avanzando al punto de partida
        platina.ir_a_pasos(-ida[0] // 2, -ida[1] // 2)
        platina.ir_a_pasos(0, 0)
        time.sleep(cfg.espera_s)
        a = camara.capturar()
        platina.ir_a_pasos(*ida)
        time.sleep(cfg.espera_s)
        b = camara.capturar()
        platina.ir_a_pasos(0, 0)                  # vuelve retrocediendo: mide el juego
        time.sleep(cfg.espera_s)
        c = camara.capturar()
        dx, dy, r1 = desplazamiento_px(a, b)
        jx, jy, r2 = desplazamiento_px(a, c)
        mov_mm = math.hypot(dx, dy) * cfg.um_por_px / 1000
        comp = dx if eje == "x" else dy
        # Si el contenido de la imagen se corre hacia +x, el punto que queda al centro es uno que estaba
        # a la izquierda: la posición del centro sobre el disco disminuye. signo = sentido del centro.
        ang = math.degrees(math.atan2(dy, dx) if eje == "x" else math.atan2(-dx, dy))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        res[eje] = {
            "pasos_enviados": n,
            "corrimiento_px": [round(dx, 2), round(dy, 2)],
            "pasos_por_mm": round(n / mov_mm, 2) if mov_mm > 0 else None,
            "signo": -1 if comp > 0 else 1,
            "angulo_grados": round(ang, 2),
            "juego_mm": round(math.hypot(jx, jy) * cfg.um_por_px / 1000, 4),
            "calidad_correlacion": round(min(r1, r2), 3),
        }
        avisar(f"Eje {eje}: {res[eje]}")
    platina.ir_a_pasos(0, 0)
    return res


def guardar_en_config(ruta, cambios, tipo, detalle=None):
    """Escribe valores calibrados en captura.json (lo crea si no existe) y deja registro de la fecha,
    para no tener que copiarlos a mano en cada corrida."""
    datos = {}
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as fh:
            datos = json.load(fh)
    datos.update(cambios)
    registro = datos.setdefault("calibraciones", [])
    registro.append({"fecha": dt.datetime.now().isoformat(timespec="seconds"), "tipo": tipo,
                     "valores": cambios, **({"detalle": detalle} if detalle else {})})
    datos["calibraciones"] = registro[-20:]
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False)
    return datos


def valores_platina(res):
    """De la salida de calibrar_platina a las claves de captura.json. El sobrepaso es el doble
    del juego medido (mínimo 0,1 mm), para llegar siempre desde el mismo lado con margen."""
    juego = max(res["x"]["juego_mm"], res["y"]["juego_mm"])
    cambios = {"signo_x": res["x"]["signo"], "signo_y": res["y"]["signo"],
               "juego_mm": round(max(2 * juego, 0.1), 3)}
    if res["x"]["pasos_por_mm"]:
        cambios["pasos_mm_x"] = res["x"]["pasos_por_mm"]
    if res["y"]["pasos_por_mm"]:
        cambios["pasos_mm_y"] = res["y"]["pasos_por_mm"]
    return cambios
