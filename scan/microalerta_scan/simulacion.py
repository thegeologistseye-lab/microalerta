"""Disco teñido con Rojo de Nilo simulado, para probar la captura y el conteo sin el equipo.

La escena tiene partículas conocidas: fragmentos y fibras de plástico que brillan, partículas
que brillan pero miden menos de 20 µm, grumos de tinte, granos de arena y fibras de algodón (que no
brillan). `fotografiar` entrega la foto de un campo con luz azul (F) o blanca (W) en cualquier
posición, con viñeteo, desenfoque y ruido, para que la platina simulada recorra el disco.
"""
import math
import numpy as np
import cv2

# color de emisión aproximado (B, G, R) detrás del filtro naranjo, y color en luz blanca
POLIMEROS = {
    "PE":    {"fluo": (8, 170, 200), "blanca": (215, 215, 212), "peso": 0.30},
    "PP":    {"fluo": (8, 120, 210), "blanca": (190, 150, 230), "peso": 0.25},
    "PS":    {"fluo": (10, 185, 190), "blanca": (225, 225, 225), "peso": 0.15},
    "PET":   {"fluo": (6, 70, 150), "blanca": (200, 200, 195), "peso": 0.15},
    "nylon": {"fluo": (6, 90, 175), "blanca": (90, 170, 80), "peso": 0.15},
}
SUB = 4                                     # bits de subpíxel para cv2 (shift)


def _poligono(rng, r_um, n=None):
    n = n or int(rng.integers(5, 10))
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r_um * rng.uniform(0.55, 1.0, n)
    rad[0] = r_um
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])      # µm, relativos al centro


def _fibra(rng, largo_um):
    n = 12
    ang0 = rng.uniform(0, 2 * np.pi)
    curva = rng.normal(0, 0.12, n).cumsum()
    seg = largo_um / (n - 1)
    pts = [np.zeros(2)]
    for k in range(n - 1):
        a = ang0 + curva[k]
        pts.append(pts[-1] + seg * np.array([math.cos(a), math.sin(a)]))
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def _feret(pts):
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return float(d.max())


class Escena:
    def __init__(self, diametro_mm=21.0, semilla=0, n_fragmentos=45, n_fibras=30, n_pequenas=25,
                 n_grumos=25, n_arena=160, n_algodon=12, grumo_radio_um=(3, 6), grumo_brillo=1.0):
        self.R = diametro_mm / 2
        rng = np.random.default_rng(semilla)
        self.rng_ruido = np.random.default_rng(semilla + 1000)
        self.p = []
        nombres = list(POLIMEROS)
        pesos = np.array([POLIMEROS[k]["peso"] for k in nombres])
        pesos /= pesos.sum()

        def lugar(margen_mm=0.4):
            while True:
                x, y = rng.uniform(-self.R, self.R, 2)
                if math.hypot(x, y) < self.R - margen_mm:
                    return float(x), float(y)

        def separado(x, y, r_mm):
            return all(math.hypot(x - q["x"], y - q["y"]) > r_mm + q["r_mm"] + 0.03 for q in self.p)

        def agregar(tipo, pts_um, ancho_um, fluo, pol=None, brillo=1.0, cerrado=True):
            r_mm = float(np.sqrt((pts_um ** 2).sum(1)).max() / 1000 + ancho_um / 2000)
            for _ in range(200):
                x, y = lugar()
                if separado(x, y, r_mm):
                    break
            largo = _feret(pts_um) + (ancho_um if not cerrado else 0.0)
            self.p.append({"tipo": tipo, "x": x, "y": y, "pts": pts_um, "ancho_um": ancho_um, "cerrado": cerrado,
                           "fluo": fluo, "pol": pol, "brillo": brillo, "r_mm": r_mm, "largo_um": largo})

        for _ in range(n_fragmentos):
            pol = str(rng.choice(nombres, p=pesos))
            r = float(np.exp(rng.uniform(np.log(15), np.log(220))))           # 30–440 µm de largo
            agregar("fragmento", _poligono(rng, r), 0.0, True, pol, float(rng.uniform(0.55, 1.0)))
        for _ in range(n_fibras):
            pol = str(rng.choice(["PET", "PET", "nylon", "PP"]))
            agregar("fibra", _fibra(rng, float(rng.uniform(120, 900))), float(rng.uniform(12, 22)), True, pol,
                    float(rng.uniform(0.6, 1.0)), cerrado=False)
        for _ in range(n_pequenas):                                          # brillan, pero < 20 µm
            pol = str(rng.choice(nombres, p=pesos))
            agregar("pequena", _poligono(rng, float(rng.uniform(4, 7.5))), 0.0, True, pol, float(rng.uniform(0.6, 1.0)))
        for _ in range(n_grumos):                                            # grumos de tinte: puntos chicos
            agregar("grumo", _poligono(rng, float(rng.uniform(*grumo_radio_um)), 8), 0.0, True, "grumo", grumo_brillo)
        for _ in range(n_arena):                                             # no brillan
            agregar("arena", _poligono(rng, float(np.exp(rng.uniform(np.log(10), np.log(120))))), 0.0, False)
        for _ in range(n_algodon):
            agregar("algodon", _fibra(rng, float(rng.uniform(200, 900))), float(rng.uniform(12, 20)), False, cerrado=False)

    # -------------------------------------------------------------- verdad
    def verdad(self, limite_um=20.0):
        """Partículas que el programa debería contar: plásticos que brillan y miden ≥ límite."""
        return [q for q in self.p if q["tipo"] in ("fragmento", "fibra") and q["largo_um"] >= limite_um]

    # -------------------------------------------------------------- fotos
    def _fondo_suave(self, X, Y):
        # variación lenta del papel (igual en campos vecinos, así el mosaico empalma)
        return (1.0 + 0.06 * np.sin(1.3 * X + 0.4) * np.cos(0.9 * Y - 0.2)
                + 0.04 * np.sin(0.5 * X * Y + 1.0))

    def fotografiar(self, x_mm, y_mm, ancho_px, alto_px, um_px, modo="F"):
        w, h = int(ancho_px), int(alto_px)
        mm_px = um_px / 1000
        jj = (np.arange(w) - w / 2) * mm_px + x_mm
        ii = (np.arange(h) - h / 2) * mm_px + y_mm
        X, Y = np.meshgrid(jj, ii)
        suave = self._fondo_suave(X, Y).astype(np.float32)
        rng = self.rng_ruido
        textura = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5)
        textura /= textura.std() + 1e-6
        dentro = (X ** 2 + Y ** 2) <= self.R ** 2
        if modo == "F":
            base = np.array([5, 14, 24], np.float32)                         # papel teñido, débil
            img = base[None, None, :] * (suave * (1 + 0.08 * textura))[:, :, None]
            img[~dentro] *= 0.8                                              # borde del disco (bajo el empaque)
        else:
            base = np.array([196, 203, 208], np.float32)
            img = base[None, None, :] * (suave * (1 + 0.025 * textura))[:, :, None]
            anillo = np.abs(np.sqrt(X ** 2 + Y ** 2) - self.R) < 0.05
            img[anillo] *= 0.9                                               # huella del empaque

        # partículas del campo
        x0, x1 = jj[0] - 0.6, jj[-1] + 0.6
        y0, y1 = ii[0] - 0.6, ii[-1] + 0.6
        for q in self.p:
            if not (x0 < q["x"] < x1 and y0 < q["y"] < y1):
                continue
            if modo == "F" and not q["fluo"]:
                continue
            pts = (q["pts"] / 1000 + np.array([q["x"], q["y"]]))
            px = np.column_stack([(pts[:, 0] - x_mm) / mm_px + w / 2, (pts[:, 1] - y_mm) / mm_px + h / 2])
            pad = int(q["ancho_um"] / um_px) + 4
            bx0, by0 = int(np.floor(px[:, 0].min())) - pad, int(np.floor(px[:, 1].min())) - pad
            bx1, by1 = int(np.ceil(px[:, 0].max())) + pad, int(np.ceil(px[:, 1].max())) + pad
            cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, w), min(by1, h)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            loc = px - np.array([bx0, by0])
            pxi = np.round(loc * (1 << SUB)).astype(np.int32).reshape(-1, 1, 2)
            m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            if q["cerrado"]:
                cv2.fillPoly(m, [pxi], 255, cv2.LINE_AA, SUB)
            else:
                grosor = max(1, int(round(q["ancho_um"] / um_px)))
                cv2.polylines(m, [pxi], False, 255, grosor, cv2.LINE_AA, SUB)
            a = (m[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0].astype(np.float32) / 255.0)[:, :, None]
            if not a.any():
                continue
            roi = img[cy0:cy1, cx0:cx1]
            if modo == "F":
                col = np.array(POLIMEROS[q["pol"]]["fluo"] if q["pol"] in POLIMEROS else (20, 210, 235), np.float32)
                img[cy0:cy1, cx0:cx1] = roi * (1 - a) + a * col * q["brillo"]
            else:
                if q["tipo"] == "arena":
                    col = np.array([60, 85, 110], np.float32) * (0.8 + 0.4 * ((hash(q["x"]) % 100) / 100))
                elif q["tipo"] == "algodon":
                    col = np.array([175, 180, 182], np.float32)
                elif q["tipo"] == "grumo":
                    col = np.array([150, 120, 190], np.float32)
                else:
                    col = np.array(POLIMEROS[q["pol"]]["blanca"], np.float32) * 0.9
                img[cy0:cy1, cx0:cx1] = roi * (1 - 0.8 * a) + 0.8 * a * col

        # viñeteo, desenfoque y ruido de la cámara
        r2 = ((np.arange(w) - w / 2)[None, :] ** 2 + (np.arange(h) - h / 2)[:, None] ** 2) / (0.5 * math.hypot(w, h)) ** 2
        img *= (1 - 0.18 * r2)[:, :, None]
        img = cv2.GaussianBlur(img, (0, 0), 0.9)
        img += rng.normal(0, 1.6 if modo == "F" else 2.5, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)
