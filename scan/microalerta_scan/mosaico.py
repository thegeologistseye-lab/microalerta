"""Campos solapados del mosaico: cada partícula y cada punto del filtro se cuentan una sola vez.

Cuando las fotos se solapan (10–20 % recomendado para absorber el juego de la platina), una
partícula en la franja común aparece en dos campos.
- Partículas enteras: las detecciones de campos vecinos con casi la misma posición sobre el disco y
  tamaño parecido son la misma partícula ("gemelas"). Se cuenta una vez, en el campo cuyo centro
  está más cerca de su posición promedio. Si el vecino no la detectó, se cuenta igual.
- Trozos: una partícula cortada por el borde de la foto no se cuenta en ese campo. Si ningún campo
  la ve entera (fibra más larga que el solape), sus trozos se agrupan y se cuenta una vez.
- Área: cada punto del filtro "pertenece" al campo cuyo centro está más cerca, entre los campos que
  lo ven. El área analizada es la unión de los campos, sin contar dos veces la franja común.

Supone que los ejes de la imagen coinciden con los de la platina (x a la derecha, y hacia abajo).
"""
import numpy as np
import cv2

ESCALA = 8  # submuestreo de las máscaras para calcular áreas


class Campo:
    def __init__(self, nombre, x_mm, y_mm, um_px, mask):
        self.nombre = nombre
        self.x, self.y = x_mm, y_mm
        self.um_px = um_px
        self.h, self.w = mask.shape[:2]
        self.mask_s = cv2.resize((mask > 0).astype(np.uint8), (max(1, self.w // ESCALA), max(1, self.h // ESCALA)),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        self.mask_e = cv2.erode(self.mask_s.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    def a_mm(self, cx_px, cy_px):
        return (self.x + (cx_px - self.w / 2) * self.um_px / 1000,
                self.y + (cy_px - self.h / 2) * self.um_px / 1000)

    def ve(self, xs, ys, holgura=False):
        """¿Qué puntos del filtro (mm) caen dentro de la zona útil de este campo?
        Con holgura=True exige además un margen respecto del borde (para "ver entera" una partícula)."""
        px = (np.asarray(xs) - self.x) * 1000 / self.um_px + self.w / 2
        py = (np.asarray(ys) - self.y) * 1000 / self.um_px + self.h / 2
        ix = np.floor(px / ESCALA).astype(int)
        iy = np.floor(py / ESCALA).astype(int)
        hs, ws = self.mask_s.shape
        ok = (ix >= 0) & (iy >= 0) & (ix < ws) & (iy < hs)
        res = np.zeros(ok.shape, bool)
        res[ok] = (self.mask_e if holgura else self.mask_s)[iy[ok], ix[ok]]
        return res


def _dueno(campos, xs, ys, cercanos):
    """Índice del campo dueño de cada punto (−1 si ningún campo lo ve)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mejor = np.full(xs.shape, -1)
    dmin = np.full(xs.shape, np.inf)
    for j in cercanos:
        c = campos[j]
        d = (xs - c.x) ** 2 + (ys - c.y) ** 2
        v = c.ve(xs, ys) & (d < dmin)
        mejor[v] = j
        dmin[v] = d[v]
    return mejor


def resolver_solapes(campos, particulas):
    """Marca en cada partícula `fuera_de_su_zona` y devuelve el área propia (mm²) de cada campo.

    campos: lista de Campo. particulas: dicts con 'campo', 'cx_px', 'cy_px'.
    """
    idx = {c.nombre: i for i, c in enumerate(campos)}
    radio = [np.hypot(c.w, c.h) / 2 * c.um_px / 1000 for c in campos]

    def vecinos(i):
        c = campos[i]
        return [j for j, o in enumerate(campos)
                if np.hypot(o.x - c.x, o.y - c.y) <= radio[i] + radio[j]]

    vec = [vecinos(i) for i in range(len(campos))]
    n_solapes = sum(1 for i in range(len(campos)) for j in vec[i] if j > i)

    # 1) posición de cada partícula sobre el disco, según el campo que la vio
    for f in particulas:
        c = campos[idx[f["campo"]]]
        f["_xy"] = c.a_mm(f["cx_px"], f["cy_px"])
        f["fuera_de_su_zona"] = False
    # 2) gemelas: la misma partícula vista entera en dos campos vecinos (posiciones casi iguales y
    #    tamaños parecidos). Se decide el dueño una sola vez por grupo, con la posición promedio,
    #    para que dos campos no se "regalen" mutuamente la partícula por un pequeño error de posición.
    enteras = [f for f in particulas if not f.get("toca_borde_campo")]
    por_campo = {}
    for k, f in enumerate(enteras):
        por_campo.setdefault(idx[f["campo"]], []).append(k)
    padre = list(range(len(enteras)))

    def raiz(k):
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k
    for i in range(len(campos)):
        for j in vec[i]:
            if j <= i:
                continue
            for a in por_campo.get(i, []):
                fa = enteras[a]
                la = fa.get("largo_um") or 0.0
                tol = min(max(0.04, 0.3 * la / 1000), 0.3)
                for b in por_campo.get(j, []):
                    fb = enteras[b]
                    lb = fb.get("largo_um") or 0.0
                    if la and lb and not (0.5 <= la / lb <= 2.0):
                        continue
                    if np.hypot(fa["_xy"][0] - fb["_xy"][0], fa["_xy"][1] - fb["_xy"][1]) <= tol:
                        padre[raiz(a)] = raiz(b)
    grupos = {}
    for k in range(len(enteras)):
        grupos.setdefault(raiz(k), []).append(k)
    for miembros in grupos.values():
        if len(miembros) < 2:
            continue
        x = float(np.mean([enteras[k]["_xy"][0] for k in miembros]))
        y = float(np.mean([enteras[k]["_xy"][1] for k in miembros]))
        def dist(k):
            c = campos[idx[enteras[k]["campo"]]]
            return ((x - c.x) ** 2 + (y - c.y) ** 2, idx[enteras[k]["campo"]])
        dueno = min(miembros, key=dist)
        for k in miembros:
            enteras[k]["fuera_de_su_zona"] = k != dueno
    for f in particulas:
        f.pop("_xy", None)

    _agrupar_piezas(campos, idx, particulas)

    areas = {}
    for i, c in enumerate(campos):
        hs, ws = c.mask_s.shape
        gy, gx = np.mgrid[0:hs, 0:ws]
        sel = c.mask_s
        px = (gx[sel] + 0.5) * ESCALA
        py = (gy[sel] + 0.5) * ESCALA
        xs, ys = c.a_mm(px, py)
        propio = _dueno(campos, xs, ys, vec[i]) == i
        areas[c.nombre] = float(propio.sum()) * (ESCALA * c.um_px / 1000) ** 2
    return areas, n_solapes


def _caja_mm(c, f, holgura_mm=0.01):
    x0, y0, x1, y1 = f["bbox_px"]
    ax, ay = c.a_mm(x0, y0)
    bx, by = c.a_mm(x1, y1)
    return (min(ax, bx) - holgura_mm, min(ay, by) - holgura_mm, max(ax, bx) + holgura_mm, max(ay, by) + holgura_mm)


def _se_tocan(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _agrupar_piezas(campos, idx, particulas):
    """Objetos más largos que el solape (fibras largas): ningún campo los ve enteros, así que en
    cada campo aparecen cortados por el borde de la foto. Los trozos de campos distintos cuyas cajas
    se tocan sobre el disco son el mismo objeto: se cuenta una sola vez (el trozo más grande), salvo
    que algún campo lo vea entero, en cuyo caso ya se contó ahí."""
    enteras = [_caja_mm(campos[idx[f["campo"]]], f) for f in particulas
               if not f.get("toca_borde_campo") and not f.get("fuera_de_su_zona")]
    trozos = [f for f in particulas if f.get("toca_borde_campo")]
    cajas = [_caja_mm(campos[idx[f["campo"]]], f) for f in trozos]
    padre = list(range(len(trozos)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for i in range(len(trozos)):
        for j in range(i + 1, len(trozos)):
            if trozos[i]["campo"] != trozos[j]["campo"] and _se_tocan(cajas[i], cajas[j]):
                padre[raiz(i)] = raiz(j)
    grupos = {}
    for i in range(len(trozos)):
        grupos.setdefault(raiz(i), []).append(i)
    for f in particulas:
        f["pieza_representante"] = False
    for miembros in grupos.values():
        if len(miembros) < 2:
            continue                           # un trozo solo: está en el borde exterior del mosaico
        if any(_se_tocan(cajas[i], e) for i in miembros for e in enteras):
            continue
        mejor = max(miembros, key=lambda i: trozos[i].get("area_um2") or 0)
        trozos[mejor]["pieza_representante"] = True
