"""Medición por partícula: dimensiones en µm, forma y descriptores ópticos."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import measure, morphology
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SQRT2 = np.sqrt(2.0)

# Rasgos que usa el clasificador (independientes del color de la tinción)
RASGOS_MODELO = [
    "area_um2", "largo_um", "ancho_um", "razon_aspecto", "circularidad", "solidez",
    "excentricidad", "rugosidad_borde", "curvatura", "contraste_medio", "contraste_max",
    "dL", "da", "db", "uniformidad_L", "uniformidad_color", "frac_cresta",
    "indice_birrefringencia", "birrefringencia_p95", "frac_brillante_pol", "indice_fluorescencia",
]


def largo_ancho_esqueleto(comp):
    """Largo = camino más largo dentro del esqueleto (px, diagonales = √2), así una fibra curva mide
    su largo real y las ramitas de una mancha irregular no cuentan. Ancho = área / largo, acotado
    por abajo con 2 × distancia al borde sobre el esqueleto."""
    sk = morphology.skeletonize(comp)
    n = int(sk.sum())
    if n == 0:
        return 1.0, 1.0
    ys, xs = np.nonzero(sk)
    idx = -np.ones(sk.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    filas, cols, pesos = [], [], []
    H, W = sk.shape
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < H) & (x2 >= 0) & (x2 < W)
        j = idx[y2[ok], x2[ok]]
        m = j >= 0
        filas.append(np.arange(n)[ok][m]); cols.append(j[m]); pesos.append(np.full(m.sum(), w))
    G = csr_matrix((np.concatenate(pesos), (np.concatenate(filas), np.concatenate(cols))), shape=(n, n))
    d0 = dijkstra(G, directed=False, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    d1 = dijkstra(G, directed=False, indices=a)
    d1[~np.isfinite(d1)] = -1
    dist = ndi.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
    ancho_dt = max(2.0 * float(np.median(dist[sk])) - 1.0, 1.0)
    camino = float(d1.max())
    largo = camino + ancho_dt                        # el esqueleto no llega a los extremos
    ancho = max(float(comp.sum()) / max(largo, 1.0), ancho_dt * 0.5, 1.0)
    return max(largo, 1.0), ancho, camino


def feret(comp):
    """Feret máximo (máxima distancia entre puntos del contorno) y mínimo (caja rotada mínima)."""
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) >= 2:
        d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1))
        fmax = float(d.max()) + 0.5
    else:
        fmax = 1.0
    (_, _), (a, b), _ = cv2.minAreaRect(pts)
    fmin = float(min(a, b)) + 0.5
    per_convexo = float(cv2.arcLength(cv2.convexHull(pts), True)) if len(pts) > 2 else 1.0
    return fmax, fmin, max(per_convexo, 1.0)


def clasificar_forma(r, p):
    """Fibra / fragmento / película según umbrales geométricos estándar.
    'película' es heurística (2D no mide espesor): translúcida + irregular. Validar con Raman."""
    if r["razon_aspecto"] > p["razon_fibra"]:
        return "fibra"
    if r["contraste_medio"] < p["pelicula_contraste_max"] and r["solidez"] < p["pelicula_solidez_max"]:
        return "pelicula"
    return "fragmento"


def medir(etiquetas, img_bgr, razon, fibra_map, mask_campo, p, um_px=None, pol=None, fluo=None):
    escala = um_px if um_px else 1.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1:] -= 128.0
    fondo_px = (mask_campo > 0) & (etiquetas == 0)
    # fondo de referencia local: anillo alrededor de cada partícula
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    pol_g = cv2.cvtColor(pol, cv2.COLOR_BGR2GRAY).astype(np.float32) if pol is not None else None
    fluo_r = fluo[:, :, 2].astype(np.float32) if fluo is not None else None   # emisión naranja/roja
    pol_bg = float(np.median(pol_g[fondo_px])) if pol_g is not None and fondo_px.any() else None
    pol_ruido = None
    if pol_bg is not None:
        d = pol_g[fondo_px] - pol_bg
        pol_ruido = max(6.0, 4 * 1.4826 * float(np.median(np.abs(d))))   # 4σ robusto del fondo
    fluo_bg = float(np.median(fluo_r[fondo_px])) if fluo_r is not None and fondo_px.any() else None

    h, w = etiquetas.shape
    filas = []
    for reg in measure.regionprops(etiquetas):
        sl = reg.slice
        comp = reg.image
        if comp.sum() < 3:
            continue
        # anillo de fondo (recorte ampliado)
        y0, x0 = max(sl[0].start - 10, 0), max(sl[1].start - 10, 0)
        y1, x1 = min(sl[0].stop + 10, h), min(sl[1].stop + 10, w)
        local = etiquetas[y0:y1, x0:x1] == reg.label
        anillo = cv2.dilate(local.astype(np.uint8), kernel).astype(bool) & ~local & fondo_px[y0:y1, x0:x1]

        lab_in = lab[y0:y1, x0:x1][local]
        lab_bg = lab[y0:y1, x0:x1][anillo] if anillo.any() else lab_in
        raz_in = razon[y0:y1, x0:x1][local]

        area_px = float(reg.area)
        per_px = max(float(reg.perimeter), 1.0)
        fmax, fmin, per_conv = feret(comp)
        largo_sk, ancho_sk, camino_sk = largo_ancho_esqueleto(comp)
        # largo/ancho: si el objeto es curvo (su línea media es bastante más larga que el Feret
        # máximo) se mide a lo largo del esqueleto; si es recto o compacto, con los calibres de Feret.
        curvo = camino_sk > 1.15 * fmax and ancho_sk < 0.25 * fmax
        if curvo:
            largo, ancho = largo_sk, ancho_sk
        else:
            largo, ancho = fmax, fmin
        razon_sk = razon_fe = largo / max(ancho, 1.0)

        r = {
            "id": int(reg.label),
            "cx_px": float(reg.centroid[1]), "cy_px": float(reg.centroid[0]),
            "bbox_px": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
            "area_um2": area_px * escala ** 2,
            "perimetro_um": per_px * escala,
            "diam_equiv_um": float(np.sqrt(4 * area_px / np.pi)) * escala,
            "largo_um": largo * escala,
            "ancho_um": ancho * escala,
            "feret_max_um": fmax * escala,
            "razon_aspecto": float(razon_fe),
            "curvatura": float(camino_sk / fmax),
            "circularidad": float(min(4 * np.pi * area_px / per_px ** 2, 1.0)),
            "solidez": float(reg.solidity),
            "excentricidad": float(reg.eccentricity),
            "rugosidad_borde": float(per_px / per_conv),
            "contraste_medio": float(1 - raz_in.mean()),
            "contraste_max": float(1 - np.percentile(raz_in, 5)),
            "dL": float(lab_in[:, 0].mean() - lab_bg[:, 0].mean()),
            "da": float(lab_in[:, 1].mean() - lab_bg[:, 1].mean()),
            "db": float(lab_in[:, 2].mean() - lab_bg[:, 2].mean()),
            "uniformidad_L": float(lab_in[:, 0].std()),
            "uniformidad_color": float(np.hypot(lab_in[:, 1].std(), lab_in[:, 2].std())),
            "frac_cresta": float((fibra_map[sl] & comp).sum() / area_px),
            "toca_borde_campo": bool(_toca_borde(local, mask_campo[y0:y1, x0:x1])),
            "indice_birrefringencia": None, "birrefringencia_p95": None, "frac_brillante_pol": None,
            "indice_fluorescencia": None,
        }
        if pol_g is not None:
            vp = pol_g[y0:y1, x0:x1][local] - pol_bg
            r["indice_birrefringencia"] = float(vp.mean() / 255.0)
            # cristales (cuarzo, sal) suelen dar puntos muy brillantes; fibras de PET/nylon, brillo a lo largo
            r["birrefringencia_p95"] = float(np.percentile(vp, 95) / 255.0)
            r["frac_brillante_pol"] = float((vp > pol_ruido).mean())
        if fluo_r is not None:
            r["indice_fluorescencia"] = float((fluo_r[y0:y1, x0:x1][local].mean() - fluo_bg) / 255.0)

        # filtros de tamaño mínimo
        if r["razon_aspecto"] > p["razon_fibra"]:
            lim = p["largo_min_fibra_um"] if um_px else p["largo_min_fibra_px"]
            if r["largo_um"] < lim:
                continue
        r["forma"] = clasificar_forma(r, p)
        if r["forma"] == "fibra":
            # Ancho "óptico": oscuridad integrada / (largo × contraste del núcleo). Corrige el
            # ensanchamiento que produce el desenfoque en fibras de pocos píxeles de ancho.
            dil = cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            osc = np.clip(1.0 - razon[y0:y1, x0:x1][dil], 0, 1).sum()
            nucleo = max(r["contraste_max"], 0.05)
            ancho_opt = osc / (largo * nucleo)
            r["ancho_um"] = float(min(max(ancho_opt, 1.0), ancho) * escala)
        r["unidad"] = "um" if um_px else "px"
        filas.append(r)
    return filas


def _toca_borde(local, campo_local):
    borde = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return bool((borde & (campo_local == 0)).any())
