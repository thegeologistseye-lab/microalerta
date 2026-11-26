"""Preprocesamiento y segmentación.

Idea central: las tinciones (Lugol, azul de metileno) colorean TODO el papel, así que el color
absoluto no sirve. Se trabaja con la razón intensidad / fondo local: una partícula es una
región claramente más oscura que el papel que la rodea, sin importar el color de la tinción.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature


# ---------------------------------------------------------------- campo visual
def _ajustar_circulo(pts, forma):
    """Ajuste algebraico (Kåsa) solo con puntos del contorno que NO están en el borde de la foto:
    si el círculo del ocular sale de la imagen, igual se recupera su radio real."""
    h, w = forma
    libre = (pts[:, 0] > 2) & (pts[:, 0] < w - 3) & (pts[:, 1] > 2) & (pts[:, 1] < h - 3)
    q = pts[libre] if libre.sum() >= 20 else pts
    x, y = q[:, 0], q[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a1 / 2, a2 / 2
    r = np.sqrt(max(a3 + cx ** 2 + cy ** 2, 1.0))
    if not np.isfinite(r) or r > 3 * max(h, w):
        (cx, cy), r = cv2.minEnclosingCircle(cv2.convexHull(pts.astype(np.float32)))
    return (float(cx), float(cy)), float(r)


def detectar_campo(img_bgr, margen=0.04):
    """Encuentra el círculo iluminado del ocular. Devuelve (mascara uint8, (cx, cy, r)).
    Soporta círculos cortados por el borde de la foto."""
    h, w = img_bgr.shape[:2]
    esc = 600.0 / max(h, w)
    peq = cv2.resize(img_bgr, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
    v = cv2.GaussianBlur(peq.max(axis=2), (0, 0), 3)
    _, bw = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    mejor, mejor_score, mejor_circ = None, -1, None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.02 * bw.size:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        (cx, cy), r = _ajustar_circulo(cnt, bw.shape)
        circ = np.zeros_like(comp)
        cv2.circle(circ, (int(cx), int(cy)), int(r), 1, -1)
        relleno = area / max(circ.sum(), 1)       # cuán "lleno" está el círculo (área recortada por la foto)
        score = area * relleno ** 3
        if score > mejor_score:
            mejor, mejor_score, mejor_circ = i, score, (cx, cy, r)
    if mejor is None:                              # sin círculo claro: usar toda la imagen
        return np.full((h, w), 255, np.uint8), None
    cx, cy, r = (c / esc for c in mejor_circ)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * (1 - margen)), 255, -1)
    # el círculo puede extenderse fuera de la foto: intersectar con la región realmente iluminada
    ilum = cv2.resize((lab == mejor).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    ilum = cv2.erode(ilum, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r * margen * 2) | 1,) * 2))
    mask = cv2.bitwise_and(mask, ilum)
    return mask, (float(cx), float(cy), float(r))


# ---------------------------------------------------------------- fondo
def luminancia(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _terminos(xn, yn, grado):
    return [xn ** i * yn ** j for i in range(grado + 1) for j in range(grado + 1 - i)]


def fondo_polinomial(canal, mask, grado=4, iters=6):
    """Iluminación de fondo como superficie polinomial suave, ajustada de forma robusta:
    en cada iteración se excluyen los píxeles claramente más oscuros (partículas), así que
    incluso fragmentos grandes no 'contaminan' el fondo. Captura viñeteo y gradientes."""
    h, w = canal.shape
    esc = 160.0 / max(h, w)
    sh, sw = max(int(h * esc), 8), max(int(w * esc), 8)
    cs = cv2.resize(canal, (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    if ms.sum() < 50:
        return np.full_like(canal, float(np.median(canal)))
    yy, xx = np.mgrid[0:sh, 0:sw]
    T = _terminos(xx / sw * 2 - 1, yy / sh * 2 - 1, grado)
    A = np.stack([t[ms] for t in T], 1)
    b = cs[ms]
    keep = np.ones(len(b), bool)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        pred = A @ c
        res = (b - pred) / np.maximum(pred, 1.0)
        rk = res[keep]
        s = 1.4826 * np.median(np.abs(rk - np.median(rk))) + 1e-6
        keep = res > -2.5 * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = sum(ci * t for ci, t in zip(c, _terminos(xx / w * 2 - 1, yy / h * 2 - 1, grado)))
    return np.maximum(F.astype(np.float32), 1.0)


def canal_trabajo(img_bgr, mask):
    """Las tinciones dejan un canal saturado (rojo con Lugol, azul con azul de metileno) y otro casi
    negro y ruidoso. Se usa el canal no saturado más brillante: mejor relación señal/ruido."""
    dentro = mask > 0
    mejor, mejor_med = None, -1
    for k in range(3):
        v = img_bgr[:, :, k][dentro]
        if v.size == 0:
            continue
        sat, med = (v >= 250).mean(), float(np.median(v))
        if sat < 0.15 and med > mejor_med:
            mejor, mejor_med = k, med
    if mejor is None:
        mejor = int(np.argmin([(img_bgr[:, :, k][dentro] >= 250).mean() for k in range(3)]))
    return mejor


def excluir_zonas_oscuras(c, util, p):
    """Zonas oscuras EXTENSAS (borde del filtro, pliegues, sombras del ocular) no son partículas.
    Se buscan en la imagen muy suavizada, relativas a la mediana del campo, y se excluyen si ocupan
    más del 8 % del campo (ninguna partícula individual es tan grande en un campo)."""
    h, w = c.shape
    sig = max(h, w) * p.get("zona_oscura_sigma", 0.03)
    med = float(np.median(c[util]))
    suave = cv2.GaussianBlur(np.where(util, c, med).astype(np.float32), (0, 0), sig)
    oscuro = ((suave < p.get("zona_oscura_razon", 0.75) * med) & util).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(oscuro, connectivity=8)
    area_util = max(int(util.sum()), 1)
    excl = np.zeros_like(util)
    borde = util & ~cv2.erode(util.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    toca = set(np.unique(lab[borde]).tolist())
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        # una zona que entra desde el borde del campo es continuación de algo externo (borde del filtro)
        lim = p.get("zona_oscura_area_borde", 0.015) if i in toca else p.get("zona_oscura_area_max", 0.08)
        if a > lim * area_util:
            excl |= lab == i
    if excl.any():
        excl = cv2.dilate(excl.astype(np.uint8), np.ones((int(sig) | 1,) * 2, np.uint8)).astype(bool)
        util = util & ~excl
    return util


def razon_fondo(img_bgr, mask, p):
    """Razón intensidad/fondo en el canal de trabajo. 1 = papel, <1 = más oscuro.
    Además reduce la máscara a la zona útil: sin viñeteo fuerte ni zonas oscuras extensas."""
    k = canal_trabajo(img_bgr, mask)
    c = img_bgr[:, :, k].astype(np.float32)
    util = excluir_zonas_oscuras(c, mask > 0, p)
    m8 = util.astype(np.uint8) * 255
    F = fondo_polinomial(c, m8, p.get("fondo_grado", 4))
    razon = np.clip(cv2.GaussianBlur(c, (0, 0), 1.0) / F, 0, 2)
    util &= F >= p.get("iluminacion_min_rel", 0.55) * float(np.median(F[util])) if util.any() else util
    return k, util.astype(np.uint8) * 255, razon


def umbrales_adaptativos(razon, mask, p):
    """El papel tiene textura propia: el umbral se ajusta a su variabilidad (σ robusta), con topes."""
    v = razon[mask > 0]
    med = float(np.median(v))
    sigma = float(1.4826 * np.median(np.abs(v - med)))
    semilla = float(np.clip(med - p["k_sigma_semilla"] * sigma, p["semilla_min"], p["umbral_oscuro_blob"]))
    crece = float(np.clip(med - p["k_sigma_crece"] * sigma, semilla + 0.08, p["umbral_oscuro_fibra"]))
    return {"mediana": med, "sigma_textura": sigma, "semilla": semilla, "crecimiento": crece}


# ---------------------------------------------------------------- segmentación
def _quitar_pequenos(bw, area_min):
    """Elimina componentes (8-conectividad) con menos de area_min píxeles."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min
    return keep[lab]


def _limite_area_px(p, um_px):
    return p["area_min_um2"] / um_px ** 2 if um_px else p["area_min_px"]


def segmentar(img_bgr, mask, p, um_px=None):
    """Devuelve (etiquetas int32, mapa_fibra bool, razon float, umbrales dict, mascara_util uint8)."""
    mask_in = mask
    canal, mask, razon = razon_fondo(img_bgr, mask, p)
    dentro = mask > 0
    u = umbrales_adaptativos(razon, mask, p)
    u["canal_bgr"] = int(canal)
    u["fraccion_campo_util"] = float(dentro.sum() / max((mask_in > 0).sum(), 1))

    # 1) partículas oscuras: semillas muy oscuras que crecen hasta el umbral de crecimiento
    semillas = (razon < u["semilla"]) & dentro
    semillas = ndi.binary_opening(semillas, structure=morphology.disk(1))
    blob = filters.apply_hysteresis_threshold(-razon, -u["crecimiento"], -u["semilla"]) & dentro
    lab_b, _ = ndi.label(blob, structure=np.ones((3, 3)))
    con_semilla = np.unique(lab_b[semillas])
    blob = np.isin(lab_b, con_semilla[con_semilla > 0])

    # 2) fibras: crestas oscuras finas (filtro de Sato sobre la razón) + más oscuras que el papel
    ridge = filters.sato(razon, sigmas=p["fibra_sigmas_px"], black_ridges=True, mode="reflect")
    ridge[~dentro] = 0
    # umbral de cresta adaptado a la textura: en papel limpio se detectan fibras tenues,
    # en papel muy texturado se exige una cresta marcada
    rv = ridge[dentro]
    rmed = float(np.median(rv)) if rv.size else 0.0
    rsig = float(1.4826 * np.median(np.abs(rv - rmed))) if rv.size else 0.0
    alto = float(np.clip(rmed + p["k_sigma_cresta"] * rsig, p["fibra_ridge_alto_min"], p["fibra_ridge_alto"]))
    bajo = alto * p["fibra_ridge_bajo"] / p["fibra_ridge_alto"]
    lim_fibra = float(np.clip(u["mediana"] - 3 * u["sigma_textura"], u["crecimiento"], p["umbral_fibra_max"]))
    u.update({"cresta_alto": alto, "cresta_bajo": bajo, "razon_max_fibra": lim_fibra})
    fibra = filters.apply_hysteresis_threshold(ridge, bajo, alto)
    fibra &= (razon < lim_fibra) & dentro
    fibra = _quitar_pequenos(fibra, max(int(p["largo_min_fibra_px"] * 1.5), 10))

    union = ndi.binary_closing(blob | fibra, structure=morphology.disk(1))
    union = ndi.binary_fill_holes(union) & dentro
    area_min = _limite_area_px(p, um_px)
    union = _quitar_pequenos(union, max(int(area_min), 4))

    etiquetas = measure.label(union, connectivity=2)

    # 3) watershed solo en componentes compactas (sin fibra) para separar partículas pegadas
    if p.get("watershed", True):
        etiquetas = _separar_compactas(etiquetas, fibra, area_min)

    return etiquetas.astype(np.int32), fibra, razon, u, mask


def _separar_compactas(etiquetas, fibra, area_min):
    out = etiquetas.copy()
    siguiente = etiquetas.max() + 1
    for reg in measure.regionprops(etiquetas):
        if reg.area < 4 * area_min:
            continue
        sl = reg.slice
        comp = reg.image
        if (fibra[sl] & comp).sum() > 0.2 * reg.area:
            continue
        dist = ndi.distance_transform_edt(comp)
        h = max(2.0, 0.35 * dist.max())                  # conservador: evita sobresegmentar
        picos = morphology.h_maxima(dist, h)
        marcadores = measure.label(picos)
        if marcadores.max() < 2:
            continue
        ws = segmentation.watershed(-dist, marcadores, mask=comp)
        sub = out[sl]
        for k in range(1, ws.max() + 1):
            parte = ws == k
            if parte.sum() < area_min:
                continue
            if k == 1:
                continue                                 # conserva la etiqueta original
            sub[parte] = siguiente
            siguiente += 1
    return out


# ---------------------------------------------------------------- multimodal
def alinear(ref_bgr, mov_bgr):
    """Alinea una imagen de otra modalidad (polarizada / fluorescencia) con la de luz blanca.
    Usa magnitud de gradiente para que el cambio de contraste entre modalidades no confunda el ECC.
    Devuelve (imagen_alineada, matriz 2x3, correlación)."""
    def grad(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 2)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        m = cv2.magnitude(gx, gy)
        return m / (m.max() + 1e-6)
    h, w = ref_bgr.shape[:2]
    esc = 800.0 / max(h, w)
    r = cv2.resize(grad(ref_bgr), None, fx=esc, fy=esc)
    m = cv2.resize(grad(cv2.resize(mov_bgr, (w, h))), None, fx=esc, fy=esc)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(r, m, W, cv2.MOTION_EUCLIDEAN,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        cc = float("nan")
    W[:, 2] /= esc
    alineada = cv2.warpAffine(cv2.resize(mov_bgr, (w, h)), W, (w, h),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return alineada, W, float(cc)
