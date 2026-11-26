"""Detección de partículas teñidas con Rojo de Nilo (foto _F: luz azul + filtro naranjo).

Una partícula es una región claramente más brillante que el papel que la rodea. El fondo se
estima en una imagen reducida con una apertura morfológica (quita todo lo brillante más chico
que el elemento) y un desenfoque, así no lo contaminan las partículas grandes. El umbral es
el fondo + k veces el ruido robusto del fondo, con un piso absoluto (`fluo_umbral_min`) que se
fija con los blancos de procedimiento (puesta a punto C).

Para reutilizar `rasgos.medir`, se entrega una "razón" fondo/señal: 1 en el papel y menor que 1
en las partículas que brillan, igual que la razón de las partículas oscuras en luz blanca.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .vision import _quitar_pequenos, _limite_area_px, _separar_compactas


def senal(img_bgr):
    """Emisión del Rojo de Nilo detrás del filtro naranjo: canales verde y rojo (el azul queda bloqueado)."""
    return img_bgr[:, :, 1:3].max(axis=2).astype(np.float32)


def fondo(s, mask, tam_px):
    """Fondo suave sin partículas: apertura morfológica + desenfoque en una imagen reducida."""
    h, w = s.shape
    esc = min(1.0, 400.0 / max(h, w))
    peq = cv2.resize(s, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(tam_px * esc)) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ab = cv2.morphologyEx(peq, cv2.MORPH_OPEN, el)
    ab = cv2.GaussianBlur(ab, (0, 0), k / 2)
    return np.maximum(cv2.resize(ab, (w, h), interpolation=cv2.INTER_LINEAR), 1.0)


def residuo(img_f, mask, p, um_px=None):
    """Brillo sobre el fondo (d), zona útil, fondo, mediana y ruido robusto (σ) de d."""
    s = senal(img_f)
    s = cv2.GaussianBlur(s, (0, 0), 0.8)
    dentro = mask > 0
    tam_um = p.get("fluo_fondo_um", 600.0)
    tam_px = tam_um / um_px if um_px else 0.25 * max(s.shape)
    bg = fondo(s, mask, tam_px)
    d = s - bg
    dv = d[dentro]
    med = float(np.median(dv)) if dv.size else 0.0
    sigma = float(1.4826 * np.median(np.abs(dv - med))) if dv.size else 1.0
    return s, d, dentro, bg, med, max(sigma, 0.5)


def picos_fluorescencia(img_f, mask, p, um_px, limite_um=20.0):
    """Para fijar el umbral con los blancos: brillo máximo sobre el fondo (d) de cada mancha de al
    menos `limite_um` que el programa detectaría si no hubiera piso (fondo + fluo_k_alto σ). El piso
    fluo_umbral_min decide cuáles de ellas se cuentan."""
    s, d, dentro, bg, med, sigma = residuo(img_f, mask, p, um_px)
    alto = med + p.get("fluo_k_alto", 6.0) * sigma
    bajo = max(med + p.get("fluo_k_bajo", 3.0) * sigma, 0.5 * alto)
    bw = filters.apply_hysteresis_threshold(d, bajo, alto) & dentro
    lab = measure.label(bw, connectivity=2)
    frac = p.get("fluo_fraccion_borde", 0.5)
    picos = []
    for reg in measure.regionprops(lab, intensity_image=d):
        sl, comp = reg.slice, reg.image
        # tamaño medido como en segmentar_fluorescencia: contorno a media altura del brillo
        fino = comp & (d[sl] >= max(bajo, frac * float(np.percentile(d[sl][comp], 95))))
        if not fino.any():
            continue
        largo = measure.regionprops(fino.astype(np.uint8))[0].feret_diameter_max * (um_px or 1.0)
        if largo >= limite_um:
            picos.append(float(reg.intensity_max))
    return picos


def segmentar_fluorescencia(img_f, mask, p, um_px=None):
    """Devuelve (etiquetas int32, mapa_fibra bool, razon float, umbrales dict, mascara_util uint8)."""
    s, d, dentro, bg, med, sigma = residuo(img_f, mask, p, um_px)
    alto = max(med + p.get("fluo_k_alto", 6.0) * sigma, p.get("fluo_umbral_min", 12.0))
    bajo = max(med + p.get("fluo_k_bajo", 3.0) * sigma, 0.5 * alto)
    bw = filters.apply_hysteresis_threshold(d, bajo, alto) & dentro
    bw = ndi.binary_fill_holes(ndi.binary_closing(bw, structure=morphology.disk(1))) & dentro
    # Borde a media altura: el brillo de una partícula se desparrama (desenfoque, halo), así que el
    # contorno se toma donde la señal cae a la mitad de su máximo (como el ancho a media altura).
    lab0 = measure.label(bw, connectivity=2)
    fino = np.zeros_like(bw)
    frac = p.get("fluo_fraccion_borde", 0.5)
    for reg in measure.regionprops(lab0, intensity_image=d):
        sl = reg.slice
        comp = reg.image
        pico = float(np.percentile(d[sl][comp], 95))
        fino[sl] |= comp & (d[sl] >= max(bajo, frac * pico))
    bw = ndi.binary_fill_holes(fino) & dentro
    area_min = _limite_area_px(p, um_px)
    bw = _quitar_pequenos(bw, max(int(area_min), 4))
    etiquetas = measure.label(bw, connectivity=2).astype(np.int32)
    # fibras: lo alargado y fino (se usa solo para el rasgo frac_cresta y para no partir fibras)
    fibra = np.zeros_like(bw)
    for reg in measure.regionprops(etiquetas):
        if reg.axis_major_length > 3 * max(reg.axis_minor_length, 1.0):
            fibra[reg.slice] |= reg.image
    # sin watershed por defecto: los fragmentos irregulares se partirían en varios
    if p.get("fluo_watershed", False):
        etiquetas = _separar_compactas(etiquetas, fibra, area_min).astype(np.int32)
    razon = np.clip(bg / np.maximum(s, 1.0), 0.0, 1.0).astype(np.float32)
    u = {"fondo_mediano": float(np.median(bg[dentro])) if dentro.any() else None,
         "ruido_fondo": sigma, "umbral_alto": alto, "umbral_bajo": bajo,
         "fraccion_campo_util": float(dentro.mean())}
    return etiquetas, fibra.astype(bool), razon, u, mask


def color_fluorescencia(img_f, comp_local, y0, x0, y1, x1):
    """Color de emisión: el Rojo de Nilo brilla más verde-amarillo en plásticos apolares (PE, PP)
    y más naranjo-rojo en los polares (PET, nylon). Se informa como orientación, no como polímero."""
    sub = img_f[y0:y1, x0:x1].astype(np.float32)
    g = float(sub[:, :, 1][comp_local].mean())
    r = float(sub[:, :, 2][comp_local].mean())
    rg = r / max(g, 1.0)
    if rg < 0.9:
        nombre = "verde"
    elif rg < 1.3:
        nombre = "amarillo"
    elif rg < 2.0:
        nombre = "naranjo"
    else:
        nombre = "rojo"
    return rg, nombre
