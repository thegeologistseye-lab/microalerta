"""Calibración espacial (µm/píxel) y corrección de iluminación plana (flat-field)."""
import numpy as np
import cv2


# ---------------------------------------------------------------- escala espacial
def um_por_px_desde_campo(diametro_campo_um, diametro_campo_px):
    """Para fotos por el ocular: el diámetro del campo visual en µm = número de campo (FN, mm,
    grabado en el ocular) / aumento del objetivo × 1000. Ej.: FN 18, objetivo 4× -> 4500 µm."""
    return float(diametro_campo_um) / float(diametro_campo_px)


def um_por_px_desde_dos_puntos(p1, p2, distancia_um):
    """Calibración manual: dos puntos marcados sobre el portaobjetos micrométrico."""
    d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return distancia_um / d


def periodo_marcas_fft(img_gray, recorte=1024):
    """Estima el periodo (px) de las marcas de un portaobjetos micrométrico con la FFT 2D.
    Devuelve (periodo_px, angulo_grados, calidad). calidad = pico/mediana del espectro."""
    h, w = img_gray.shape
    n = min(recorte, h, w)
    y0, x0 = (h - n) // 2, (w - n) // 2
    c = img_gray[y0:y0 + n, x0:x0 + n].astype(np.float32)
    c = c - c.mean()
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(c * win)))
    cy, cx = n // 2, n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    F[r < 4] = 0                      # quitar componente continua y fondo lento
    F[r > n // 2 - 2] = 0
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    k = np.hypot(iy - cy, ix - cx)
    calidad = float(F[iy, ix] / (np.median(F[F > 0]) + 1e-9))
    # Si existe un pico fuerte a la mitad de frecuencia en la misma dirección, era un armónico
    hy, hx = cy + (iy - cy) / 2, cx + (ix - cx) / 2
    hy, hx = int(round(hy)), int(round(hx))
    if 0 <= hy < n and 0 <= hx < n:
        vecino = F[max(hy - 1, 0):hy + 2, max(hx - 1, 0):hx + 2].max()
        if vecino > 0.5 * F[iy, ix]:
            k = k / 2
    periodo = n / k
    angulo = float(np.degrees(np.arctan2(iy - cy, ix - cx)))
    return float(periodo), angulo, calidad


def um_por_px_desde_micrometro(img_bgr, um_entre_marcas):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    periodo, ang, calidad = periodo_marcas_fft(gray)
    return um_entre_marcas / periodo, {"periodo_px": periodo, "angulo": ang, "calidad_pico": calidad}


# ---------------------------------------------------------------- flat-field
def corregir_flat(img, flat, dark=None):
    """Corrección estándar: (I - D) / (F - D) * media(F - D).
    flat = foto de un filtro limpio (blanco) con la misma iluminación y exposición.
    dark = foto con la luz apagada (opcional)."""
    I = img.astype(np.float32)
    F = flat.astype(np.float32)
    D = dark.astype(np.float32) if dark is not None else np.zeros_like(F)
    FD = np.maximum(F - D, 1.0)
    out = (I - D) / FD * FD.mean(axis=(0, 1), keepdims=True)
    return np.clip(out, 0, 255).astype(np.uint8)
