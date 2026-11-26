"""Parámetros por defecto. Se pueden sobrescribir con un JSON (--parametros)."""
import json
from copy import deepcopy

DEFAULTS = {
    # --- Campo visual (fotos a través del ocular) ---
    "detectar_campo_circular": True,   # False para imágenes del escáner (campo rectangular completo)
    "margen_campo": 0.05,              # fracción del radio que se descarta en el borde del círculo (viñeteo)

    # --- Fondo ---
    "fondo_grado": 4,                  # grado del polinomio de iluminación de fondo
    "iluminacion_min_rel": 0.55,
    "zona_oscura_sigma": 0.012,         # suavizado para detectar zonas oscuras extensas (borde de filtro, sombras)
    "zona_oscura_razon": 0.75,
    "zona_oscura_area_borde": 0.015,   # ... o >1,5 % si entra desde el borde del campo
    "zona_oscura_area_max": 0.08,      # si una zona oscura suave ocupa >8 % del campo se excluye del análisis       # se excluye la zona con iluminación < 55 % de la mediana (viñeteo)

    # --- Segmentación ---
    # Umbrales sobre la razón intensidad/fondo (1 = papel). Se adaptan a la textura del papel:
    # semilla = mediana − k_sigma_semilla·σ, acotada a [semilla_min, umbral_oscuro_blob]
    "umbral_oscuro_blob": 0.60,        # tope de la semilla (núcleo oscuro obligatorio)
    "semilla_min": 0.30,
    "k_sigma_semilla": 4.0,
    "umbral_oscuro_fibra": 0.78,       # tope del crecimiento (bordes de partículas y fibras)
    "k_sigma_crece": 2.0,
    "fibra_sigmas_px": [1.0, 2.0, 3.0],
    "fibra_ridge_alto": 0.12,          # tope del umbral alto de cresta (papel muy texturado)
    "fibra_ridge_alto_min": 0.04,      # piso del umbral alto (papel limpio)
    "k_sigma_cresta": 8.0,
    "umbral_fibra_max": 0.88,          # una fibra debe ser al menos 12 % más oscura que el papel          # histéresis del filtro de crestas (Sato), normalizado 0-1
    "fibra_ridge_bajo": 0.05,
    "watershed": True,                 # separar partículas compactas que se tocan

    # --- Filtros de tamaño (en µm si hay calibración, si no en píxeles) ---
    "area_min_um2": 150.0,             # ~ partícula de 14 µm de diámetro equivalente
    "area_min_px": 25,
    "largo_min_fibra_um": 50.0,
    "largo_min_fibra_px": 20,

    # --- Clasificación morfológica (umbrales estándar) ---
    "razon_fibra": 3.0,                # largo/ancho > 3  -> fibra
    "pelicula_contraste_max": 0.25,    # partícula translúcida (poco más oscura que el fondo) ...
    "pelicula_solidez_max": 0.85,      # ... e irregular -> película (heurístico, requiere validación)

    # --- Conteo ---
    "contar_borde": False,             # partículas que tocan el borde del área analizada: se listan pero no se cuentan

    # --- Clasificador PCM ---
    "umbral_p_pcm": 0.5,

    # --- Fluorescencia (Rojo de Nilo, foto _F con luz azul y filtro naranjo) ---
    "fluo_umbral_min": 12.0,           # brillo mínimo sobre el fondo (0-255); se fija con los blancos
    "fluo_k_alto": 6.0,                # semilla: fondo + 6 σ del ruido del fondo
    "fluo_k_bajo": 3.0,                # crecimiento: fondo + 3 σ
    "fluo_fondo_um": 600.0,            # tamaño del elemento que estima el fondo (más grande que las partículas)
    "fluo_largo_min_fibra_um": 20.0,   # en fluorescencia, las fibras cortas también se cuentan
    "fluo_fraccion_borde": 0.5,        # contorno a media altura del brillo de cada partícula

    # --- Mosaico ---
    "margen_disco_mm": 0.15,           # con posiciones: se analiza hasta (diámetro/2 − margen) del centro del disco

    # --- Informe ---
    "limite_minimo_um": None,          # largo mínimo que se cuenta (20 µm con papel filtro); None = sin límite
}


def cargar(path=None):
    p = deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            p.update(json.load(f))
    return p
