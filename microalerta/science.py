"""Cálculos científicos de MicroAlerta: concentración, TDC, semáforo y control de calidad."""
import math

from .config import UMBRALES, NIVELES_MP

# Igual que microalerta_scan.muestra.evaluar_campana (programa MicroAlerta Scan) y que la sección 5
# del protocolo. Si cambia en un lado, cambiarlo en los tres.
FORMULA_CRITERIO = ("confianza reforzada si réplicas >= 3, controles >= 3 y media(N_m) > media(N_c) + 3*DE; "
                    "N = n_contado / fraccion_analizada; DE = max(s_c con n-1, sqrt(max(media n_contado controles, 1)) "
                    "/ fraccion media de los controles); si no, exploratorio")


def concentracion(particulas: float, volumen_ml: float) -> float:
    """Partículas por mL de agua de la placa."""
    if volumen_ml <= 0:
        raise ValueError("El volumen debe ser mayor que cero")
    return particulas / volumen_ml


def tdc(muestra: float, control: float, area_cm2: float, horas: float) -> float:
    """Tasa de deposición corregida en PCM/(m²·día): máx(0, Nm − Nc) / (A × t)."""
    if area_cm2 <= 0 or horas <= 0:
        raise ValueError("El área y el tiempo deben ser mayores que cero")
    return max(0.0, muestra - control) / ((area_cm2 / 1e4) * (horas / 24.0))


def semaforo(conc_p_ml: float) -> str:
    for limite, nombre in UMBRALES:
        if conc_p_ml < limite:
            return nombre
    return UMBRALES[-1][1]


def nivel_mp(pm25: float, pm10: float) -> dict:
    """Nivel chileno para cada contaminante y el peor de los dos."""
    i25 = next(i for i, n in enumerate(NIVELES_MP) if pm25 < n[1])
    i10 = next(i for i, n in enumerate(NIVELES_MP) if pm10 < n[2])
    return {"mp25": NIVELES_MP[i25][0], "mp10": NIVELES_MP[i10][0], "peor": NIVELES_MP[max(i25, i10)][0]}


def advertencias(muestra: float, control: float) -> list:
    """Avisos de calidad para el administrador."""
    avisos = []
    if control > 0:
        avisos.append("El control tiene partículas: posible contaminación al manipular.")
    if control > muestra:
        avisos.append("El control supera a la muestra: revisar el procedimiento.")
    return avisos


def extrapolar(contadas: float, fraccion: float) -> float:
    """Conteo del disco completo a partir de lo contado en una fracción del disco (0 < fracción ≤ 1)."""
    if not 0 < fraccion <= 1:
        raise ValueError("La fracción del disco analizada debe estar entre 1 % y 100 %")
    return contadas / fraccion


def evaluar_calidad(muestras: list, controles: list) -> dict:
    """Criterio de calidad del dato. muestras y controles: listas de (n_contado, fraccion_analizada).

    N = n_contado / fracción; DE = máx(s_c con n − 1, √máx(media de n_contado de los controles, 1) ÷
    fracción media de los controles). "confianza reforzada" si hay ≥ 3 réplicas, ≥ 3 controles y
    media(N_m) > media(N_c) + 3·DE; si no, "exploratorio". Evalúa la calidad del dato, no la química."""
    if not muestras or not controles:
        raise ValueError("Faltan réplicas o controles")
    nm = [extrapolar(n, f) for n, f in muestras]
    nc = [extrapolar(n, f) for n, f in controles]
    media = lambda v: sum(v) / len(v)
    mc = media(nc)
    sd_c = math.sqrt(sum((x - mc) ** 2 for x in nc) / (len(nc) - 1)) if len(nc) > 1 else None
    de_poisson = math.sqrt(max(media([n for n, _ in controles]), 1.0)) / media([f for _, f in controles])
    de = max(sd_c, de_poisson) if sd_c is not None else None
    umbral = mc + 3 * de if de is not None else None
    reforzada = (len(nm) >= 3 and len(nc) >= 3 and umbral is not None
                 and media(nm) > umbral + 1e-9 * max(1.0, abs(umbral)))
    return {"calidad": "confianza reforzada" if reforzada else "exploratorio", "formula": FORMULA_CRITERIO,
            "replicas": len(nm), "controles": len(nc), "media_muestra": media(nm), "media_control": mc,
            "de_control": sd_c, "de_minima_poisson": de_poisson, "de_usada": de, "umbral": umbral}
