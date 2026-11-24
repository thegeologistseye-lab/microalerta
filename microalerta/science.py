"""Cálculos científicos de MicroAlerta: concentración, TDC, semáforo y control de calidad."""
from .config import UMBRALES, NIVELES_MP


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
