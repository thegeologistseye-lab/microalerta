"""Conteo por muestra, extrapolación al filtro completo, TDC y criterio de calidad del dato.

TDC = máx(0, Nm − Nc) / (A × t)   [PCM / (m²·día)]   (una muestra)
  Nm, Nc: PCM de la muestra y del control (estimados sobre el filtro completo)
En una campaña, cada réplica guarda su diferencia corregida (Nm − N̄c) / (A × t) SIN truncar; la TDC
del punto es máx(0, media de esas diferencias). Truncar cada réplica antes de promediar sube la media
cuando los conteos están cerca del fondo. Si la media de N_m no pasa N̄c + 3·DE, el punto queda
«bajo el límite de detección», que es 3·DE / (A × t).
  A: área efectiva del CAPTADOR (placa expuesta), no del filtro
  t: tiempo de exposición en días
"""
import math
import numpy as np
from scipy import stats


def ic95_poisson(n):
    """Intervalo exacto (Garwood) para un conteo de Poisson."""
    lo = 0.0 if n == 0 else stats.chi2.ppf(0.025, 2 * n) / 2
    hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2
    return float(lo), float(hi)


def fraccion_analizada(area_campos_mm2, filtro_diametro_mm, filtro_completo=False):
    if filtro_completo:
        return 1.0, None
    if not filtro_diametro_mm or area_campos_mm2 is None:
        return None, ("Falta el diámetro efectivo del filtro o la calibración µm/px: no se puede "
                      "extrapolar el conteo al filtro completo ni calcular la TDC.")
    area_filtro = math.pi * (filtro_diametro_mm / 2) ** 2
    f = area_campos_mm2 / area_filtro
    aviso = None
    if f > 1.0:
        aviso = ("El área fotografiada supera el área del filtro: probablemente hay campos repetidos "
                 "o solapados. Se usa fracción = 1 y el conteo puede estar inflado.")
        f = 1.0
    elif f < 0.10:
        aviso = (f"Solo se analizó el {f:.1%} del filtro: la extrapolación tiene alta incertidumbre "
                 "(distribución no homogénea de partículas).")
    return f, aviso


def resumen_conteo(n, fraccion):
    lo, hi = ic95_poisson(n)
    if fraccion:
        return {"n_contado": n, "n_estimado_filtro": n / fraccion,
                "ic95_filtro": [lo / fraccion, hi / fraccion]}
    return {"n_contado": n, "n_estimado_filtro": None, "ic95_filtro": None}


def diferencia_corregida(n_m, n_c, area_m2, tiempo_dias):
    """(Nm − Nc) / (A × t) sin truncar: puede ser negativa si la muestra queda bajo el fondo."""
    if n_m is None or n_c is None or not area_m2 or not tiempo_dias:
        return None
    return (n_m - n_c) / (area_m2 * tiempo_dias)


def tdc(n_m, n_c, area_m2, tiempo_dias):
    d = diferencia_corregida(n_m, n_c, area_m2, tiempo_dias)
    return None if d is None else max(0.0, d)


FORMULA_CRITERIO = ("confianza reforzada si réplicas >= 3, controles >= 3 y media(N_m) > media(N_c) + 3*DE; "
                    "N = n_contado / fraccion_analizada; DE = max(s_c con n-1, sqrt(max(media n_contado controles, 1)) "
                    "/ fraccion media de los controles); si no, exploratorio")


def evaluar_campana(resultados_muestra, resultados_control):
    """Combina réplicas y controles (salidas JSON de `scan`) y aplica el criterio de calidad.

    Fórmula exacta (la misma que debe mostrar la app):

        N_m,i  conteo de la réplica i extendido al disco completo = n_contado ÷ f   (n_estimado_filtro)
        N_c,j  lo mismo para el control j (placa control cerrada)
        N̄_m, N̄_c   medias de réplicas y de controles
        s_c    desviación estándar muestral de los N_c,j (con n − 1)
        σ_P    piso de Poisson = √max(media de los n_contado de los controles, 1) ÷ f̄_c
               (f̄_c = fracción media del disco analizada en los controles)
        DE     = max(s_c, σ_P)

        "confianza reforzada"  si  réplicas ≥ 3  y  controles ≥ 3  y  N̄_m > N̄_c + 3·DE
        "exploratorio"         en cualquier otro caso

    Ejemplo: controles en 0 con el disco completo (f = 1) → DE = 1 y la media de las réplicas debe
    pasar de 3 partículas. Con el celular (f ≈ 0,4) → DE = 2,5 en el disco y la media debe pasar de
    7,5, es decir, de 3 partículas contadas. Evalúa la calidad del dato, no la identidad química."""
    def n_y_base(r):
        c = r["conteo"]
        return c["pcm"]["n_estimado_filtro"], c["base_conteo"], r["muestra"]

    m = [n_y_base(r) for r in resultados_muestra]
    c = [n_y_base(r) for r in resultados_control]
    avisos = []
    bases = {b for _, b, _ in m + c}
    if len(bases) > 1:
        avisos.append("Muestras y controles se contaron con bases distintas (modelo vs. candidatas). No son comparables.")
    if any(n is None for n, _, _ in m + c):
        return {"tdc": None, "calidad": "incompleto",
                "avisos": avisos + ["Hay muestras sin conteo extrapolable (falta diámetro del filtro o calibración)."]}

    nm = np.array([n for n, _, _ in m], float)
    nc = np.array([n for n, _, _ in c], float)
    nc_ref = float(nc.mean())
    filas = []
    for (n, _, meta) in m:
        filas.append({"muestra_id": meta.get("muestra_id"), "n_m": n, "n_c_ref": nc_ref,
                      "area_captador_m2": meta.get("area_captador_m2"), "tiempo_dias": meta.get("tiempo_dias"),
                      "diferencia_corregida": diferencia_corregida(n, nc_ref, meta.get("area_captador_m2"),
                                                                   meta.get("tiempo_dias")),
                      "tdc": tdc(n, nc_ref, meta.get("area_captador_m2"), meta.get("tiempo_dias"))})
    # la media y la dispersión se calculan con las diferencias SIN truncar
    valores = np.array([f["diferencia_corregida"] for f in filas if f["diferencia_corregida"] is not None], float)
    if len(valores) < len(filas):
        avisos.append("Falta área del captador o tiempo de exposición en alguna muestra.")

    sd_c = float(nc.std(ddof=1)) if len(nc) > 1 else None
    # Piso de Poisson: con pocos controles (a menudo todos en 0) la DE observada puede ser 0 y
    # cualquier muestra con 1 partícula pasaría el umbral. Se usa como mínimo la DE de un conteo
    # de Poisson con media ≥ 1, llevada a la escala del filtro completo (÷ fracción analizada).
    def _num(r, *ruta):
        v = r
        for k in ruta:
            v = (v or {}).get(k)
        return v
    crudos = [_num(r, "conteo", "pcm", "n_contado") for r in resultados_control]
    fracs = [_num(r, "conteo", "fraccion_filtro_analizada") for r in resultados_control]
    media_cruda = float(np.mean(crudos)) if all(v is not None for v in crudos) else nc_ref
    frac_media = float(np.mean(fracs)) if all(v for v in fracs) else 1.0
    de_poisson = math.sqrt(max(media_cruda, 1.0)) / frac_media
    de_ef = max(sd_c, de_poisson) if sd_c is not None else None
    criterio = {"formula": FORMULA_CRITERIO, "replicas": len(nm), "controles": len(nc),
                "media_muestra": float(nm.mean()), "media_control": nc_ref, "de_control": sd_c,
                "de_minima_poisson": de_poisson, "de_usada": de_ef,
                "umbral_media_control_mas_3de": (nc_ref + 3 * de_ef) if de_ef is not None else None}
    # comparación estricta; la tolerancia evita que el redondeo decida un empate exacto (p. ej. 7,5 vs 7,5)
    umbral_c = nc_ref + 3 * de_ef if de_ef is not None else None
    reforzada = (len(nm) >= 3 and len(nc) >= 3 and umbral_c is not None
                 and nm.mean() > umbral_c + 1e-9 * max(1.0, abs(umbral_c)))
    if sd_c is not None and de_poisson > sd_c:
        avisos.append("La dispersión de los controles es menor que la esperable por azar (conteos bajos): "
                      "el umbral usa la desviación mínima de Poisson.")
    if "candidatas_sin_clasificar" in bases:
        avisos.append("Conteo sin modelo: la TDC corresponde a CANDIDATAS detectadas, no a PCM confirmadas.")
    if any(n > mm for n in nc for mm in nm):
        avisos.append("Algún control supera a alguna muestra: revisar el procedimiento (posible contaminación).")
    # A·t común (si las réplicas difieren, se usa la media) para llevar DE y límite a PCM/(m²·día)
    at = [f["area_captador_m2"] * f["tiempo_dias"] for f in filas if f["area_captador_m2"] and f["tiempo_dias"]]
    at_ref = float(np.mean(at)) if at else None
    sobre_umbral = umbral_c is not None and nm.mean() > umbral_c + 1e-9 * max(1.0, abs(umbral_c))
    bajo_ld = umbral_c is not None and not sobre_umbral
    if bajo_ld:
        avisos.append("La media de las réplicas no supera el fondo de los controles + 3·DE: el punto se informa "
                      "como «bajo el límite de detección», no como una deposición medida.")
    res = {
        "tdc_por_replica": filas,
        "tdc_media": max(0.0, float(valores.mean())) if len(valores) else None,
        "diferencia_media_sin_truncar": float(valores.mean()) if len(valores) else None,
        "bajo_limite_deteccion": bool(bajo_ld),
        "limite_deteccion_tdc": (3 * de_ef / at_ref) if (de_ef is not None and at_ref) else None,
        "de_control_tdc": (de_ef / at_ref) if (de_ef is not None and at_ref) else None,
        "tdc_de": float(valores.std(ddof=1)) if len(valores) > 1 else None,
        "desviacion_relativa": float(valores.std(ddof=1) / valores.mean())
        if len(valores) > 1 and valores.mean() > 0 else None,
        "cv_porcentaje": float(100 * valores.std(ddof=1) / valores.mean())
        if len(valores) > 1 and valores.mean() > 0 else None,
        "unidad": "PCM/(m²·día)",
        "base_conteo": next(iter(bases)) if len(bases) == 1 else "mixta",
        "calidad": "confianza reforzada" if reforzada else "exploratorio",
        "criterio_calidad": criterio,
        "nota": "La TDC no representa concentración en aire, PM2.5, PM10 ni dosis inhalada.",
        "avisos": avisos,
    }
    return res
