"""Prueba completa sin equipo: disco simulado → captura en cuadrícula → cuadrícula → conteo → comparación.

    python -m microalerta_scan.cli prueba --salida prueba
"""
import json
import math
import os

import numpy as np
import cv2

from .captura import (ConfigCaptura, PlatinaSimulada, CamaraSimulada, plan_cuadricula, capturar,
                      armar_cuadricula)
from .simulacion import Escena


def capturar_simulado(escena, fotos, diam, ancho, alto, um_px, semilla=0, antes=None):
    """Fotografía una escena simulada con la platina simulada, como `capturar --modo motores`."""
    cfg = ConfigCaptura(diametro_mm=diam, um_por_px=um_px, campo_ancho_mm=ancho * um_px / 1000,
                        campo_alto_mm=alto * um_px / 1000, solape=0.15, espera_s=0.0, juego_mm=0.3,
                        modos=["F", "W"], luces="arduino", camara_ancho=ancho, camara_alto=alto)
    platina = PlatinaSimulada(cfg, juego_real_mm=0.08, error_mm=0.004, semilla=semilla)
    platina.cero()
    camara = CamaraSimulada(escena, platina, cfg, ancho, alto)
    campos = plan_cuadricula(cfg)
    if antes:
        antes(cfg, platina, campos)
    capturar(campos, platina, camara, cfg, fotos, avisar=lambda *_: None)
    with open(os.path.join(fotos, "captura.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    info["modo"] = "simulado"
    with open(os.path.join(fotos, "captura.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return cfg, platina, campos


def correr_prueba(salida, semilla=0, rapida=False, avisar=print):
    os.makedirs(salida, exist_ok=True)
    if rapida:      # disco chico y menos partículas (para las pruebas automáticas)
        diam, ancho, alto, um_px = 5.0, 648, 486, 2.2
        escena = Escena(diam, semilla, n_fragmentos=10, n_fibras=6, n_pequenas=6, n_grumos=6, n_arena=30,
                        n_algodon=3)
    else:           # cámara de 5 MP con 2×2 píxeles juntos: 1296 × 972 a 2,2 µm/px (campo 2,85 × 2,14 mm)
        diam, ancho, alto, um_px = 21.0, 1296, 972, 2.2
        escena = Escena(diam, semilla)
    fotos = os.path.join(salida, "fotos")
    cfg, platina, campos = capturar_simulado(escena, fotos, diam, ancho, alto, um_px, semilla,
                                             antes=lambda c, pl, cs: avisar(
                                                 f"1/4 Captura simulada: {len(cs)} campos con solape de {c.solape:.0%} "
                                                 f"(platina con {pl.juego * 1000:.0f} µm de juego)"))
    avisar("2/4 Cuadrícula")
    cf = armar_cuadricula(fotos, "F")
    cw = armar_cuadricula(fotos, "W")

    muestra = {
        "muestra_id": "PRUEBA-SIMULADA", "tipo": "muestra", "tiempo_dias": 1.0, "area_captador_m2": 0.00636,
        "filtro_material": "papel", "filtro_diametro_mm": diam, "filtro_completo": False,
        "tincion": "rojo_de_nilo", "deteccion": "fluorescencia", "limite_minimo_um": 20,
        "calibracion": {"um_por_px": um_px}, "imagenes": ["fotos/"],
    }
    ruta_m = os.path.join(salida, "muestra.json")
    with open(ruta_m, "w", encoding="utf-8") as fh:
        json.dump(muestra, fh, indent=2, ensure_ascii=False)

    avisar("3/4 Conteo con el programa")
    from .cli import main as cli_main
    res_dir = os.path.join(salida, "resultado")
    cli_main(["scan", "--config", ruta_m, "--salida", res_dir, "--posiciones", os.path.join(fotos, "campos.csv"),
              "--sin-campo-circular"])
    with open(os.path.join(res_dir, "resultado.json"), encoding="utf-8") as fh:
        res = json.load(fh)

    avisar("4/4 Comparación con lo que de verdad había en el disco")
    informe = comparar(escena, res, um_px)
    marcas = [(q["x"], q["y"], (0, 255, 0) if q["encontrada"] else (0, 0, 255)) for q in informe.pop("_verdad")]
    marcas += [(f["x_filtro_mm"], f["y_filtro_mm"], (255, 0, 255)) for f in informe.pop("_extras")]
    informe["cuadricula_F"] = os.path.relpath(cf, salida)
    informe["cuadricula_W"] = os.path.relpath(cw, salida)
    informe["cuadricula_revisada"] = os.path.relpath(_marcar(cf, fotos, marcas, os.path.join(salida, "cuadricula_revisada.jpg")), salida)
    with open(os.path.join(salida, "informe_prueba.json"), "w", encoding="utf-8") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)
    avisar(texto_informe(informe))
    return informe


def comparar(escena, res, um_px, radio_um=None):
    verdad = escena.verdad(20.0)
    contadas = [f for f in res["particulas"] if f["contada"] and f["x_filtro_mm"] is not None]
    usadas = set()
    for q in verdad:
        r_mm = max(q["largo_um"] / 2000 + 0.03, 0.05) if radio_um is None else radio_um / 1000
        mejor, dmin = None, None
        for i, f in enumerate(contadas):
            d = math.hypot(f["x_filtro_mm"] - q["x"], f["y_filtro_mm"] - q["y"])
            if d <= r_mm and (dmin is None or d < dmin):
                mejor, dmin = i, d
        q["encontrada"] = mejor is not None
        if mejor is not None:
            usadas.add(mejor)
    extras = [f for i, f in enumerate(contadas) if i not in usadas]
    no_plastico = [q for q in escena.p if q["tipo"] in ("arena", "algodon")]
    n_v, n_ok = len(verdad), sum(q["encontrada"] for q in verdad)
    return {
        "verdad_plasticos_que_brillan_20um": n_v,
        "contadas_por_el_programa": res["conteo"]["pcm"]["n_contado"],
        "encontradas": n_ok,
        "perdidas": n_v - n_ok,
        "contadas_de_mas": len(extras),
        "recuperacion": round(n_ok / n_v, 3) if n_v else None,
        "brillan_pero_menores_de_20um": sum(1 for q in escena.p if q["tipo"] == "pequena"),
        "grumos_de_tinte": sum(1 for q in escena.p if q["tipo"] == "grumo"),
        "arena_y_algodon_que_no_brillan": len(no_plastico),
        "campos": len(res["procesamiento"]["campos"]),
        "modo_captura": res["conteo"]["modo_captura"],
        "_verdad": verdad, "_extras": extras,
    }


def _marcar(ruta_cuadricula, carpeta, marcas, salida):
    """Círculos sobre la cuadrícula: verde = encontrada, rojo = perdida, magenta = contada de más."""
    from .captura import leer_plan
    img = cv2.imread(ruta_cuadricula)
    with open(os.path.join(carpeta, "captura.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)["configuracion"]
    campos = leer_plan(os.path.join(carpeta, "campos.csv"))
    wmm, hmm = cfg["camara_ancho"] * cfg["um_por_px"] / 1000, cfg["camara_alto"] * cfg["um_por_px"] / 1000
    xmin = min(c["x_mm"] for c in campos) - wmm / 2
    xmax = max(c["x_mm"] for c in campos) + wmm / 2
    ymin = min(c["y_mm"] for c in campos) - hmm / 2
    esc = img.shape[1] / (xmax - xmin)
    for x, y, col in marcas:
        cv2.circle(img, (int(round((x - xmin) * esc)), int(round((y - ymin) * esc))), 9, col, 2, cv2.LINE_AA)
    cv2.imwrite(salida, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return salida


def texto_informe(i):
    return (f"\nRESULTADO DE LA PRUEBA\n"
            f"  Plásticos que brillan y miden 20 µm o más: {i['verdad_plasticos_que_brillan_20um']}\n"
            f"  El programa contó: {i['contadas_por_el_programa']} "
            f"({i['encontradas']} encontradas, {i['perdidas']} perdidas, {i['contadas_de_mas']} de más)\n"
            f"  No debía contar: {i['brillan_pero_menores_de_20um']} partículas menores de 20 µm, "
            f"{i['grumos_de_tinte']} grumos de tinte y {i['arena_y_algodon_que_no_brillan']} granos de arena "
            f"y fibras de algodón\n"
            f"  Campos: {i['campos']} ({i['modo_captura']})\n"
            f"  Revise {i['cuadricula_revisada']}: verde = encontrada, rojo = perdida, magenta = contada de más.")
