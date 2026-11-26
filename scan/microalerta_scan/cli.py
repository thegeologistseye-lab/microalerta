"""Línea de comandos.

  python -m microalerta_scan.cli scan  --config muestra.json --salida out/
  python -m microalerta_scan.cli tdc   --muestras out_m1/resultado.json ... --controles out_c1/resultado.json ...
  python -m microalerta_scan.cli calibrar --imagen micrometro.jpg --um-entre-marcas 10 --guardar captura.json
  python -m microalerta_scan.cli capturar --modo motores|manual|simulado --config captura.json --salida fotos/
  python -m microalerta_scan.cli ordenar  --fotos celular/ --salida fotos/
  python -m microalerta_scan.cli cuadricula --carpeta fotos/
  python -m microalerta_scan.cli calibrar-platina --puerto COM3 --camara 0 --config captura.json
  python -m microalerta_scan.cli umbral   --blancos fotos/B1 fotos/B2 fotos/B3 --referencia fotos/PRUEBA --guardar parametros_campana.json
  python -m microalerta_scan.cli prueba   --salida prueba/
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

from . import __version__
from . import config as cfg
from .calibracion import corregir_flat, um_por_px_desde_campo, um_por_px_desde_micrometro
from .vision import detectar_campo, segmentar, alinear
from .rasgos import medir
from .muestra import fraccion_analizada, resumen_conteo, evaluar_campana
from .mosaico import Campo, resolver_solapes
from .fluorescencia import segmentar_fluorescencia, color_fluorescencia

EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
IGNORAR = ("cuadricula", "vista_previa", "grilla_manual")
COLORES = {"fibra": (255, 0, 255), "fragmento": (0, 255, 0), "pelicula": (0, 220, 255)}


def mascara_disco(mask, pos, um_px, d_disco, p):
    """Deja en la máscara solo lo que cae dentro del círculo de filtrado (centro del disco = 0, 0):
    el borde del depósito (huella del empaque) deja un escalón de brillo que no es partícula."""
    hh, ww = mask.shape[:2]
    xs = pos[0] + (np.arange(ww) - ww / 2) * um_px / 1000
    ys = pos[1] + (np.arange(hh) - hh / 2) * um_px / 1000
    r_util = d_disco / 2 - p.get("margen_disco_mm", 0.15)
    dentro = (xs[None, :] ** 2 + ys[:, None] ** 2) <= r_util ** 2
    return np.where(dentro, mask, 0).astype(np.uint8)


def mascara_cuadro(shape):
    """Cuadro completo con 2 px de borde en cero: así se marcan las partículas cortadas por el borde."""
    mask = np.full(shape[:2], 255, np.uint8)
    mask[:2, :] = mask[-2:, :] = 0
    mask[:, :2] = mask[:, -2:] = 0
    return mask


def leer(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"No se pudo leer {path}")
    return img


def agrupar_imagenes(entradas, mm):
    """Agrupa archivos por campo: base_W (blanca), base_P (polarizada), base_F (fluorescencia).
    Archivos sin sufijo se tratan como luz blanca."""
    archivos = []
    for e in entradas:
        if os.path.isdir(e):
            archivos += [os.path.join(e, f) for f in sorted(os.listdir(e)) if f.lower().endswith(EXT)]
        else:
            archivos += sorted(glob.glob(e))
    # imágenes que arma el propio programa (cuadrícula, vista previa, hoja de la grilla): no son campos
    archivos = [f for f in archivos if not os.path.basename(f).lower().startswith(IGNORAR)]
    sw, sp, sf = mm.get("sufijo_blanca", "_W"), mm.get("sufijo_pol", "_P"), mm.get("sufijo_fluo", "_F")
    pat = re.compile(rf"^(.*?)({re.escape(sw)}|{re.escape(sp)}|{re.escape(sf)})$")
    grupos = {}
    for a in archivos:
        stem = os.path.splitext(os.path.basename(a))[0]
        m = pat.match(stem)
        base, suf = (m.group(1), m.group(2)) if m else (stem, sw)
        clave = os.path.join(os.path.dirname(a), base)
        g = grupos.setdefault(clave, {})
        g[{sw: "blanca", sp: "pol", sf: "fluo"}[suf]] = a
    return {k: v for k, v in grupos.items() if "blanca" in v or "fluo" in v}


def limite_minimo_um(p, um_px):
    """Tamaño mínimo medible (µm, diámetro equivalente): el mayor entre el área mínima
    aceptada por la segmentación y ~3 píxeles de ancho."""
    if not um_px:
        return None
    area_um2 = max(p["area_min_um2"], 4 * um_px ** 2)
    return round(max(2 * math.sqrt(area_um2 / math.pi), 3 * um_px), 2)


def dhash(img, n=16):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def dibujar(img, etiquetas, filas, mask_campo, unidad):
    out = img.copy()
    cnts, _ = cv2.findContours(mask_campo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    for f in filas:
        comp = (etiquetas == f["id"]).astype(np.uint8)
        c, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = COLORES[f["forma"]]
        if f.get("p_pcm") is not None and f["p_pcm"] < 0.5:
            col = (128, 128, 128)
        cv2.drawContours(out, c, -1, col, 2)
        x0, y0 = f["bbox_px"][0], f["bbox_px"][1]
        txt = f'{f["id"]} {f["forma"][:4]} {f["largo_um"]:.0f}{unidad}'
        cv2.putText(out, txt, (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def recortar(img, f, ruta, pad=12, img_w=None):
    """Recorte de la partícula. Con foto de luz blanca del mismo campo, la pone al lado (F | W)."""
    x0, y0, x1, y1 = f["bbox_px"]
    h, w = img.shape[:2]
    sl = (slice(max(y0 - pad, 0), min(y1 + pad, h)), slice(max(x0 - pad, 0), min(x1 + pad, w)))
    rec = img[sl]
    if img_w is not None and img_w.shape[:2] == img.shape[:2]:
        sep = np.full((rec.shape[0], 3, 3), 255, np.uint8)
        rec = np.hstack([rec, sep, img_w[sl]])
    cv2.imwrite(ruta, rec)


def cmd_scan(a):
    meta = {}
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            meta = json.load(fh)
    avisos_inicio = []
    # Parámetros de la campaña (umbral de fluorescencia de los blancos): por --parametros o por la clave
    # "parametros_campana" de la ficha (ruta relativa a la ficha). Así no se copian a mano en cada muestra.
    ruta_param = a.parametros
    if not ruta_param and meta.get("parametros_campana"):
        ruta_param = meta["parametros_campana"]
        if not os.path.isabs(ruta_param):
            ruta_param = os.path.join(os.path.dirname(os.path.abspath(a.config)), ruta_param)
        if not os.path.exists(ruta_param):
            sys.exit(f"La ficha pide {ruta_param} y no existe: corra antes el comando umbral con los blancos.")
    p = cfg.cargar(ruta_param)
    campana = {}
    if ruta_param:
        with open(ruta_param, encoding="utf-8") as fh:
            campana = json.load(fh)
    if a.sin_campo_circular:
        p["detectar_campo_circular"] = False
    deteccion = a.deteccion or meta.get("deteccion", "oscuras")
    fluo_modo = deteccion == "fluorescencia"
    for k in ("fluo_umbral_min", "limite_minimo_um"):
        if k in meta and meta[k] is not None:
            if k in campana and campana[k] != meta[k]:
                avisos_inicio.append(f"{k}: la ficha dice {meta[k]} y {os.path.basename(ruta_param)} dice "
                                     f"{campana[k]}; se usa {campana[k]}, el de la puesta a punto vigente.")
            else:
                p[k] = meta[k]
    origen_umbral = (os.path.basename(ruta_param) if "fluo_umbral_min" in campana
                     else "ficha" if meta.get("fluo_umbral_min") is not None else "de fábrica")
    lim_inf = p.get("limite_minimo_um")
    if fluo_modo:
        # en fluorescencia la textura del papel no molesta: las fibras cortas (≥ 20 µm) también cuentan
        p["largo_min_fibra_um"] = p.get("fluo_largo_min_fibra_um", 20.0)
    cal = meta.get("calibracion", {}) or {}
    um_px_fijo = a.um_por_px
    um_ficha = cal.get("um_por_px")
    diam_campo_um = a.diam_campo_um or cal.get("diametro_campo_um")
    entradas = a.imagenes or meta.get("imagenes") or []
    if a.config:
        base_cfg = os.path.dirname(os.path.abspath(a.config))
        entradas = [e if os.path.isabs(e) else os.path.join(base_cfg, e) for e in entradas]
    grupos = agrupar_imagenes(entradas, meta.get("multimodal", {}))
    if not grupos:
        sys.exit("No se encontraron imágenes.")

    flat = leer(a.flat) if a.flat else None
    dark = leer(a.dark) if a.dark else None
    modelo = None
    if a.modelo:
        from .clasificador import cargar_modelo, predecir
        modelo = cargar_modelo(a.modelo)

    posiciones = {}
    if a.posiciones:
        dfp = pd.read_csv(a.posiciones)
        posiciones = {str(r.campo): (float(r.x_mm), float(r.y_mm)) for r in dfp.itertuples()}
        # la escala calibrada de la cámara queda en captura.json, junto a campos.csv: no hay que copiarla
        # (orden: --um-por-px > captura.json de la captura con motores > calibracion de la ficha)
        ruta_cap = os.path.join(os.path.dirname(os.path.abspath(a.posiciones)), "captura.json")
        if um_px_fijo is None and os.path.exists(ruta_cap):
            with open(ruta_cap, encoding="utf-8") as fh:
                cap = json.load(fh)
            if cap.get("modo") in ("motores", "simulado") and cap.get("configuracion", {}).get("um_por_px"):
                um_px_fijo = float(cap["configuracion"]["um_por_px"])
                print(f"Escala tomada de {ruta_cap}: {um_px_fijo:g} µm/px")
                if um_ficha and abs(float(um_ficha) - um_px_fijo) > 0.01 * um_px_fijo:
                    avisos_inicio.append(f"Escala: la ficha dice {um_ficha} µm/px y captura.json dice {um_px_fijo:g}; "
                                         "se usa la de captura.json (calibración con que se sacaron las fotos).")
    if um_px_fijo is None and um_ficha:
        um_px_fijo = float(um_ficha)

    os.makedirs(os.path.join(a.salida, "superpuestas"), exist_ok=True)
    os.makedirs(os.path.join(a.salida, "recortes"), exist_ok=True)

    todas, info_imgs, avisos = [], [], list(avisos_inicio)
    for av in avisos_inicio:
        print("AVISO:", av)
    campos_mosaico = []
    area_campos_mm2 = 0.0
    hashes = {}
    for clave, g in grupos.items():
        nombre = os.path.basename(clave)
        img_w = None
        if fluo_modo:
            if "fluo" not in g:
                avisos.append(f"{nombre}: no tiene foto de fluorescencia (_F) y se omitió.")
                continue
            img = leer(g["fluo"])
            img_w = leer(g["blanca"]) if "blanca" in g else None
            principal = g["fluo"]
        else:
            if "blanca" not in g:
                continue
            img = leer(g["blanca"])
            principal = g["blanca"]
        # Fotos repetidas: si el campo tiene posición de platina no hace falta revisarlo. En
        # fluorescencia se compara la foto blanca: los fondos oscuros se parecen todos entre sí.
        if nombre not in posiciones:
            ref = img_w if img_w is not None else img
            hsh = dhash(ref)
            dup = next((k for k, hv in hashes.items() if np.count_nonzero(hv != hsh) <= a.tolerancia_duplicado), None)
            if dup and not a.incluir_duplicados:
                avisos.append(f"{nombre} parece la misma foto que {dup}: se omitió para no contar dos veces.")
                print(f"{nombre}: DUPLICADO de {dup}, omitido")
                continue
            hashes[nombre] = hsh
        if flat is not None and not fluo_modo:
            img = corregir_flat(img, cv2.resize(flat, img.shape[1::-1]),
                                cv2.resize(dark, img.shape[1::-1]) if dark is not None else None)
        if p["detectar_campo_circular"]:
            mask, circ = detectar_campo(img_w if img_w is not None else img, p["margen_campo"])
        else:
            mask, circ = mascara_cuadro(img.shape), None

        um_px, origen = um_px_fijo, "fija"
        if um_px is None and diam_campo_um and circ:
            um_px, origen = um_por_px_desde_campo(diam_campo_um, 2 * circ[2]), "diametro_campo"
        if um_px is None:
            origen = "sin_calibrar (px)"
        # Con posiciones (centro del disco = 0, 0) se analiza solo dentro del círculo de filtrado:
        # el borde del depósito (huella del empaque) deja un escalón de brillo que no es partícula.
        d_disco = meta.get("filtro_diametro_mm")
        if nombre in posiciones and um_px and d_disco:
            mask = mascara_disco(mask, posiciones[nombre], um_px, d_disco, p)
            if not mask.any():
                avisos.append(f"{nombre}: queda fuera del círculo de filtrado y no se analizó.")
                continue

        pol = fluo = None
        reg_info = {}
        if fluo_modo:
            fluo = img
            etiquetas, fibra, razon, umbrales, mask = segmentar_fluorescencia(img, mask, p, um_px)
        else:
            if "pol" in g:
                pol, _, cc = alinear(img, leer(g["pol"]))
                reg_info["correlacion_pol"] = cc
            if "fluo" in g:
                fluo, _, cc = alinear(img, leer(g["fluo"]))
                reg_info["correlacion_fluo"] = cc
            etiquetas, fibra, razon, umbrales, mask = segmentar(img, mask, p, um_px)
        filas = medir(etiquetas, img, razon, fibra, mask, p, um_px, pol, fluo)
        for f in filas:
            f["color_fluo"] = f["razon_rg_fluo"] = None
            if fluo_modo:
                x0, y0, x1, y1 = f["bbox_px"]
                comp = etiquetas[y0:y1, x0:x1] == f["id"]
                f["razon_rg_fluo"], f["color_fluo"] = color_fluorescencia(img, comp, y0, x0, y1, x1)
            f["bajo_limite"] = bool(lim_inf and um_px and f["largo_um"] < lim_inf)
        if modelo is not None:
            for f, pr in zip(filas, predecir(modelo, filas)):
                f["p_pcm"] = pr
                f["es_pcm"] = bool(pr >= p["umbral_p_pcm"])
        unidad = "um" if um_px else "px"
        for f in filas:
            f["imagen"] = os.path.basename(principal)
            f["campo"] = nombre
            f["muestra_id"] = meta.get("muestra_id")
            rc = os.path.join("recortes", f"{nombre}_p{f['id']:04d}.png")
            recortar(img, f, os.path.join(a.salida, rc), img_w=img_w)
            f["recorte"] = rc
            # coordenadas para ir a buscar la partícula al Raman: centro de la imagen = posición de la platina
            f["x_filtro_mm"] = f["y_filtro_mm"] = None
            if nombre in posiciones and um_px:
                hh, ww = img.shape[:2]
                f["x_filtro_mm"] = posiciones[nombre][0] + (f["cx_px"] - ww / 2) * um_px / 1000
                f["y_filtro_mm"] = posiciones[nombre][1] + (f["cy_px"] - hh / 2) * um_px / 1000
            f.setdefault("p_pcm", None)
            f.setdefault("es_pcm", None)
            f["etiqueta"] = ""                  # para completar a mano / con Raman
        cv2.imwrite(os.path.join(a.salida, "superpuestas", f"{nombre}.jpg"),
                    dibujar(img, etiquetas, filas, mask, unidad), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if nombre in posiciones and um_px:
            campos_mosaico.append(Campo(nombre, posiciones[nombre][0], posiciones[nombre][1], um_px, mask))
        area_campo_px = float((mask > 0).sum())
        area_mm2 = area_campo_px * (um_px ** 2) / 1e6 if um_px else None
        if area_mm2 is not None and area_campos_mm2 is not None:
            area_campos_mm2 += area_mm2
        else:
            area_campos_mm2 = None
        info_imgs.append({"campo": nombre, "archivos": g, "um_por_px": um_px, "origen_calibracion": origen,
                          "limite_minimo_um": limite_minimo_um(p, um_px),
                          "circulo_campo_px": circ, "area_campo_mm2": area_mm2, "n_particulas": len(filas),
                          "umbrales": umbrales,
                          **reg_info})
        todas += filas
        print(f"{nombre}: {len(filas)} partículas ({origen})")

    # ---------------- mosaico: campos solapados
    for f in todas:
        f["fuera_de_su_zona"] = False
    if len(campos_mosaico) > 1 and len(campos_mosaico) == len(info_imgs):
        areas, n_solapes = resolver_solapes(campos_mosaico, todas)
        for i in info_imgs:
            i["area_propia_mm2"] = areas.get(i["campo"])
        area_campos_mm2 = sum(areas.values())
        n_fuera = sum(f["fuera_de_su_zona"] for f in todas)
        n_piezas = sum(1 for f in todas if f.get("pieza_representante"))
        print(f"Mosaico: {n_solapes} pares de campos solapados; {n_fuera} partículas vistas en un campo vecino "
              f"se cuentan solo una vez; {n_piezas} objetos más largos que el solape se cuentan una vez por sus trozos.")
        if n_piezas:
            avisos.append(f"{n_piezas} partículas (sobre todo fibras largas) no caben enteras en ningún campo: "
                          "se contaron una vez, pero su largo medido es el del trozo más grande.")
    elif len(info_imgs) > 1 and campos_mosaico:
        avisos.append("Solo algunos campos tienen posición de platina: no se pudieron resolver los solapes.")

    if len(campos_mosaico) > 1 and len(campos_mosaico) == len(info_imgs):
        modo_captura = "mosaico_con_posiciones"      # los solapes se resuelven solos
    elif len(info_imgs) > 1:
        modo_captura = "manual_sin_solape"           # exige campos que no se repitan ni se pisen
    else:
        modo_captura = "campo_unico"

    # ---------------- conteo y extrapolación
    base = "pcm_modelo" if modelo is not None else "candidatas_sin_clasificar"
    def cuenta(f):
        if f.get("bajo_limite"):
            return False
        if f.get("pieza_representante"):
            return True                          # objeto más largo que el solape: se cuenta una vez
        if not p["contar_borde"] and f["toca_borde_campo"]:
            return False
        if f["fuera_de_su_zona"]:
            return False
        return f["es_pcm"] if modelo is not None else True
    for f in todas:
        f["contada"] = bool(cuenta(f))
    n = sum(f["contada"] for f in todas)
    n_borde = sum(1 for f in todas if f["toca_borde_campo"])
    frac, aviso = fraccion_analizada(area_campos_mm2, meta.get("filtro_diametro_mm"),
                                     meta.get("filtro_completo", False))
    if aviso:
        avisos.append(aviso)
    if modelo is None:
        avisos.append("Sin modelo entrenado: el conteo son CANDIDATAS (partículas detectadas), no PCM. "
                      "Etiquete particulas.csv y entrene con clasificador.py.")
    if len(grupos) > 1 and not (len(campos_mosaico) > 1 and len(campos_mosaico) == len(info_imgs)):
        avisos.append("Se asume que los campos no se solapan ni se repiten; fotos repetidas del mismo "
                      "campo inflan el conteo.")
    por_forma = {k: sum(1 for f in todas if f["forma"] == k and f["contada"]) for k in COLORES}
    if n_borde and not p["contar_borde"]:
        avisos.append(f"{n_borde} partículas tocan el borde del área analizada y no se contaron "
                      "(pueden estar cortadas o ser artefactos del borde del filtro).")

    lims = [i["limite_minimo_um"] for i in info_imgs]
    lim_global = None if (not lims or None in lims) else max(lims)
    if lim_global is not None and lim_inf:
        lim_global = max(lim_global, float(lim_inf))
    n_bajo = sum(1 for f in todas if f.get("bajo_limite"))
    if n_bajo:
        avisos.append(f"{n_bajo} partículas miden menos de {lim_inf:g} µm de largo y no se contaron "
                      "(límite de informe).")
    d_filtro = meta.get("filtro_diametro_mm")
    area_filtro = math.pi * (d_filtro / 2) ** 2 if d_filtro else None
    if lim_global is None:
        avisos.append("Sin calibración µm/px: no se conoce el tamaño mínimo medible.")
    resultado = {
        "esquema": "microalerta-scan/0.1",
        "muestra": {k: v for k, v in meta.items() if k not in ("imagenes",)},
        "procesamiento": {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "version": __version__,
                          "modelo": os.path.basename(a.modelo) if a.modelo else None,
                          "flat_field": bool(a.flat), "deteccion": deteccion, "parametros": p, "campos": info_imgs},
        "conteo": {"base_conteo": base, "n_detectadas": len(todas), "n_tocan_borde": n_borde,
                   "pcm": resumen_conteo(n, frac), "por_forma": por_forma,
                   "fraccion_filtro_analizada": frac, "area_campos_mm2": area_campos_mm2,
                   "area_escaneada_mm2": area_campos_mm2, "area_filtro_efectiva_mm2": area_filtro,
                   "filtro_completo": bool(meta.get("filtro_completo", False)),
                   "limite_minimo_um": lim_global, "modo_captura": modo_captura, "deteccion": deteccion,
                   **({"umbral_fluorescencia": p.get("fluo_umbral_min"), "origen_umbral": origen_umbral}
                      if fluo_modo else {})},
        "particulas": [{k: f[k] for k in ("campo", "id", "forma", "largo_um", "ancho_um", "area_um2",
                                           "razon_aspecto", "p_pcm", "es_pcm", "color_fluo", "toca_borde_campo",
                                           "bajo_limite", "fuera_de_su_zona", "contada",
                                           "cx_px", "cy_px", "x_filtro_mm", "y_filtro_mm", "recorte", "unidad")}
                       for f in todas],
        "avisos": avisos,
    }
    with open(os.path.join(a.salida, "resultado.json"), "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, indent=2, ensure_ascii=False, default=float)
    cols = ["muestra_id", "campo", "imagen", "id", "forma", "etiqueta", "p_pcm", "es_pcm", "largo_um",
            "ancho_um", "area_um2", "diam_equiv_um", "feret_max_um", "perimetro_um", "razon_aspecto",
            "circularidad", "solidez", "excentricidad", "rugosidad_borde", "curvatura", "contraste_medio",
            "contraste_max", "dL", "da", "db", "uniformidad_L", "uniformidad_color", "frac_cresta",
            "indice_birrefringencia", "birrefringencia_p95", "frac_brillante_pol", "indice_fluorescencia",
            "color_fluo", "razon_rg_fluo", "toca_borde_campo", "bajo_limite", "fuera_de_su_zona", "contada", "cx_px", "cy_px",
            "x_filtro_mm", "y_filtro_mm",
            "unidad", "recorte"]
    pd.DataFrame(todas).reindex(columns=cols).to_csv(os.path.join(a.salida, "particulas.csv"), index=False)
    galeria(todas, os.path.join(a.salida, "galeria.html"), meta.get("muestra_id"), fluo_modo)
    print(f"\nTotal: {len(todas)} detectadas, {n} contadas | base: {base} | resultados en {a.salida}")
    for av in avisos:
        print("AVISO:", av)


def galeria(filas, ruta, titulo, fluo_modo=False):
    """Página para revisar y etiquetar recortes: muestra ID, forma y medidas de cada partícula."""
    import html
    tarjetas = []
    for f in filas:
        u = "µm" if f["unidad"] == "um" else "px"
        pp = "" if f.get("p_pcm") is None else f' · p(PCM)={f["p_pcm"]:.2f}'
        if f.get("color_fluo"):
            pp += f' · brilla {f["color_fluo"]}'
        if not f.get("contada", True):
            pp += " · no contada"
        tarjetas.append(
            f'<figure><img src="{html.escape(f["recorte"])}" loading="lazy">'
            f'<figcaption><b>{html.escape(f["campo"])} #{f["id"]}</b><br>{f["forma"]} · '
            f'{f["largo_um"]:.0f}×{f["ancho_um"]:.0f} {u}{pp}</figcaption></figure>')
    doc = f"""<!DOCTYPE html><html lang="es"><meta charset="utf-8"><title>Galería {html.escape(str(titulo or ""))}</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f6f4}}h1{{font-size:18px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;padding:6px}}img{{width:100%;height:150px;object-fit:contain;background:#222}}
figcaption{{font-size:12px;line-height:1.35}}</style>
<h1>Galería de partículas — {html.escape(str(titulo or ""))} ({len(filas)})</h1>
<p>Para etiquetar: abra <code>particulas.csv</code> y escriba en la columna <code>etiqueta</code> PCM, organico, mineral,
{"grumo_tinte" if fluo_modo else "residuo_sal"} u otro según el campo y el número (#) de cada recorte.
{"Cada recorte muestra la foto con luz azul (izquierda) y, si existe, la de luz blanca (derecha)." if fluo_modo else ""}</p><main>{''.join(tarjetas)}</main></html>"""
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(doc)


def cmd_tdc(a):
    def cargar(lista):
        out = []
        for pth in lista:
            with open(pth, encoding="utf-8") as fh:
                out.append(json.load(fh))
        return out
    res = evaluar_campana(cargar(a.muestras), cargar(a.controles))
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)


def cmd_calibrar(a):
    img = leer(a.imagen)
    um_px, info = um_por_px_desde_micrometro(img, a.um_entre_marcas)
    print(json.dumps({"um_por_px": um_px, **info}, indent=2))
    if info["calidad_pico"] < 20:
        print("AVISO: pico de frecuencia débil; verifique midiendo dos marcas a mano.")
        if a.guardar:
            print("No se guardó: repita la foto del micrómetro (más enfocada o con más divisiones).")
            return
    if a.guardar:
        from .captura import ConfigCaptura, guardar_en_config
        actual = ConfigCaptura.cargar(a.guardar if os.path.exists(a.guardar) else None)
        um = round(float(um_px), 5)
        alto_px, ancho_px = img.shape[:2]
        if (ancho_px, alto_px) != (actual.camara_ancho, actual.camara_alto):
            print(f"AVISO: la foto mide {ancho_px}×{alto_px} px y captura.json dice {actual.camara_ancho}×"
                  f"{actual.camara_alto}: saque la foto del micrómetro con la misma cámara y resolución.")
        # el campo de la cámara sale de la escala y del tamaño de la foto: se actualiza junto con ella
        cambios = {"um_por_px": um,
                   "campo_ancho_mm": round(ancho_px * um / 1000, 4),
                   "campo_alto_mm": round(alto_px * um / 1000, 4)}
        guardar_en_config(a.guardar, cambios, "micrometro",
                          {"imagen": os.path.basename(a.imagen), "calidad_pico": info["calidad_pico"]})
        print(f"Guardado en {a.guardar}: {cambios}")


def _picos_carpeta(carpeta, p, um_px_cli, limite_um):
    """Brillo de las manchas de ≥ limite_um de cada foto _F de una carpeta (blanco o referencia).
    Usa captura.json y campos.csv si están: escala, cuadro completo y solo el círculo de filtrado."""
    from .fluorescencia import picos_fluorescencia
    cap = {}
    if os.path.exists(os.path.join(carpeta, "captura.json")):
        with open(os.path.join(carpeta, "captura.json"), encoding="utf-8") as fh:
            cap = json.load(fh)
    conf = cap.get("configuracion", {})
    motores = cap.get("modo") in ("motores", "simulado")
    um_px = um_px_cli or (conf.get("um_por_px") if motores else None)
    if not um_px:
        sys.exit(f"{carpeta}: falta la escala. Use --um-por-px (las fotos con motores la traen en captura.json).")
    posiciones = {}
    if motores and os.path.exists(os.path.join(carpeta, "campos.csv")):
        dfp = pd.read_csv(os.path.join(carpeta, "campos.csv"))
        posiciones = {str(r.campo): (float(r.x_mm), float(r.y_mm)) for r in dfp.itertuples()}
    fotos = [f for f in sorted(os.listdir(carpeta))
             if f.lower().endswith(EXT) and not f.lower().startswith(IGNORAR)
             and os.path.splitext(f)[0].upper().endswith("_F")]
    if not fotos:
        sys.exit(f"{carpeta}: no hay fotos _F")
    picos = []
    for f in fotos:
        img = leer(os.path.join(carpeta, f))
        base, ext = os.path.splitext(f)
        nombre = base[:-2]
        if motores:
            mask = mascara_cuadro(img.shape)
        else:
            ruta_w = os.path.join(carpeta, nombre + "_W" + ext)
            mask, _ = detectar_campo(leer(ruta_w) if os.path.exists(ruta_w) else img, p["margen_campo"])
        if nombre in posiciones and conf.get("diametro_mm"):
            mask = mascara_disco(mask, posiciones[nombre], um_px, conf["diametro_mm"], p)
            if not mask.any():
                continue
        picos += picos_fluorescencia(img, mask, p, um_px, limite_um)
    return len(fotos), sorted(picos, reverse=True), um_px


def umbral_desde_blancos(picos_por_blanco, piso=12.0, margen=1.10):
    """Regla del umbral: se acepta en promedio hasta 1 mancha falsa por blanco. Con n blancos, el
    umbral queda sobre la mancha número n+1 más brillante de todas, multiplicada por el margen, y
    nunca bajo el piso (fluo_umbral_min de fábrica)."""
    todos = sorted((x for ps in picos_por_blanco for x in ps), reverse=True)
    n = len(picos_por_blanco)
    return max(piso, todos[n] * margen) if len(todos) > n else piso


def cmd_umbral(a):
    """Fija el umbral de fluorescencia con los blancos de procedimiento (puesta a punto C)."""
    p = cfg.cargar(a.parametros)
    por_blanco = [(c, *_picos_carpeta(c, p, a.um_por_px, a.limite_um)) for c in a.blancos]
    piso = float(cfg.cargar().get("fluo_umbral_min", 12.0))
    umbral = umbral_desde_blancos([ps for _, _, ps, _ in por_blanco], piso, a.margen)
    print(f"Umbral de fluorescencia: {umbral:.1f} (brillo sobre el fondo, escala 0–255)")
    for carpeta, n_fotos, ps, _ in por_blanco:
        quedan = sum(1 for x in ps if x >= umbral)
        print(f"  {os.path.basename(os.path.normpath(carpeta))}: {n_fotos} fotos, {len(ps)} manchas de "
              f"≥ {a.limite_um:g} µm sobre el ruido; {quedan} pasarían el umbral")
        if len(ps) > 10:
            print("    AVISO: muchas manchas en un blanco: grumos de tinte o contaminación. Revise el blanco antes de usarlo.")
    ref = None
    if a.referencia:
        n_ref, ps_ref, _ = _picos_carpeta(a.referencia, p, a.um_por_px, a.limite_um)
        pasan = sum(1 for x in ps_ref if x >= umbral)
        ref = {"carpeta": os.path.basename(os.path.normpath(a.referencia)), "manchas": len(ps_ref),
               "sobre_umbral": pasan, "fraccion": round(pasan / len(ps_ref), 3) if ps_ref else None}
        print(f"Referencia {ref['carpeta']}: {pasan} de {len(ps_ref)} manchas quedan sobre el umbral")
        if ps_ref and pasan / len(ps_ref) < 0.9:
            print("  AVISO: el umbral deja fuera más del 10 % de las partículas de referencia. Los blancos brillan "
                  "demasiado: repita los blancos (tinte recién preparado, enjuague completo) en vez de subir el umbral.")
    if a.guardar:
        datos = {}
        if os.path.exists(a.guardar):
            with open(a.guardar, encoding="utf-8") as fh:
                datos = json.load(fh)
        todos = sorted((x for _, _, ps, _ in por_blanco for x in ps), reverse=True)
        datos["fluo_umbral_min"] = round(umbral, 1)
        datos["_umbral"] = {"fecha": dt.datetime.now().isoformat(timespec="seconds"),
                            "blancos": [os.path.basename(os.path.normpath(c)) for c, _, _, _ in por_blanco],
                            "regla": "hasta 1 mancha falsa por blanco en promedio, por el margen, nunca bajo el piso",
                            "margen": a.margen, "piso": piso, "limite_um": a.limite_um,
                            "manchas_mas_brillantes": [round(x, 1) for x in todos[:10]],
                            **({"referencia": ref} if ref else {})}
        with open(a.guardar, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, indent=2, ensure_ascii=False)
        print(f"Guardado en {a.guardar}. Úselo con: scan --parametros {a.guardar}")
    return umbral


def _cfg_captura(a):
    from .captura import ConfigCaptura
    cambios = {k: getattr(a, k, None) for k in ("diametro_mm", "solape", "um_por_px", "campo_celular_mm")}
    cfg = ConfigCaptura.cargar(a.config, **cambios)
    if getattr(a, "luces_manual", False):
        cfg.luces = "manual"
    if getattr(a, "modos", None):
        cfg.modos = [m.strip().upper() for m in a.modos.split(",") if m.strip()]
    return cfg


def _mover_a_mano(platina, camara, cfg, carpeta):
    """Antes de capturar: mover la platina con el teclado hasta dejar el centro del disco al centro
    de la imagen y enfocar. Cada orden guarda vista_previa.jpg (y la muestra si hay ventana)."""
    from .captura import Recorrido
    rec = Recorrido(platina, cfg)
    platina.cero()
    x = y = 0.0
    ventana = True
    print("Mueva la platina hasta que el centro del disco quede al centro de la imagen.\n"
          "Órdenes: x 0.5 · y -1 · luz azul · luz blanca · listo")
    while True:
        img = camara.capturar()
        cv2.imwrite(os.path.join(carpeta, "vista_previa.jpg"), img)
        if ventana:
            try:
                cv2.imshow("MicroAlerta", cv2.resize(img, None, fx=0.4, fy=0.4))
                cv2.waitKey(1)
            except cv2.error:
                ventana = False
                print(f"(Sin ventana: abra {os.path.join(carpeta, 'vista_previa.jpg')} para ver la imagen)")
        orden = input("> ").strip().lower().split()
        if not orden:
            continue
        if orden[0] == "listo":
            break
        if orden[0] == "luz" and len(orden) > 1:
            platina.luces(cfg.brillo_azul, 0) if orden[1].startswith("az") else platina.luces(0, cfg.brillo_blanca)
            continue
        try:
            d = float(orden[1])
        except (IndexError, ValueError):
            print("No entendí. Ejemplos: x 0.5 · y -1 · luz azul · listo")
            continue
        if orden[0] == "x":
            x += d
        elif orden[0] == "y":
            y += d
        rec.ir(x, y)
    # el centro del disco queda como cero de la captura
    platina.cero()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Cero fijado. Anote la lectura del vernier en X e Y (movió {x:+.2f}, {y:+.2f} mm desde el inicio).")


def cmd_capturar(a):
    from . import captura as cap
    cfg = _cfg_captura(a)
    os.makedirs(a.salida, exist_ok=True)
    if a.modo == "manual":
        campos = cap.plan_manual(cfg)
        if not campos:
            sys.exit("El campo del celular es más grande que el disco: revise campo_celular_mm.")
        cap.guardar_plan(campos, os.path.join(a.salida, "plan.csv"))
        hoja = cap.hoja_manual(cfg, campos, os.path.join(a.salida, "grilla_manual.png"))
        info = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "n_campos": len(campos), "modo": "manual",
                "cobertura": round(cap.cobertura_manual(cfg, campos), 3), "configuracion": cfg.__dict__}
        with open(os.path.join(a.salida, "captura.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, ensure_ascii=False)
        print(f"Grilla manual: {len(campos)} campos que no se tocan, cubren el {info['cobertura']:.0%} del disco.\n"
              f"Imprima {hoja}. Fotografíe los campos en orden, con " +
              (" y ".join("luz azul" if m == "F" else "luz blanca" for m in cfg.modos)) +
              " en cada uno, y después use:  ordenar --fotos <carpeta del celular> --salida " + a.salida)
        return
    if a.modo == "simulado":
        from .simulacion import Escena
        escena = Escena(cfg.diametro_mm, a.semilla)
        cfg.espera_s = 0.0
        platina = cap.PlatinaSimulada(cfg, semilla=a.semilla)
        platina.cero()
        w = int(round(cfg.campo_ancho_mm * 1000 / cfg.um_por_px))
        h = int(round(cfg.campo_alto_mm * 1000 / cfg.um_por_px))
        camara = cap.CamaraSimulada(escena, platina, cfg, w, h)
    else:
        if not a.puerto:
            sys.exit("Falta --puerto (por ejemplo COM3 en Windows o /dev/ttyUSB0 en Linux).")
        platina = cap.PlatinaSerial(a.puerto)
        camara = cap.CamaraOpenCV(a.camara, cfg)
        for av in camara.avisos:
            print("AVISO:", av)
    try:
        if a.modo == "motores" and not a.sin_centrar:
            _mover_a_mano(platina, camara, cfg, a.salida)
        campos = cap.plan_cuadricula(cfg)
        print(f"Capturando {len(campos)} campos × {len(cfg.modos)} luces (solape {cfg.solape:.0%})…")
        cap.capturar(campos, platina, camara, cfg, a.salida)
    finally:
        platina.cerrar()
        camara.cerrar()
    for m in cfg.modos:
        print("Cuadrícula:", cap.armar_cuadricula(a.salida, m))
    print(f"Listo. Para contar: scan --deteccion fluorescencia --posiciones {os.path.join(a.salida, 'campos.csv')} "
          "--sin-campo-circular …")


def cmd_ordenar(a):
    from . import captura as cap
    plan = a.plan or os.path.join(a.salida, "plan.csv")
    if not os.path.exists(plan):
        sys.exit(f"No encuentro {plan}. Primero: capturar --modo manual --salida {a.salida}")
    campos = cap.leer_plan(plan)
    info = {}
    ruta_info = os.path.join(a.salida, "captura.json")
    if os.path.exists(ruta_info):
        with open(ruta_info, encoding="utf-8") as fh:
            info = json.load(fh)
    cfgd = info.get("configuracion", {})
    modos = [m.strip().upper() for m in (a.modos or ",".join(cfgd.get("modos", ["F", "W"]))).split(",")]
    try:
        pares = cap.ordenar_fotos(a.fotos, a.salida, campos, modos, a.orden)
    except ValueError as e:
        sys.exit(str(e))
    print(f"{len(pares)} fotos ordenadas en {a.salida}")
    for m in modos:
        try:
            print("Cuadrícula:", cap.armar_cuadricula(a.salida, m, um_por_px=a.um_por_px,
                                                      campo_circular_mm=cfgd.get("campo_celular_mm"),
                                                      diametro_mm=cfgd.get("diametro_mm")))
        except (FileNotFoundError, ValueError) as e:
            print("AVISO:", e)


def cmd_cuadricula(a):
    from .captura import armar_cuadricula
    print(armar_cuadricula(a.carpeta, a.modo.upper(), um_por_px=a.um_por_px))


def cmd_calibrar_platina(a):
    from . import captura as cap
    cfg = _cfg_captura(a)
    platina = cap.PlatinaSerial(a.puerto)
    camara = cap.CamaraOpenCV(a.camara, cfg)
    try:
        platina.luces(0, cfg.brillo_blanca)
        res = cap.calibrar_platina(platina, camara, cfg, a.mover_mm)
    finally:
        platina.cerrar()
        camara.cerrar()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if max(abs(res["x"]["angulo_grados"]), abs(res["y"]["angulo_grados"])) > 1:
        print("AVISO: la cámara está girada más de 1° respecto de la platina: gírela en el adaptador y repita.")
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    destino = a.guardar or a.config
    calidad = min(res["x"]["calidad_correlacion"], res["y"]["calidad_correlacion"])
    if calidad < 0.1:
        print(f"AVISO: la comparación de fotos salió débil (calidad {calidad:.2f} < 0,1). No se guardó: "
              "ponga un disco de papel usado con luz blanca, enfoque y repita.")
    elif destino and not a.no_guardar:
        cambios = cap.valores_platina(res)
        cap.guardar_en_config(destino, cambios, "platina", res)
        print(f"Guardado en {destino}: {cambios}")
    else:
        print("No se guardó: copie pasos_por_mm, signo y juego a captura.json, o use --guardar captura.json.")


def cmd_prueba(a):
    from .prueba import correr_prueba
    correr_prueba(a.salida, a.semilla, a.rapida)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="microalerta_scan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="Segmenta, mide y clasifica partículas de una muestra")
    s.add_argument("--config", help="JSON de la muestra (ver ejemplos/muestra_ejemplo.json)")
    s.add_argument("--imagenes", nargs="*", help="carpetas o patrones de imágenes (si no van en --config)")
    s.add_argument("--salida", required=True)
    s.add_argument("--um-por-px", type=float)
    s.add_argument("--diam-campo-um", type=float, help="diámetro del campo visual en µm (FN/aumento×1000)")
    s.add_argument("--flat", help="imagen de filtro limpio para corrección flat-field")
    s.add_argument("--dark", help="imagen con la luz apagada (opcional)")
    s.add_argument("--modelo", help="modelo_pcm.joblib entrenado")
    s.add_argument("--parametros", help="JSON con parámetros a sobrescribir")
    s.add_argument("--posiciones", help="CSV campo,x_mm,y_mm con la posición de la platina de cada campo")
    s.add_argument("--incluir-duplicados", action="store_true", help="no omitir fotos repetidas")
    s.add_argument("--tolerancia-duplicado", type=int, default=6, help="bits distintos (de 256) para considerar duplicado")
    s.add_argument("--sin-campo-circular", action="store_true", help="imágenes del escáner (sin ocular)")
    s.add_argument("--deteccion", choices=["oscuras", "fluorescencia"],
                   help="oscuras: partículas oscuras en luz blanca (_W). fluorescencia: partículas que brillan "
                        "con Rojo de Nilo en la foto _F (por defecto, lo que diga la ficha; si no, oscuras)")
    s.set_defaults(f=cmd_scan)

    t = sub.add_parser("tdc", help="Calcula la TDC y la calidad del dato a partir de resultados de scan")
    t.add_argument("--muestras", nargs="+", required=True)
    t.add_argument("--controles", nargs="+", required=True)
    t.add_argument("--salida")
    t.set_defaults(f=cmd_tdc)

    c = sub.add_parser("calibrar", help="µm/px desde una foto del portaobjetos micrométrico")
    c.add_argument("--imagen", required=True)
    c.add_argument("--um-entre-marcas", type=float, required=True, help="típico: 10 µm")
    c.add_argument("--guardar", help="captura.json donde escribir um_por_px")
    c.set_defaults(f=cmd_calibrar)

    u = sub.add_parser("umbral", help="Fija el umbral de fluorescencia con los blancos de procedimiento")
    u.add_argument("--blancos", nargs="+", required=True, help="una carpeta de fotos (_F, y _W si hay) por blanco")
    u.add_argument("--um-por-px", dest="um_por_px", type=float)
    u.add_argument("--limite-um", dest="limite_um", type=float, default=20.0)
    u.add_argument("--margen", type=float, default=1.10, help="el umbral queda este factor sobre la mancha de corte")
    u.add_argument("--referencia", help="carpeta de fotos de la muestra de prueba (7B): revisa que el umbral no la pierda")
    u.add_argument("--parametros", help="JSON con parámetros a sobrescribir")
    u.add_argument("--guardar", help="JSON de parámetros de la campaña (se usa con scan --parametros)")
    u.set_defaults(f=cmd_umbral)

    k = sub.add_parser("capturar", help="Recorre el disco con los motores y saca las fotos (o arma la grilla manual)")
    k.add_argument("--modo", choices=["motores", "manual", "simulado"], required=True)
    k.add_argument("--config", help="captura.json (ver ejemplos/captura_ejemplo.json)")
    k.add_argument("--salida", required=True)
    k.add_argument("--puerto", help="puerto del Arduino (COM3, /dev/ttyUSB0…)")
    k.add_argument("--camara", type=int, default=0, help="número de la cámara USB")
    k.add_argument("--modos", help="luces a fotografiar en cada campo: F,W (azul y blanca) o F")
    k.add_argument("--luces-manual", action="store_true", help="las luces no están en el Arduino: dos pasadas")
    k.add_argument("--diametro-mm", dest="diametro_mm", type=float)
    k.add_argument("--solape", type=float)
    k.add_argument("--um-por-px", dest="um_por_px", type=float)
    k.add_argument("--campo-celular-mm", dest="campo_celular_mm", type=float)
    k.add_argument("--sin-centrar", action="store_true", help="no pedir centrar el disco antes de empezar")
    k.add_argument("--semilla", type=int, default=0)
    k.set_defaults(f=cmd_capturar)

    o = sub.add_parser("ordenar", help="Nombra las fotos del celular según la grilla manual y arma la cuadrícula")
    o.add_argument("--fotos", required=True, help="carpeta con las fotos del celular, sin otras fotos")
    o.add_argument("--salida", required=True, help="carpeta donde se creó la grilla (plan.csv)")
    o.add_argument("--plan")
    o.add_argument("--modos", help="F,W (por defecto, los de la grilla)")
    o.add_argument("--orden", choices=["por-campo", "por-pasada"], default="por-campo",
                   help="por-campo: azul y blanca en cada campo; por-pasada: todas las azules y después las blancas")
    o.add_argument("--um-por-px", dest="um_por_px", type=float)
    o.set_defaults(f=cmd_ordenar)

    q = sub.add_parser("cuadricula", help="Arma el mosaico del disco con el número de cada campo")
    q.add_argument("--carpeta", required=True)
    q.add_argument("--modo", default="F")
    q.add_argument("--um-por-px", dest="um_por_px", type=float)
    q.set_defaults(f=cmd_cuadricula)

    cp = sub.add_parser("calibrar-platina", help="Mide pasos por mm, signo, ángulo y juego de cada eje con la cámara")
    cp.add_argument("--puerto", required=True)
    cp.add_argument("--camara", type=int, default=0)
    cp.add_argument("--config")
    cp.add_argument("--mover-mm", dest="mover_mm", type=float, default=0.8)
    cp.add_argument("--salida", help="JSON con el detalle de la medición")
    cp.add_argument("--guardar", help="captura.json donde escribir pasos por mm, signos y juego (por defecto, --config)")
    cp.add_argument("--no-guardar", action="store_true")
    cp.set_defaults(f=cmd_calibrar_platina, diametro_mm=None, solape=None, um_por_px=None, campo_celular_mm=None)

    pr = sub.add_parser("prueba", help="Prueba sin equipo: disco simulado, captura, cuadrícula, conteo y comparación")
    pr.add_argument("--salida", required=True)
    pr.add_argument("--semilla", type=int, default=0)
    pr.add_argument("--rapida", action="store_true", help="disco chico, para probar en segundos")
    pr.set_defaults(f=cmd_prueba)

    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    main()
