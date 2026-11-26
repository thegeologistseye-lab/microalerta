"""Pruebas con imágenes sintéticas de dimensiones conocidas.  Ejecutar:  python -m pytest -q"""
import math
import numpy as np
import cv2
import pytest

from microalerta_scan import config
from microalerta_scan.vision import segmentar, detectar_campo, alinear
from microalerta_scan.rasgos import medir
from microalerta_scan.calibracion import um_por_px_desde_micrometro, corregir_flat
from microalerta_scan.muestra import tdc, fraccion_analizada, evaluar_campana, ic95_poisson

UM_PX = 2.0   # escala sintética: 2 µm por píxel


def papel(h=900, w=900, semilla=0):
    """Fondo tipo papel: textura suave + viñeteo, color de tinción (anaranjado)."""
    rng = np.random.default_rng(semilla)
    ruido = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    ruido = ruido / ruido.std() * 0.04
    yy, xx = np.mgrid[0:h, 0:w]
    vin = 1 - 0.25 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2)
    base = np.clip(vin * (1 + ruido), 0, 1.2)
    img = np.stack([base * 90, base * 170, base * 235], -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def escena():
    img = papel()
    oscuro = (25, 30, 40)
    # fibra recta: 300 µm de largo; cv2.line con grosor 3 dibuja 5 px reales = 10 µm de ancho
    cv2.line(img, (150, 200), (300, 200), oscuro, 3)
    # fragmento circular: diámetro 100 µm -> radio 25 px
    cv2.circle(img, (600, 250), 25, oscuro, -1)
    # fibra curva (arco de radio 80 px, 180°): largo ≈ π·80 px = 251 px = 503 µm
    cv2.ellipse(img, (450, 600), (80, 80), 0, 180, 360, oscuro, 3)
    return img


@pytest.fixture(scope="module")
def resultado():
    p = config.cargar()
    p["detectar_campo_circular"] = False
    img = escena()
    mask = np.full(img.shape[:2], 255, np.uint8)
    et, fib, raz, u, mask = segmentar(img, mask, p, UM_PX)
    filas = medir(et, img, raz, fib, mask, p, UM_PX)
    return filas


def _cerca(filas, x, y, tol=40):
    c = [f for f in filas if abs(f["cx_px"] - x) < tol and abs(f["cy_px"] - y) < tol]
    assert c, f"no se detectó partícula cerca de ({x},{y})"
    return c[0]


def test_numero_de_particulas(resultado):
    assert len(resultado) == 3, [(round(f["cx_px"]), round(f["cy_px"])) for f in resultado]


def test_fibra_recta(resultado):
    f = _cerca(resultado, 225, 200)
    assert f["forma"] == "fibra"
    assert f["largo_um"] == pytest.approx(300, rel=0.08)
    assert f["ancho_um"] == pytest.approx(10, abs=3)


def test_fragmento_circular(resultado):
    f = _cerca(resultado, 600, 250)
    assert f["forma"] == "fragmento"
    assert f["diam_equiv_um"] == pytest.approx(100, rel=0.08)
    assert f["circularidad"] > 0.8


def test_fibra_curva_mide_largo_real(resultado):
    f = _cerca(resultado, 450, 560, tol=60)
    assert f["forma"] == "fibra"
    assert f["largo_um"] == pytest.approx(math.pi * 80 * UM_PX, rel=0.10)


def test_control_limpio_sin_falsos_positivos():
    p = config.cargar()
    p["detectar_campo_circular"] = False
    img = papel(semilla=3)
    mask = np.full(img.shape[:2], 255, np.uint8)
    et, fib, raz, u, mask = segmentar(img, mask, p, UM_PX)
    assert len(medir(et, img, raz, fib, mask, p, UM_PX)) == 0


def test_detecta_campo_circular_cortado():
    img = np.zeros((1200, 900, 3), np.uint8)
    cv2.circle(img, (450, 600), 520, (60, 160, 230), -1)      # círculo más ancho que la foto
    mask, (cx, cy, r) = detectar_campo(img, 0.0)
    assert r == pytest.approx(520, rel=0.03)
    assert cx == pytest.approx(450, abs=10)


def test_calibracion_micrometro():
    img = np.full((800, 800), 220, np.uint8)
    for x in range(0, 800, 25):                                # una marca cada 25 px = 10 µm
        cv2.line(img, (x, 200), (x, 600), 40, 2)
    um_px, info = um_por_px_desde_micrometro(img, 10.0)
    assert um_px == pytest.approx(0.4, rel=0.03)


def test_flat_field_elimina_vineteo():
    flat = papel(semilla=1)
    img = flat.copy()
    out = corregir_flat(img, flat)
    g = out[:, :, 1].astype(float)
    assert g.std() < 2.0


def test_alineacion_multimodal():
    img = escena()
    M = np.float32([[1, 0, 7], [0, 1, -5]])
    mov = cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    alin, W, cc = alinear(img, mov)
    assert W[0, 2] == pytest.approx(7, abs=1.0) and W[1, 2] == pytest.approx(-5, abs=1.0)


def test_tdc_ejemplo_del_ensayo():
    # Nm = 63, Nc = 0, A = 0,01 m², t = 1 día -> 6300 PCM/(m²·día)
    assert tdc(63, 0, 0.01, 1) == pytest.approx(6300)
    assert tdc(5, 9, 0.01, 1) == 0.0                     # nunca negativa


def test_fraccion_y_extrapolacion():
    f, aviso = fraccion_analizada(area_campos_mm2=173.0, filtro_diametro_mm=47 * 0.5 ** 0.5 * 2 ** 0.5)
    assert 0 < f <= 1
    f, aviso = fraccion_analizada(None, None)
    assert f is None and "diámetro" in aviso


def test_criterio_calidad():
    def r(n, base="pcm_modelo"):
        return {"conteo": {"pcm": {"n_estimado_filtro": n}, "base_conteo": base},
                "muestra": {"muestra_id": "x", "area_captador_m2": 0.00636, "tiempo_dias": 1}}
    res = evaluar_campana([r(60), r(70), r(65)], [r(2), r(3), r(1)])
    assert res["calidad"] == "confianza reforzada"
    # junto a la categoría se informan la DE y el CV de las réplicas (no es un intervalo de confianza)
    import numpy as np
    tdc = np.array([f["tdc"] for f in res["tdc_por_replica"]])
    assert abs(res["tdc_de"] - tdc.std(ddof=1)) < 1e-9
    assert abs(res["cv_porcentaje"] - 100 * tdc.std(ddof=1) / tdc.mean()) < 1e-9
    res = evaluar_campana([r(60)], [r(0)])
    assert res["calidad"] == "exploratorio"


def test_criterio_controles_en_cero_no_regala_confianza():
    """Controles todos en 0 dan DE = 0: sin piso de Poisson, 1 partícula bastaría."""
    def r(n, contado=None, frac=1.0):
        return {"conteo": {"pcm": {"n_estimado_filtro": n, "n_contado": n if contado is None else contado},
                           "base_conteo": "pcm_modelo", "fraccion_filtro_analizada": frac},
                "muestra": {"muestra_id": "x", "area_captador_m2": 0.00636, "tiempo_dias": 1}}
    ceros = [r(0), r(0), r(0)]
    assert evaluar_campana([r(1), r(2), r(1)], ceros)["calidad"] == "exploratorio"
    assert evaluar_campana([r(10), r(12), r(9)], ceros)["calidad"] == "confianza reforzada"
    # con 10 % del filtro analizado, el piso escala ×10: 12 estimadas (≈1 contada) no alcanzan
    ceros10 = [r(0, 0, 0.1)] * 3
    assert evaluar_campana([r(10, 1, 0.1), r(20, 2, 0.1), r(10, 1, 0.1)], ceros10)["calidad"] == "exploratorio"


def test_ic95_poisson():
    lo, hi = ic95_poisson(0)
    assert lo == 0 and hi == pytest.approx(3.689, rel=0.01)


def test_limite_minimo_y_area_escaneada():
    from microalerta_scan.cli import limite_minimo_um
    p = {"area_min_um2": 150.0}
    assert limite_minimo_um(p, None) is None
    assert abs(limite_minimo_um(p, 1.0) - 13.82) < 0.05          # dominado por el área mínima
    assert abs(limite_minimo_um(p, 10.0) - 30.0) < 1e-6          # dominado por 3 píxeles


def test_rasgos_polarizacion():
    import numpy as np, cv2
    from microalerta_scan.rasgos import RASGOS_MODELO
    for c in ("birrefringencia_p95", "frac_brillante_pol"):
        assert c in RASGOS_MODELO


def test_mosaico_solapado_cuenta_una_vez(tmp_path):
    """Dos campos de 1,6 mm que se solapan 0,4 mm: la partícula de la franja común se cuenta una vez
    y el área es la de la unión, no la suma."""
    import json, cv2, numpy as np
    from microalerta_scan.cli import main
    um = 2.0
    H, W = 600, 800                                  # 1,2 × 1,6 mm
    rng = np.random.default_rng(1)
    filtro = (np.full((H, 1400, 3), 215) + rng.normal(0, 3, (H, 1400, 3))).clip(0, 255).astype(np.uint8)
    cv2.circle(filtro, (300, 300), 14, (60, 60, 60), -1)     # solo en el campo A
    cv2.circle(filtro, (700, 250), 14, (60, 60, 60), -1)     # en la franja común (x 600–800)
    cv2.circle(filtro, (1100, 350), 14, (60, 60, 60), -1)    # solo en el campo B
    d = tmp_path / "fotos"; d.mkdir()
    cv2.imwrite(str(d / "A_W.png"), filtro[:, 0:800])
    cv2.imwrite(str(d / "B_W.png"), filtro[:, 600:1400])
    (tmp_path / "pos.csv").write_text("campo,x_mm,y_mm\nA,0.8,0.6\nB,2.0,0.6\n")
    (tmp_path / "m.json").write_text(json.dumps({"muestra_id": "MOS", "filtro_diametro_mm": 35,
        "calibracion": {"um_por_px": um}, "imagenes": ["fotos/"],
        "multimodal": {"sufijo_blanca": "_W", "sufijo_pol": "_P", "sufijo_fluo": "_F"}}))
    out = tmp_path / "out"
    main(["scan", "--config", str(tmp_path / "m.json"), "--salida", str(out),
          "--posiciones", str(tmp_path / "pos.csv"), "--sin-campo-circular"])
    r = json.loads((out / "resultado.json").read_text())
    assert r["conteo"]["n_detectadas"] == 4                   # la del medio aparece en ambos campos
    assert r["conteo"]["pcm"]["n_contado"] == 3
    union = 1400 * 600 * (um / 1000) ** 2                     # 2,8 × 1,2 mm = 3,36 mm²
    assert abs(r["conteo"]["area_escaneada_mm2"] - union) / union < 0.03


def test_mosaico_fibra_cortada_se_cuenta_en_el_vecino():
    """Cada partícula se cuenta una vez, en el campo más cercano entre los que la ven entera."""
    import numpy as np
    from microalerta_scan.mosaico import Campo, resolver_solapes
    um = 2.0                                     # 1 px = 0,002 mm
    mA = np.full((600, 800), 255, np.uint8)
    mA[:, 700:] = 0                              # zona útil de A: x < 1,4 mm (p. ej. viñeteo)
    A = Campo("A", 0.8, 0.6, um, mA)
    B = Campo("B", 2.0, 0.6, um, np.full((600, 800), 255, np.uint8))   # B ve x 1,2–2,8 mm
    # 1) partícula chica en x≈1,3 mm: ambos la ven entera, A está más cerca → solo A
    pa = {"campo": "A", "cx_px": 650, "cy_px": 300, "bbox_px": [640, 290, 660, 310], "toca_borde_campo": False}
    pb = {"campo": "B", "cx_px": 50, "cy_px": 300, "bbox_px": [40, 290, 60, 310], "toca_borde_campo": False}
    # 2) fibra de x 1,25 a 1,45 mm: A la ve cortada (toca su borde útil), B entera → solo B
    fa = {"campo": "A", "cx_px": 662, "cy_px": 300, "bbox_px": [625, 298, 700, 302], "toca_borde_campo": True}
    fb = {"campo": "B", "cx_px": 75, "cy_px": 300, "bbox_px": [25, 298, 125, 302], "toca_borde_campo": False}
    resolver_solapes([A, B], [pa, pb, fa, fb])
    assert pa["fuera_de_su_zona"] is False and pb["fuera_de_su_zona"] is True
    assert fb["fuera_de_su_zona"] is False       # contada en B (fa no cuenta por tocar el borde)


def test_criterio_borde_exacto_de_la_formula():
    """Controles en 0 con el disco completo: DE = 1, hay que pasar de 3 en promedio (no basta 3)."""
    def r(n, frac=1.0):
        return {"conteo": {"pcm": {"n_estimado_filtro": n / frac, "n_contado": n}, "base_conteo": "pcm_modelo",
                           "fraccion_filtro_analizada": frac},
                "muestra": {"muestra_id": "x", "area_captador_m2": 0.00636, "tiempo_dias": 1}}
    ceros = [r(0), r(0), r(0)]
    assert evaluar_campana([r(3), r(3), r(3)], ceros)["calidad"] == "exploratorio"
    res = evaluar_campana([r(3), r(4), r(3)], ceros)
    assert res["calidad"] == "confianza reforzada"
    assert res["criterio_calidad"]["de_usada"] == 1.0 and res["criterio_calidad"]["formula"]
    # controles con dispersión real: manda s_c (n − 1) y no el piso
    res = evaluar_campana([r(30), r(30), r(30)], [r(2), r(6), r(10)])
    assert res["criterio_calidad"]["de_usada"] == 4.0                     # s_c de 2, 6, 10 = 4
    assert res["criterio_calidad"]["umbral_media_control_mas_3de"] == 18.0
    # con el celular (40 % del disco) el piso es 1 / 0,4 = 2,5 en el disco: siguen siendo 3 contadas
    ceros4 = [r(0, 0.4)] * 3
    assert evaluar_campana([r(3, 0.4)] * 3, ceros4)["calidad"] == "exploratorio"
    assert evaluar_campana([r(4, 0.4)] * 3, ceros4)["calidad"] == "confianza reforzada"


def test_tdc_no_trunca_cada_replica_y_marca_limite_de_deteccion():
    """Cerca del fondo, truncar cada réplica a 0 antes de promediar sube la media (revisión v6, punto D)."""
    def r(n):
        return {"conteo": {"pcm": {"n_estimado_filtro": n, "n_contado": n}, "base_conteo": "pcm_modelo",
                           "fraccion_filtro_analizada": 1.0},
                "muestra": {"muestra_id": "x", "area_captador_m2": 0.01, "tiempo_dias": 1}}
    res = evaluar_campana([r(0), r(1), r(5)], [r(1), r(2), r(3)])
    difs = [f["diferencia_corregida"] for f in res["tdc_por_replica"]]
    assert difs == [-200.0, -100.0, 300.0]            # se guardan sin truncar
    assert res["tdc_media"] == 0.0                     # media de (−2, −1, 3) = 0; truncando antes daría 100
    assert res["bajo_limite_deteccion"] and res["calidad"] == "exploratorio"
    assert abs(res["limite_deteccion_tdc"] - 3 * res["criterio_calidad"]["de_usada"] / 0.01) < 1e-9
    alto = evaluar_campana([r(60), r(70), r(65)], [r(1), r(2), r(3)])
    assert not alto["bajo_limite_deteccion"] and alto["calidad"] == "confianza reforzada"


def test_clasificador_valida_por_disco_y_prueba_detecta_discos_repetidos(tmp_path):
    import pandas as pd
    from microalerta_scan import clasificador
    from microalerta_scan.rasgos import RASGOS_MODELO
    rng = np.random.default_rng(0)
    def tabla(discos, n=30):
        filas = []
        for d in discos:
            for i in range(n):
                pcm = i % 2 == 0
                fila = {c: rng.normal(1.0 if pcm else 0.0, 0.3) for c in RASGOS_MODELO}
                fila.update(muestra_id=d, etiqueta="PCM" if pcm else "mineral")
                filas.append(fila)
        return pd.DataFrame(filas)
    ent, pru = tmp_path / "ent.csv", tmp_path / "pru.csv"
    tabla(["D1", "D2", "D3"]).to_csv(ent, index=False)
    tabla(["D4", "D5"], 24).to_csv(pru, index=False)
    inf = clasificador.entrenar(ent, tmp_path / "m.joblib")
    assert inf["validacion_cruzada"].startswith("por disco")
    ok = clasificador.probar(tmp_path / "m.joblib", pru, 0.8, csv_entrenamiento=ent)
    assert ok["aprueba"] and ok["informar_como"] == "PCM"
    tabla(["D3", "D6"], 24).to_csv(pru, index=False)   # D3 también se usó para entrenar
    mal = clasificador.probar(tmp_path / "m.joblib", pru, 0.8, csv_entrenamiento=ent)
    assert not mal["aprueba"] and "D3" in mal["avisos"][-1]
