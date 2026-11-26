"""Pruebas de la captura en cuadrícula, la detección por fluorescencia y la prueba simulada."""
import json
import math
import os
import time

import cv2
import numpy as np
import pytest

from microalerta_scan import config as cfgmod
from microalerta_scan.captura import (ConfigCaptura, PlatinaSimulada, Recorrido, plan_cuadricula, plan_manual,
                                      cobertura_manual, ordenar_fotos, guardar_plan, armar_cuadricula)
from microalerta_scan.cli import agrupar_imagenes
from microalerta_scan.fluorescencia import segmentar_fluorescencia
from microalerta_scan.rasgos import medir
from microalerta_scan.prueba import correr_prueba


def test_cuadricula_cubre_todo_el_disco():
    cfg = ConfigCaptura()
    campos = plan_cuadricula(cfg)
    rng = np.random.default_rng(0)
    r = cfg.diametro_mm / 2
    for _ in range(3000):
        a, rr = rng.uniform(0, 2 * np.pi), r * math.sqrt(rng.uniform())
        x, y = rr * math.cos(a), rr * math.sin(a)
        assert any(abs(x - c["x_mm"]) <= cfg.campo_ancho_mm / 2 and abs(y - c["y_mm"]) <= cfg.campo_alto_mm / 2
                   for c in campos)
    # cada fila se recorre de izquierda a derecha
    for f in {c["fila"] for c in campos}:
        xs = [c["x_mm"] for c in campos if c["fila"] == f]
        assert xs == sorted(xs)


def test_grilla_manual_campos_no_se_tocan():
    cfg = ConfigCaptura(campo_celular_mm=4.5)
    campos = plan_manual(cfg)
    assert len(campos) >= 5
    for i, a in enumerate(campos):
        assert math.hypot(a["x_mm"], a["y_mm"]) + cfg.campo_celular_mm / 2 <= cfg.diametro_mm / 2 + 1e-9
        for b in campos[i + 1:]:
            assert math.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"]) > cfg.campo_celular_mm
    assert 0 < cobertura_manual(cfg, campos) < 1


@pytest.mark.parametrize("sx,sy", [(1, 1), (-1, -1)])
def test_llega_siempre_desde_el_mismo_lado_aunque_haya_juego(sx, sy):
    """Con 0,2 mm de juego, si se llega siempre con el motor avanzando (como al fijar el cero) el
    error es nulo y no depende del orden ni del signo de los ejes."""
    cfg = ConfigCaptura(juego_mm=0.4, signo_x=sx, signo_y=sy)
    platina = PlatinaSimulada(cfg, juego_real_mm=0.2, error_mm=0.0)
    platina.cero()
    rec = Recorrido(platina, cfg)
    rng = np.random.default_rng(1)
    puntos = [(float(x), float(y)) for x, y in rng.uniform(-5, 5, (40, 2))]
    errores = []
    for x, y in puntos:
        rec.ir(x, y)
        rx, ry = platina.posicion_real_mm()
        errores.append((rx - x, ry - y))
    errores = np.array(errores)
    assert np.abs(errores).max() < 0.02


def test_ordenar_fotos_del_celular(tmp_path):
    cfg = ConfigCaptura()
    campos = plan_manual(cfg)[:3]
    cel = tmp_path / "celular"
    cel.mkdir()
    t = time.time() - 1000
    for k in range(6):                                  # F, W por campo
        ruta = cel / f"IMG_{900 - k}.jpg"               # los nombres no siguen el orden de toma
        cv2.imwrite(str(ruta), np.full((20, 20, 3), k * 30, np.uint8))
        os.utime(ruta, (t + k, t + k))
    pares = ordenar_fotos(str(cel), str(tmp_path / "salida"), campos, ("F", "W"), "por-campo")
    assert [(c["campo"], m) for _, c, m in pares] == [(campos[0]["campo"], "F"), (campos[0]["campo"], "W"),
                                                      (campos[1]["campo"], "F"), (campos[1]["campo"], "W"),
                                                      (campos[2]["campo"], "F"), (campos[2]["campo"], "W")]
    img = cv2.imread(str(tmp_path / "salida" / f"{campos[1]['campo']}_W.jpg"))
    assert int(img.mean()) == 90
    with pytest.raises(ValueError):
        ordenar_fotos(str(cel), str(tmp_path / "otra"), campos, ("F",), "por-campo")


def test_scan_ignora_la_cuadricula(tmp_path):
    for n in ("campo001_F.jpg", "campo001_W.jpg", "cuadricula_F.jpg", "vista_previa.jpg"):
        cv2.imwrite(str(tmp_path / n), np.zeros((10, 10, 3), np.uint8))
    grupos = agrupar_imagenes([str(tmp_path)], {})
    assert [os.path.basename(k) for k in grupos] == ["campo001"]


def _imagen_fluo(um_px=1.1):
    h, w = 600, 800
    rng = np.random.default_rng(3)
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (5, 14, 24)
    img += rng.normal(0, 1.5, img.shape)
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (200, 300), int(round(50 / um_px)), 255, -1)                  # fragmento de 100 µm
    cv2.line(m, (450, 200), (450 + int(400 / um_px), 200), 255, int(round(15 / um_px)))  # fibra de ~400 µm
    cv2.circle(m, (650, 450), int(round(5 / um_px)), 255, -1)                   # punto de 10 µm
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0)[:, :, None]
    img = img * (1 - a) + a * np.array([8, 160, 200], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_fluorescencia_mide_bien_y_respeta_el_limite():
    um_px = 1.1
    img = _imagen_fluo(um_px)
    p = cfgmod.cargar()
    p["largo_min_fibra_um"] = 20.0
    mask = np.full(img.shape[:2], 255, np.uint8)
    mask[:2, :] = mask[-2:, :] = 0
    mask[:, :2] = mask[:, -2:] = 0
    et, fibra, razon, u, mask = segmentar_fluorescencia(img, mask, p, um_px)
    filas = medir(et, img, razon, fibra, mask, p, um_px, None, img)
    largos = sorted(f["largo_um"] for f in filas)
    grandes = [l for l in largos if l >= 20]
    assert len(grandes) == 2
    assert abs(grandes[0] - 100) < 12               # fragmento: ±12 %
    assert abs(grandes[1] - 415) < 40               # fibra (400 µm + extremos redondeados)
    assert all(l < 20 for l in largos[:-2])         # el punto de 10 µm no llega a 20 µm


def test_prueba_simulada_cuenta_cada_particula_una_vez(tmp_path):
    inf = correr_prueba(str(tmp_path / "prueba"), semilla=0, rapida=True, avisar=lambda *_: None)
    assert inf["perdidas"] == 0
    assert inf["contadas_de_mas"] <= 1
    assert os.path.exists(tmp_path / "prueba" / "cuadricula_revisada.jpg")
    assert os.path.exists(tmp_path / "prueba" / "fotos" / "cuadricula_F.jpg")
    with open(tmp_path / "prueba" / "resultado" / "resultado.json", encoding="utf-8") as fh:
        res = json.load(fh)
    assert res["conteo"]["modo_captura"] == "mosaico_con_posiciones"
    assert res["conteo"]["deteccion"] == "fluorescencia"
    assert res["conteo"]["limite_minimo_um"] >= 20


# ------------------------------------------------------------ calibraciones que se guardan solas
class _CamaraTextura:
    """Cámara de prueba sobre una textura fija (como el papel de un disco usado con luz blanca)."""

    def __init__(self, platina, um_px, w=640, h=480, semilla=0):
        rng = np.random.default_rng(semilla)
        t = cv2.GaussianBlur(rng.normal(0, 1, (3000, 3000)).astype(np.float32), (0, 0), 3)
        self.tex = ((t / t.std()) * 25 + 128).clip(0, 255).astype(np.uint8)
        self.pl, self.um, self.w, self.h, self.rng = platina, um_px, w, h, rng

    def capturar(self):
        x, y = self.pl.posicion_real_mm()
        cx, cy = 1500 + x * 1000 / self.um, 1500 + y * 1000 / self.um
        M = np.float32([[1, 0, self.w / 2 - cx], [0, 1, self.h / 2 - cy]])
        g = cv2.warpAffine(self.tex, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
        g = np.clip(g + self.rng.normal(0, 3, g.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("sx,sy", [(1, 1), (-1, 1), (1, -1)])
def test_calibrar_platina_mide_pasos_sentido_y_juego(sx, sy, tmp_path):
    from microalerta_scan.captura import calibrar_platina, valores_platina, guardar_en_config
    real = ConfigCaptura(signo_x=sx, signo_y=sy, um_por_px=2.0, espera_s=0, pasos_mm_x=70.0, pasos_mm_y=160.0)
    supuesto = ConfigCaptura(um_por_px=2.0, espera_s=0)            # pasos de fábrica y signos +1
    pl = PlatinaSimulada(real, juego_real_mm=0.05, error_mm=0.001)
    res = calibrar_platina(pl, _CamaraTextura(pl, 2.0), supuesto, mover_mm=0.8, avisar=lambda *_: None)
    assert res["x"]["signo"] == sx and res["y"]["signo"] == sy
    assert abs(res["x"]["pasos_por_mm"] - 70) < 1.5 and abs(res["y"]["pasos_por_mm"] - 160) < 3
    assert abs(res["x"]["juego_mm"] - 0.05) < 0.01
    assert abs(res["x"]["angulo_grados"]) < 1 and abs(res["y"]["angulo_grados"]) < 1
    # se guarda en captura.json sin copiar nada a mano, y captura.json se sigue leyendo bien
    ruta = str(tmp_path / "captura.json")
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump({"diametro_mm": 21.0, "pasos_mm_x": 76.8}, fh)
    guardar_en_config(ruta, valores_platina(res), "platina", res)
    c = ConfigCaptura.cargar(ruta)
    assert c.signo_x == sx and abs(c.pasos_mm_x - 70) < 1.5 and c.diametro_mm == 21.0
    assert 0.1 <= c.juego_mm < 0.2                                 # sobrepaso = 2 × juego, mínimo 0,1 mm
    # con signo y pasos calibrados, el recorrido llega donde se le pide
    pl2 = PlatinaSimulada(real, juego_real_mm=0.05, error_mm=0.0)
    pl2.cero()
    rec = Recorrido(pl2, c)
    rec.ir(1.0, -0.5)
    x, y = pl2.posicion_real_mm()
    assert abs(x - 1.0) < 0.03 and abs(y + 0.5) < 0.03


def test_calibrar_guarda_escala_y_campo(tmp_path):
    from microalerta_scan.cli import main
    img = np.full((480, 640, 3), 230, np.uint8)
    for x in range(5, 640, 9):                                     # marcas cada 9 px = 10 µm
        cv2.line(img, (x, 100), (x, 380), (40, 40, 40), 2)
    cv2.imwrite(str(tmp_path / "micrometro.png"), img)
    ruta = str(tmp_path / "captura.json")
    main(["calibrar", "--imagen", str(tmp_path / "micrometro.png"), "--um-entre-marcas", "10", "--guardar", ruta])
    c = ConfigCaptura.cargar(ruta)
    assert abs(c.um_por_px - 10 / 9) < 0.02
    assert abs(c.campo_ancho_mm - 640 * c.um_por_px / 1000) < 1e-3
    with open(ruta, encoding="utf-8") as fh:
        assert json.load(fh)["calibraciones"][-1]["tipo"] == "micrometro"


def test_umbral_desde_blancos_y_scan_lo_usa(tmp_path):
    """Blancos con grumos de tinte grandes y débiles: el umbral queda sobre ellos, los blancos dan
    a lo más 1 partícula en promedio y la muestra de referencia no se pierde."""
    from microalerta_scan.cli import main, umbral_desde_blancos
    from microalerta_scan.prueba import capturar_simulado
    from microalerta_scan.simulacion import Escena
    assert umbral_desde_blancos([[30, 20], [25], [18]], piso=12, margen=1.1) == pytest.approx(19.8)
    assert umbral_desde_blancos([[5], [], []], piso=12) == 12
    d, w, h, um = 3.0, 648, 486, 2.2
    blancos = []
    for k in range(3):
        carpeta = str(tmp_path / f"B{k + 1}")
        esc = Escena(d, 20 + k, n_fragmentos=0, n_fibras=0, n_pequenas=0, n_grumos=4, n_arena=10, n_algodon=1,
                     grumo_radio_um=(12, 20), grumo_brillo=0.2)
        capturar_simulado(esc, carpeta, d, w, h, um, 20 + k)
        blancos.append(carpeta)
    ref = str(tmp_path / "REF")
    capturar_simulado(Escena(d, 7, n_fragmentos=8, n_fibras=4, n_pequenas=2, n_grumos=0, n_arena=10, n_algodon=1),
                      ref, d, w, h, um, 7)
    param = str(tmp_path / "parametros_campana.json")
    umbral = main(["umbral", "--blancos", *blancos, "--referencia", ref, "--guardar", param])
    with open(param, encoding="utf-8") as fh:
        guardado = json.load(fh)
    assert guardado["fluo_umbral_min"] == pytest.approx(umbral, abs=0.1) and umbral > 12
    assert guardado["_umbral"]["referencia"]["fraccion"] >= 0.9
    # scan con esos parámetros: la escala sale sola de captura.json (no se pasa --um-por-px)
    total = 0
    for k, carpeta in enumerate(blancos):
        m = {"muestra_id": f"B{k + 1}", "tipo": "blanco_procedimiento", "tiempo_dias": 1.0,
             "area_captador_m2": 0.00636, "filtro_material": "papel", "filtro_diametro_mm": d,
             "filtro_completo": False, "deteccion": "fluorescencia", "limite_minimo_um": 20, "imagenes": [carpeta]}
        extra = ["--parametros", param]
        if k == 0:      # por la ficha, con un umbral viejo copiado a mano que no debe ganar
            m.update({"parametros_campana": "parametros_campana.json", "fluo_umbral_min": 12,
                      "calibracion": {"um_por_px": 1.1}})
            extra = []
        ruta_m = str(tmp_path / f"m{k}.json")
        with open(ruta_m, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
        salida = str(tmp_path / f"r{k}")
        main(["scan", "--config", ruta_m, "--salida", salida, "--posiciones", os.path.join(carpeta, "campos.csv"),
              "--sin-campo-circular", *extra])
        with open(os.path.join(salida, "resultado.json"), encoding="utf-8") as fh:
            res = json.load(fh)
        assert res["procesamiento"]["campos"][0]["um_por_px"] == pytest.approx(um)
        assert res["conteo"]["umbral_fluorescencia"] == pytest.approx(umbral, abs=0.1)
        assert res["conteo"]["origen_umbral"] == "parametros_campana.json"
        if k == 0:
            assert any("fluo_umbral_min" in av for av in res["avisos"])
            assert any("Escala" in av for av in res["avisos"])
        total += res["conteo"]["pcm"]["n_contado"]
    assert total <= 3
