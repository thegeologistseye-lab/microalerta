import io

import pytest
from PIL import Image

from microalerta import create_app, science


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MICROALERTA_ADMIN_PASSWORD", "clave-admin-123")
    return create_app({"DATABASE": str(tmp_path / "t.db"), "UPLOAD_DIR": str(tmp_path / "up"),
                       "TESTING": True, "FETCH_WEATHER": False, "SESSION_COOKIE_SECURE": False})


@pytest.fixture
def client(app):
    return app.test_client()


def login(c, u="admin", p="clave-admin-123"):
    return c.post("/api/login", json={"usuario": u, "clave": p})


def foto():
    b = io.BytesIO()
    Image.new("RGB", (20, 20), "teal").save(b, "PNG")
    b.seek(0)
    return (b, "placa.png", "image/png")


# ---------- Ciencia ----------
def test_concentracion_y_semaforo():
    assert science.concentracion(58, 40) == pytest.approx(1.45)
    assert science.semaforo(0.2) == "Verde"
    assert science.semaforo(1.45) == "Amarillo"
    assert science.semaforo(2.0) == "Naranja"
    assert science.semaforo(5.0) == "Rojo"


def test_tdc_descuenta_control_y_no_es_negativa():
    assert science.tdc(63, 0, 100, 24) == pytest.approx(6300)
    assert science.tdc(3, 5, 63.6, 24) == 0


def test_nivel_mp_usa_el_peor():
    assert science.nivel_mp(14, 62)["peor"] == "Bueno"
    assert science.nivel_mp(90, 62)["peor"] == "Alerta"
    assert science.nivel_mp(20, 340)["peor"] == "Emergencia"


# ---------- API ----------
def test_health_y_login(client):
    assert client.get("/api/health").json["app"] == "microalerta"
    assert login(client, p="mala").status_code == 401
    assert login(client).json["user"]["rol"] == "admin"


def test_reporte_requiere_sesion(client):
    r = client.post("/api/reports", data={"punto": "Baño", "n": "10"})
    assert r.status_code == 401


def test_flujo_reporte_y_validacion(client, app):
    login(client)
    client.post("/api/admin/usuarios", json={"usuario": "colegio1", "nombre": "Liceo", "rol": "colegio",
                                             "clave": "clave-colegio-1"})
    c2 = app.test_client()
    assert login(c2, "colegio1", "clave-colegio-1").status_code == 200
    r = c2.post("/api/reports", data={"punto": "Baño", "tipo": "Muestra de campo", "n": "58", "ctrl": "0",
                                      "foto": foto()}, content_type="multipart/form-data")
    assert r.status_code == 201
    rep = r.json["report"]
    assert rep["estado"] == "pendiente" and rep["conc"] == pytest.approx(1.45) and rep["foto"]
    # anónimo no ve pendientes
    assert app.test_client().get("/api/reports").json["reports"] == []
    # colegio no puede aprobar
    assert c2.post(f"/api/reports/{rep['id']}/review", json={"estado": "aprobado"}).status_code == 403
    # rechazo exige motivo
    assert client.post(f"/api/reports/{rep['id']}/review", json={"estado": "rechazado"}).status_code == 400
    assert client.post(f"/api/reports/{rep['id']}/review", json={"estado": "aprobado"}).status_code == 200
    pub = app.test_client().get("/api/reports").json["reports"]
    assert len(pub) == 1 and pub[0]["foto"] is None  # la foto del reporte no es pública


def test_punto_invalido_y_negativos(client):
    login(client)
    assert client.post("/api/reports", data={"punto": "Marte", "n": "3"}).status_code == 400
    assert client.post("/api/reports", data={"punto": "Baño", "n": "-3"}).status_code == 400


def test_galeria_y_excel(client):
    login(client)
    r = client.post("/api/gallery", data={"cat": "fibra", "txt": "Fibra azul", "foto": foto()},
                    content_type="multipart/form-data")
    assert r.status_code == 201
    g = client.get("/api/gallery").json["photos"]
    assert g[0]["cat"] == "fibra" and client.get(g[0]["url"]).status_code == 200
    piloto = [x for x in g if x["cat"] in ("lugol", "azul", "placas")]
    assert len(piloto) >= 17 and client.get(piloto[0]["url"]).status_code == 200
    x = client.get("/api/export.xlsx")
    assert x.status_code == 200 and x.data[:2] == b"PK"


def test_archivo_no_imagen_rechazado(client):
    login(client)
    r = client.post("/api/gallery", data={"cat": "fibra", "foto": (io.BytesIO(b"hola"), "x.png", "image/png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_sirve_la_app(client):
    r = client.get("/")
    assert r.status_code == 200 and b"MicroAlerta" in r.data


# ---------- Colegios y campañas ----------
def test_colegios_flujo(client):
    assert client.post("/api/schools", json={"nombre": "Liceo X", "comuna": "Caldera"}).status_code == 401
    login(client)
    assert client.post("/api/schools", json={"nombre": "Li", "comuna": "Caldera"}).status_code == 400
    assert client.post("/api/schools", json={"nombre": "Liceo X", "comuna": "Marte"}).status_code == 400
    sid = client.post("/api/schools", json={"nombre": "Liceo X", "comuna": "Caldera"}).get_json()["id"]
    assert client.get("/api/schools").get_json()["schools"][0]["estado"] == "pendiente"
    assert client.post(f"/api/schools/{sid}/review", json={"estado": "aprobado"}).status_code == 200
    client.post("/api/logout")
    assert [s["nombre"] for s in client.get("/api/schools").get_json()["schools"]] == ["Liceo X"]


def test_campanas(client):
    assert client.get("/api/campaigns").get_json()["data"] is None
    d = {"camps": ["Mar 27", "Abr 27"], "sec": {"Comedor": [1.2, 0.8]}}
    assert client.post("/api/campaigns", json=d).status_code == 401
    login(client)
    assert client.post("/api/campaigns", json={"camps": ["A"], "sec": {"Comedor": [1, 2]}}).status_code == 400
    assert client.post("/api/campaigns", json=d).status_code == 200
    assert client.get("/api/campaigns").get_json()["data"]["sec"]["Comedor"] == [1.2, 0.8]
    client.post("/api/campaigns", json={"camps": []})
    assert client.get("/api/campaigns").get_json()["data"] is None


def test_campanas_con_replicas(client):
    login(client)
    d = {"camps": ["C1", "C2"], "sec": {"Baño": [5.5, 5.7]},
         "sd": {"Baño": [None, 0.5]}, "n": {"Baño": [1, 3]}}
    assert client.post("/api/campaigns", json=d).status_code == 200
    data = client.get("/api/campaigns").get_json()["data"]
    assert data["sd"]["Baño"] == [None, 0.5] and data["n"]["Baño"] == [1, 3]
    # sd con largo distinto o para un sector sin promedio: se rechaza
    assert client.post("/api/campaigns", json={**d, "sd": {"Baño": [0.5]}}).status_code == 400
    assert client.post("/api/campaigns", json={**d, "n": {"Calle": [1, 3]}}).status_code == 400


def test_criterio_de_calidad_igual_al_protocolo():
    """Misma fórmula que MicroAlerta Scan y la sección 5 del protocolo, con sus casos de borde."""
    ceros = [(0, 1)] * 3
    assert science.evaluar_calidad([(3, 1)] * 3, ceros)["calidad"] == "exploratorio"      # 3 justas no alcanzan
    r = science.evaluar_calidad([(3, 1), (4, 1), (3, 1)], ceros)
    assert r["calidad"] == "confianza reforzada" and r["de_usada"] == 1.0
    r = science.evaluar_calidad([(30, 1)] * 3, [(2, 1), (6, 1), (10, 1)])
    assert r["de_usada"] == 4.0 and r["umbral"] == 18.0                                  # manda s_c (n − 1)
    # celular, 40 % del disco: el piso es 2,5 en el disco completo; siguen siendo más de 3 contadas
    assert science.evaluar_calidad([(3, 0.4)] * 3, [(0, 0.4)] * 3)["calidad"] == "exploratorio"
    assert science.evaluar_calidad([(4, 0.4)] * 3, [(0, 0.4)] * 3)["calidad"] == "confianza reforzada"
    # menos de 3 réplicas o controles: siempre exploratorio
    assert science.evaluar_calidad([(50, 1)] * 2, ceros)["calidad"] == "exploratorio"
    assert science.evaluar_calidad([(50, 1)] * 3, [(0, 1)])["calidad"] == "exploratorio"


def test_api_calidad(client):
    r = client.post("/api/quality", json={"muestras": [{"n": 4, "fraccion": 0.4}] * 3,
                                          "controles": [{"n": 0, "fraccion": 0.4}] * 3})
    assert r.status_code == 200 and r.json["calidad"] == "confianza reforzada" and r.json["formula"]
    assert client.post("/api/quality", json={"muestras": [], "controles": []}).status_code == 400
    assert client.post("/api/quality", json={"muestras": [{"n": -1}], "controles": [{"n": 0}]}).status_code == 400


def test_reporte_con_parte_del_disco(client):
    login(client)
    r = client.post("/api/reports", data={"punto": "Baño", "tipo": "Muestra de campo", "n": "8", "ctrl": "1",
                                          "fraccion_pct": "40"}, content_type="multipart/form-data")
    assert r.status_code == 201
    rep = r.json["report"]
    assert rep["n"] == 20 and rep["ctrl"] == 2                    # 8 ÷ 0,4 y 1 ÷ 0,4 redondeado
    assert "40 % del disco" in rep["obs"]
    assert client.post("/api/reports", data={"punto": "Baño", "n": "8", "fraccion_pct": "140"},
                       content_type="multipart/form-data").status_code == 400


@pytest.mark.parametrize("n, f", [(float("inf"), 1), (float("nan"), 1), (1e400, 1), (5, float("nan")), (5, 0), (5, 1.5)])
def test_calidad_rechaza_valores_no_finitos(n, f):
    """Un conteo infinito ganaba «confianza reforzada» (revisión del PR #1)."""
    with pytest.raises(ValueError):
        science.evaluar_calidad([(n, f)] * 3, [(0, 1)] * 3)
    with pytest.raises(ValueError):
        science.evaluar_calidad([(5, 1)] * 3, [(n, f)] * 3)


def test_api_calidad_rechaza_infinito_y_listas_largas(client):
    # JSON con 1e400 se lee como infinito en Python
    r = client.post("/api/quality", data='{"muestras":[{"n":1e400},{"n":1e400},{"n":1e400}],'
                    '"controles":[{"n":0},{"n":0},{"n":0}]}', content_type="application/json")
    assert r.status_code == 400
    r = client.post("/api/quality", json={"muestras": [{"n": 5}] * 51, "controles": [{"n": 0}] * 3})
    assert r.status_code == 400 and "50" in r.json["error"]


def test_reporte_rechaza_conteo_infinito(client):
    login(client)
    r = client.post("/api/reports", data={"punto": "Baño", "n": "inf"}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = client.post("/api/reports", data={"punto": "Baño", "n": "5", "fraccion_pct": "nan"},
                    content_type="multipart/form-data")
    assert r.status_code == 400
