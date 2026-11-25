"""API REST de MicroAlerta: sesión, reportes, validación, galería y exportación."""
import io
import json
import os
import secrets
import time
from datetime import datetime
from functools import wraps

from flask import Blueprint, current_app, jsonify, request, send_file, send_from_directory, session
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

from . import excel, science, weather
from .config import AREA_PLACA_CM2, FOTO_TIPOS, HORAS_EXPOSICION, VOLUMEN_ML
from .db import get_db

bp = Blueprint("api", __name__, url_prefix="/api")

# Coordenadas aproximadas de los puntos (las mismas que muestra la app)
PUNTOS = {
    "Laboratorio": (-27.38767, -70.31007), "Entrada": (-27.38822, -70.31022),
    "Comedor": (-27.38782, -70.30965), "Casino": (-27.38809, -70.30937),
    "Baño": (-27.38755, -70.30942), "Calle": (-27.38849, -70.30982),
    "Copiapó centro": (-27.366, -70.332), "Tierra Amarilla": (-27.48, -70.27),
    "Caldera": (-27.067, -70.818), "Bahía Inglesa": (-27.105, -70.855),
    "Chañaral": (-26.348, -70.62), "Vallenar": (-28.576, -70.76), "Huasco": (-28.466, -71.22),
}
TIPOS = {"Muestra de campo", "Réplica", "Blanco (control cerrado)"}
GRADOS = {"observado", "compatible (óptica)", "confirmado (Raman/FTIR)"}
CATEGORIAS = {"placas", "lugol", "azul", "sin", "ref", "muestreo", "equipo", "fibra", "fragmento", "film", "organica", "raman"}
SECTORES = {"entrada", "comedor", "casino", "calle", "bano", "laboratorio", "laba", "labc", "placas", "ref", "muestreo", "equipo"}
_FALLOS = {}


def err(msg, code=400):
    return jsonify({"error": msg}), code


def usuario_actual():
    uid = session.get("uid")
    if not uid:
        return None
    return get_db().execute("SELECT id, usuario, nombre, rol FROM usuarios WHERE id=? AND activo=1", (uid,)).fetchone()


def requiere(*roles):
    def deco(f):
        @wraps(f)
        def w(*a, **k):
            u = usuario_actual()
            if not u:
                return err("Inicia sesión para continuar.", 401)
            if roles and u["rol"] not in roles:
                return err("No tienes permiso para esta acción.", 403)
            return f(u, *a, **k)
        return w
    return deco


def guardar_foto(archivo, carpeta: str) -> str:
    """Valida y re-codifica la foto (quita metadatos como la ubicación GPS)."""
    if archivo.mimetype not in FOTO_TIPOS:
        raise ValueError("Usa una foto JPG, PNG o WEBP.")
    try:
        img = Image.open(archivo.stream)
        img.verify()
        archivo.stream.seek(0)
        img = Image.open(archivo.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("El archivo no es una imagen válida.")
    img.thumbnail((2400, 2400))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    nombre = secrets.token_hex(12) + ".jpg"
    destino = os.path.join(current_app.config["UPLOAD_DIR"], carpeta)
    os.makedirs(destino, exist_ok=True)
    img.save(os.path.join(destino, nombre), "JPEG", quality=85)
    return nombre


def num(v, defecto=None, minimo=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        if defecto is None:
            raise ValueError("Número inválido")
        return defecto
    if x < minimo:
        raise ValueError("Los valores no pueden ser negativos")
    return x


def a_json(r, u):
    puede_foto = r["foto"] and u and (u["rol"] == "admin" or u["id"] == r["autor_id"])
    ms = int(datetime.fromisoformat(r["creado"]).timestamp() * 1000)
    return {
        "id": str(r["id"]), "punto": r["punto"], "tipo": r["tipo"], "grado": r["grado"],
        "conc": r["conc"], "tdc": r["tdc"], "n": r["particulas"], "ctrl": r["control"],
        "nivel": r["nivel"], "estado": r["estado"], "com": r["comentario"], "obs": r["obs"],
        "t": ms, "foto": f"/api/reportes/{r['id']}/foto" if puede_foto else None,
        "avisos": science.advertencias(r["particulas"], r["control"]) if u and u["rol"] == "admin" else [],
    }


# ---------- Sesión ----------
@bp.get("/health")
def health():
    return jsonify({"app": "microalerta", "ok": True})


@bp.get("/me")
def me():
    u = usuario_actual()
    return jsonify({"user": dict(u) if u else None})


@bp.post("/login")
def login():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    ahora = time.time()
    fallos = [t for t in _FALLOS.get(ip, []) if ahora - t < 300]
    if len(fallos) >= 5:
        return err("Demasiados intentos. Espera unos minutos.", 429)
    d = request.get_json(silent=True) or {}
    u = get_db().execute("SELECT * FROM usuarios WHERE usuario=? AND activo=1",
                         (str(d.get("usuario", "")).strip().lower(),)).fetchone()
    if not u or not check_password_hash(u["clave_hash"], str(d.get("clave", ""))):
        _FALLOS[ip] = fallos + [ahora]
        return err("Usuario o contraseña incorrectos.", 401)
    _FALLOS.pop(ip, None)
    session.clear()
    session["uid"] = u["id"]
    session.permanent = True
    return jsonify({"user": {"id": u["id"], "usuario": u["usuario"], "nombre": u["nombre"], "rol": u["rol"]}})


@bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


# ---------- Reportes ----------
@bp.get("/reports")
def listar_reportes():
    u = usuario_actual()
    db = get_db()
    if u and u["rol"] == "admin":
        rows = db.execute("SELECT * FROM reportes ORDER BY creado DESC LIMIT 500").fetchall()
    elif u:
        rows = db.execute("SELECT * FROM reportes WHERE estado='aprobado' OR autor_id=? ORDER BY creado DESC LIMIT 500",
                          (u["id"],)).fetchall()
    else:
        rows = db.execute("SELECT * FROM reportes WHERE estado='aprobado' ORDER BY creado DESC LIMIT 500").fetchall()
    return jsonify({"reports": [a_json(r, u) for r in rows]})


@bp.post("/reports")
@requiere("admin", "colegio", "ciudadano")
def crear_reporte(u):
    f = request.form
    try:
        punto = f.get("punto", "").strip()
        if punto not in PUNTOS:
            raise ValueError("Elige un punto de muestreo de la lista.")
        tipo = f.get("tipo", "Muestra de campo")
        if tipo not in TIPOS:
            raise ValueError("Tipo de plato inválido.")
        n = int(num(f.get("n")))
        ctrl = int(num(f.get("ctrl"), 0))
        vol = num(f.get("volumen_ml"), VOLUMEN_ML, 0.1)
        area = num(f.get("area_cm2"), AREA_PLACA_CM2, 0.1)
        horas = num(f.get("horas"), HORAS_EXPOSICION, 0.1)
        # Parte del disco que se contó (modo celular: unos 40 %). Se extiende al disco completo,
        # igual que MicroAlerta Scan, y se deja anotado lo contado de verdad.
        pct = num(f.get("fraccion_pct"), 100.0, 1.0)
        if pct > 100:
            raise ValueError("La parte del disco analizada no puede pasar de 100 %.")
        nota = ""
        if pct < 100:
            nota = f"Contadas {n} (control {ctrl}) en {pct:g} % del disco; extendido al disco completo. "
            n = round(science.extrapolar(n, pct / 100))
            ctrl = round(science.extrapolar(ctrl, pct / 100))
        conc = science.concentracion(n, vol)
        t = science.tdc(n, ctrl, area, horas)
        foto = guardar_foto(request.files["foto"], "reportes") if request.files.get("foto") else None
    except ValueError as e:
        return err(str(e))
    grado = f.get("grado", "")
    grado = grado if grado in GRADOS else "compatible (óptica)"
    clima = weather.clima_actual(*PUNTOS[punto]) if current_app.config["FETCH_WEATHER"] else None
    db = get_db()
    cur = db.execute(
        """INSERT INTO reportes (autor_id, punto, placa, tipo, grado, particulas, control, volumen_ml,
           area_cm2, horas, conc, tdc, nivel, obs, foto, clima) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (u["id"], punto, f.get("placa", "")[:20], tipo, grado, n, ctrl, vol, area, horas, conc, t,
         science.semaforo(conc), (nota + f.get("obs", ""))[:300], foto, weather.como_texto(clima)))
    db.commit()
    r = db.execute("SELECT * FROM reportes WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify({"report": a_json(r, u)}), 201


@bp.post("/quality")
def calidad():
    """Criterio de confianza reforzada (misma fórmula que el protocolo y MicroAlerta Scan).
    JSON: {"muestras": [{"n": 12, "fraccion": 1}, …], "controles": [{"n": 0, "fraccion": 1}, …]}"""
    d = request.get_json(silent=True) or {}
    try:
        lista = lambda k: [(float(x["n"]), float(x.get("fraccion", 1))) for x in (d.get(k) or [])][:50]
        m, c = lista("muestras"), lista("controles")
        if any(n < 0 for n, _ in m + c):
            raise ValueError("Los conteos no pueden ser negativos.")
        return jsonify(science.evaluar_calidad(m, c))
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return err(str(e) if isinstance(e, ValueError) else "Datos inválidos.")


@bp.post("/reports/<int:rid>/review")
@requiere("admin")
def revisar(u, rid):
    d = request.get_json(silent=True) or {}
    estado = d.get("estado")
    com = str(d.get("com", "")).strip()[:200]
    if estado not in ("aprobado", "rechazado"):
        return err("Estado inválido.")
    if estado == "rechazado" and not com:
        return err("Escribe el motivo del rechazo.")
    db = get_db()
    cur = db.execute("UPDATE reportes SET estado=?, comentario=?, revisor_id=?, revisado=datetime('now') WHERE id=?",
                     (estado, com, u["id"], rid))
    db.commit()
    if not cur.rowcount:
        return err("Reporte no encontrado.", 404)
    return jsonify({"ok": True})


@bp.get("/reportes/<int:rid>/foto")
@requiere("admin", "colegio", "ciudadano")
def foto_reporte(u, rid):
    r = get_db().execute("SELECT foto, autor_id FROM reportes WHERE id=?", (rid,)).fetchone()
    if not r or not r["foto"] or (u["rol"] != "admin" and u["id"] != r["autor_id"]):
        return err("Foto no disponible.", 404)
    return send_from_directory(os.path.join(current_app.config["UPLOAD_DIR"], "reportes"), r["foto"])


# ---------- Galería ----------
@bp.get("/gallery")
def galeria():
    rows = get_db().execute("SELECT * FROM galeria ORDER BY creado DESC LIMIT 300").fetchall()
    fotos = [{"id": r["id"], "cat": r["categoria"], "sector": r["sector"], "txt": r["texto"], "url": f"/api/galeria/{r['archivo']}"} for r in rows]
    # Fotos del piloto incluidas con la app (static/galeria/index.json)
    indice = os.path.join(current_app.root_path, "..", "static", "galeria", "index.json")
    if os.path.exists(indice):
        with open(indice, encoding="utf-8") as fh:
            fotos += [{"id": f"p{i}", "cat": x["cat"], "sector": x.get("sector", ""), "who": x.get("who", ""),
                       "kind": x.get("kind", "image"), "txt": x["txt"], "url": f"/galeria/{x['file']}"}
                      for i, x in enumerate(json.load(fh))]
    return jsonify({"photos": fotos})


@bp.post("/gallery")
@requiere("admin")
def subir_galeria(u):
    cat = request.form.get("cat", "")
    if cat not in CATEGORIAS:
        return err("Categoría inválida.")
    if not request.files.get("foto"):
        return err("Falta la foto.")
    try:
        nombre = guardar_foto(request.files["foto"], "galeria")
    except ValueError as e:
        return err(str(e))
    db = get_db()
    sector = request.form.get("sector", "")
    sector = sector if sector in SECTORES else ""
    db.execute("INSERT INTO galeria (archivo, categoria, sector, texto, autor_id) VALUES (?,?,?,?,?)",
               (nombre, cat, sector, request.form.get("txt", "")[:120], u["id"]))
    db.commit()
    return jsonify({"ok": True}), 201


@bp.get("/galeria/<path:nombre>")
def archivo_galeria(nombre):
    return send_from_directory(os.path.join(current_app.config["UPLOAD_DIR"], "galeria"), nombre)


# ---------- Administración ----------
@bp.get("/admin/usuarios")
@requiere("admin")
def listar_usuarios(u):
    rows = get_db().execute("SELECT id, usuario, nombre, rol, activo, creado FROM usuarios ORDER BY id").fetchall()
    return jsonify({"usuarios": [dict(r) for r in rows]})


@bp.post("/admin/usuarios")
@requiere("admin")
def crear_usuario_api(u):
    d = request.get_json(silent=True) or {}
    try:
        crear_usuario(d.get("usuario", ""), d.get("nombre", ""), d.get("rol", ""), d.get("clave", ""))
    except ValueError as e:
        return err(str(e))
    return jsonify({"ok": True}), 201


@bp.get("/export.xlsx")
@requiere("admin")
def exportar(u):
    rows = get_db().execute(
        "SELECT r.*, COALESCE(us.nombre,'') AS autor FROM reportes r LEFT JOIN usuarios us ON us.id=r.autor_id ORDER BY r.creado").fetchall()
    return send_file(io.BytesIO(excel.libro(rows)), as_attachment=True,
                     download_name=f"MicroAlerta_{datetime.now():%Y%m%d}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def crear_usuario(usuario, nombre, rol, clave, db=None):
    usuario = str(usuario).strip().lower()
    if not usuario or not str(nombre).strip():
        raise ValueError("Usuario y nombre son obligatorios.")
    if rol not in ("admin", "colegio", "ciudadano"):
        raise ValueError("Rol inválido (admin, colegio o ciudadano).")
    if len(str(clave)) < 10:
        raise ValueError("La contraseña debe tener al menos 10 caracteres.")
    db = db or get_db()
    try:
        db.execute("INSERT INTO usuarios (usuario, nombre, rol, clave_hash) VALUES (?,?,?,?)",
                   (usuario, str(nombre).strip(), rol, generate_password_hash(clave)))
        db.commit()
    except Exception as e:  # noqa: BLE001
        if "UNIQUE" in str(e):
            raise ValueError("Ese usuario ya existe.")
        raise


# ---------- Colegios de la red ----------
COMUNAS = {"Copiapó", "Caldera", "Tierra Amarilla", "Chañaral", "Diego de Almagro", "Vallenar", "Huasco",
           "Freirina", "Alto del Carmen", "Otra región"}


@bp.get("/schools")
def listar_colegios():
    u = usuario_actual()
    q = "SELECT id, nombre, comuna, estado FROM colegios"
    if not (u and u["rol"] == "admin"):
        q += " WHERE estado='aprobado'"
    rows = get_db().execute(q + " ORDER BY creado DESC LIMIT 300").fetchall()
    return jsonify({"schools": [dict(r) for r in rows]})


@bp.post("/schools")
@requiere("admin", "colegio", "ciudadano")
def crear_colegio(u):
    d = request.get_json(silent=True) or {}
    nombre = str(d.get("nombre", "")).strip()[:80]
    comuna = str(d.get("comuna", "")).strip()
    if len(nombre) < 3:
        return err("Escribe el nombre del colegio.")
    if comuna not in COMUNAS:
        return err("Elige una comuna de la lista.")
    db = get_db()
    cur = db.execute("INSERT INTO colegios (nombre, comuna, autor_id) VALUES (?,?,?)", (nombre, comuna, u["id"]))
    db.commit()
    return jsonify({"id": cur.lastrowid}), 201


@bp.post("/schools/<int:sid>/review")
@requiere("admin")
def revisar_colegio(u, sid):
    estado = (request.get_json(silent=True) or {}).get("estado")
    if estado not in ("aprobado", "rechazado"):
        return err("Estado inválido.")
    db = get_db()
    cur = db.execute("UPDATE colegios SET estado=? WHERE id=?", (estado, sid))
    db.commit()
    if not cur.rowcount:
        return err("Colegio no encontrado.", 404)
    return jsonify({"ok": True})


# ---------- Datos reales de campañas ----------
@bp.get("/campaigns")
def leer_campanas():
    r = get_db().execute("SELECT valor FROM ajustes WHERE clave='campanas'").fetchone()
    return jsonify({"data": json.loads(r["valor"]) if r else None})


@bp.post("/campaigns")
@requiere("admin")
def guardar_campanas(u):
    d = request.get_json(silent=True) or {}
    camps = d.get("camps") or []
    sec = d.get("sec") or {}
    db = get_db()
    if not camps:
        db.execute("DELETE FROM ajustes WHERE clave='campanas'")
        db.commit()
        return jsonify({"data": None})
    try:
        if not isinstance(camps, list) or len(camps) > 40 or not isinstance(sec, dict):
            raise ValueError
        camps = [str(c)[:12] for c in camps]
        limpio = {}
        for k, v in sec.items():
            if k not in PUNTOS or not isinstance(v, list) or len(v) != len(camps):
                raise ValueError
            limpio[k] = [max(0.0, float(x)) for x in v]
        # Opcional: desviación estándar y número de réplicas por sector y campaña
        extra = {}
        for campo, conv in (("sd", float), ("n", int)):
            m = d.get(campo)
            if m is None:
                continue
            if not isinstance(m, dict):
                raise ValueError
            extra[campo] = {}
            for k, v in m.items():
                if k not in limpio or not isinstance(v, list) or len(v) != len(camps):
                    raise ValueError
                extra[campo][k] = [None if x is None else max(0, conv(x)) for x in v]
    except (ValueError, TypeError):
        return err("Datos de campaña inválidos.")
    valor = {"camps": camps, "sec": limpio, **extra, "t": int(time.time() * 1000)}
    db.execute("INSERT INTO ajustes (clave, valor) VALUES ('campanas', ?) "
               "ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor, actualizado=datetime('now')",
               (json.dumps(valor, ensure_ascii=False),))
    db.commit()
    return jsonify({"data": valor})
