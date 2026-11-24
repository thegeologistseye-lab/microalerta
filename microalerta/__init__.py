"""MicroAlerta: servidor Flask que sirve la PWA y su API."""
import os
import secrets
import sqlite3
from datetime import timedelta

from flask import Flask, send_from_directory

from .config import Config, PROJECT_ROOT
from .db import init_db


def asegurar_admin(app):
    """Crea la cuenta de administración la primera vez.

    La contraseña sale de MICROALERTA_ADMIN_PASSWORD; si no existe, se genera una
    aleatoria y se muestra una sola vez en los registros (Logs de Render).
    """
    from .api import crear_usuario
    with sqlite3.connect(app.config["DATABASE"]) as con:
        con.row_factory = sqlite3.Row
        if con.execute("SELECT 1 FROM usuarios WHERE rol='admin'").fetchone():
            return
        clave = os.environ.get("MICROALERTA_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
        crear_usuario(os.environ.get("MICROALERTA_ADMIN_USER", "admin"), "Equipo MicroAlerta", "admin", clave, db=con)
        if not os.environ.get("MICROALERTA_ADMIN_PASSWORD"):
            print("=" * 60, flush=True)
            print(f"  Cuenta admin creada. Usuario: admin · Contraseña: {clave}", flush=True)
            print("  Guárdala ahora: no se volverá a mostrar.", flush=True)
            print("=" * 60, flush=True)


def create_app(test_config=None):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(Config)
    app.permanent_session_lifetime = timedelta(days=14)
    if test_config:
        app.config.update(test_config)
    init_db(app)
    asegurar_admin(app)

    from .api import bp
    app.register_blueprint(bp)
    static_dir = os.path.join(PROJECT_ROOT, "static")

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/<path:nombre>")
    def estaticos(nombre):
        return send_from_directory(static_dir, nombre)

    @app.after_request
    def cabeceras(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return resp

    return app
