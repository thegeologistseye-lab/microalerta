"""
MicroAlerta — servidor local.

Uso:
    python app.py                       # http://localhost:8000
    python app.py --port 8080
    python app.py --crear-usuario       # agrega una cuenta (colegio, ciudadano o admin)
"""
import argparse
import getpass
import os
import sqlite3

from microalerta import create_app


def crear_usuario_cli(app):
    from microalerta.api import crear_usuario
    usuario = input("Usuario: ")
    nombre = input("Nombre visible (ej. Liceo de Caldera): ")
    rol = input("Rol (admin / colegio / ciudadano): ").strip()
    clave = getpass.getpass("Contraseña (mín. 10 caracteres): ")
    with sqlite3.connect(app.config["DATABASE"]) as con:
        crear_usuario(usuario, nombre, rol, clave, db=con)
    print("Cuenta creada.")


def main():
    p = argparse.ArgumentParser(description="MicroAlerta (PWA + Flask)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--crear-usuario", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("MICROALERTA_HTTPS", "0")  # en local no hay HTTPS
    app = create_app()
    if args.crear_usuario:
        crear_usuario_cli(app)
        return
    print(f"  MicroAlerta en http://localhost:{args.port}")
    print(f"  Desde tu celular (misma Wi-Fi): http://<IP-de-esta-PC>:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
