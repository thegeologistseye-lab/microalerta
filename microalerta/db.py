"""Base de datos SQLite de MicroAlerta."""
import os
import sqlite3
from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  usuario TEXT UNIQUE NOT NULL,
  nombre TEXT NOT NULL,
  rol TEXT NOT NULL CHECK (rol IN ('admin','colegio','ciudadano')),
  clave_hash TEXT NOT NULL,
  activo INTEGER NOT NULL DEFAULT 1,
  creado TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS reportes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  autor_id INTEGER REFERENCES usuarios(id),
  punto TEXT NOT NULL,
  placa TEXT DEFAULT '',
  tipo TEXT NOT NULL,
  grado TEXT DEFAULT '',
  particulas INTEGER NOT NULL,
  control INTEGER NOT NULL DEFAULT 0,
  volumen_ml REAL NOT NULL,
  area_cm2 REAL NOT NULL,
  horas REAL NOT NULL,
  conc REAL NOT NULL,
  tdc REAL NOT NULL,
  nivel TEXT NOT NULL,
  obs TEXT DEFAULT '',
  foto TEXT,
  clima TEXT,
  estado TEXT NOT NULL DEFAULT 'pendiente' CHECK (estado IN ('pendiente','aprobado','rechazado')),
  comentario TEXT DEFAULT '',
  revisor_id INTEGER REFERENCES usuarios(id),
  demo INTEGER NOT NULL DEFAULT 0,
  creado TEXT NOT NULL DEFAULT (datetime('now')),
  revisado TEXT
);
CREATE TABLE IF NOT EXISTS galeria (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  archivo TEXT NOT NULL,
  categoria TEXT NOT NULL,
  sector TEXT DEFAULT '',
  texto TEXT DEFAULT '',
  autor_id INTEGER REFERENCES usuarios(id),
  creado TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS colegios (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nombre TEXT NOT NULL,
  comuna TEXT NOT NULL,
  estado TEXT NOT NULL DEFAULT 'pendiente' CHECK (estado IN ('pendiente','aprobado','rechazado')),
  autor_id INTEGER REFERENCES usuarios(id),
  creado TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ajustes (
  clave TEXT PRIMARY KEY,
  valor TEXT NOT NULL,
  actualizado TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    with sqlite3.connect(app.config["DATABASE"]) as con:
        con.executescript(SCHEMA)
        cols = [r[1] for r in con.execute("PRAGMA table_info(galeria)")]
        if "sector" not in cols:  # bases creadas con una versión anterior
            con.execute("ALTER TABLE galeria ADD COLUMN sector TEXT DEFAULT ''")
    app.teardown_appcontext(close_db)
