"""Configuración de MicroAlerta. Todo se puede cambiar con variables de entorno."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MICROALERTA_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Protocolo de captación
VOLUMEN_ML = 40.0            # agua destilada congelada por placa
AREA_PLACA_CM2 = 63.6        # placa Petri de 90 mm de diámetro
HORAS_EXPOSICION = 24.0

# Semáforo de deposición (part./mL en el agua de la placa). Exploratorio, no sanitario.
UMBRALES = [(0.5, "Verde"), (2.0, "Amarillo"), (5.0, "Naranja"), (float("inf"), "Rojo")]

# Niveles chilenos de material particulado (µg/m³, promedio 24 h): límite superior exclusivo.
# MP2,5: D.S. 12/2011. MP10: D.S. 12/2021 (tramos Bueno/Regular de los planes GEC del MMA).
NIVELES_MP = [
    ("Bueno", 50, 130),
    ("Regular", 80, 180),
    ("Alerta", 110, 230),
    ("Preemergencia", 170, 330),
    ("Emergencia", float("inf"), float("inf")),
]

COPIAPO = (-27.37, -70.33)
FOTO_MAX_MB = 8
FOTO_TIPOS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class Config:
    SECRET_KEY = os.environ.get("MICROALERTA_SECRET") or os.urandom(32).hex()
    DATABASE = os.environ.get("MICROALERTA_DB", os.path.join(DATA_DIR, "microalerta.db"))
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    MAX_CONTENT_LENGTH = FOTO_MAX_MB * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("MICROALERTA_HTTPS", "1") == "1"
    FETCH_WEATHER = os.environ.get("MICROALERTA_FETCH_WEATHER", "1") == "1"
