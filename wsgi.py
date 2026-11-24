"""Punto de entrada para producción (Render): waitress-serve wsgi:app"""
from microalerta import create_app

app = create_app()
