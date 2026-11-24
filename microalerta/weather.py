"""Clima de Open-Meteo para adjuntar a cada reporte (best-effort, con caché corta)."""
import json
import time

import requests

_CACHE = {}
_TTL = 600  # segundos


def clima_actual(lat: float, lon: float) -> dict | None:
    key = (round(lat, 2), round(lon, 2))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "wind_speed_unit": "kmh",
                "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,"
                           "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            },
            timeout=6,
        )
        r.raise_for_status()
        c = r.json().get("current") or {}
        data = {
            "fuente": "Open-Meteo", "hora": c.get("time"),
            "temperatura": c.get("temperature_2m"), "humedad": c.get("relative_humidity_2m"),
            "viento": c.get("wind_speed_10m"), "rafaga": c.get("wind_gusts_10m"),
            "direccion": c.get("wind_direction_10m"), "presion": c.get("surface_pressure"),
            "precipitacion": c.get("precipitation"),
        }
        _CACHE[key] = (time.time(), data)
        return data
    except (requests.RequestException, ValueError):
        return None


def como_texto(data: dict | None) -> str | None:
    return json.dumps(data, ensure_ascii=False) if data else None
