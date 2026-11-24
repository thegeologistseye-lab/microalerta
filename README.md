# MicroAlerta · servidor

App web instalable (PWA) y API para MicroAlerta: monitoreo ciudadano de partículas
compatibles con microplásticos depositadas desde el aire en Atacama.
Colegio Cervantino, Copiapó.

## Qué incluye

- **La app** (`static/index.html`): la misma que se ve en Claude. Al abrirse desde este
  servidor detecta la API y activa cuentas, reportes reales, validación y galería.
  El clima y la calidad del aire (MP2,5 y MP10) llegan en vivo desde Open-Meteo.
- **Cuentas con roles**: `admin` valida reportes y sube fotos a la galería;
  `colegio` y `ciudadano` envían reportes.
- **Reportes con trazabilidad**: cada uno guarda punto, placa, tipo, conteo, control,
  volumen, área, horas, part./mL, TDC, nivel, foto, observaciones, el clima de
  Open-Meteo en ese momento y quién lo aprobó o rechazó.
- **Validación**: nada aparece en el mapa hasta que un admin lo aprueba; rechazar
  exige un motivo.
- **Galería del microscopio** por categoría (fibras, fragmentos, films, orgánicas, Raman).
- **Excel** (`/api/export.xlsx`, solo admin): reportes, resumen por punto y notas.
- **Colegios de la red** (`/api/schools`): cualquier cuenta registra un colegio; aparece en el mapa cuando un admin lo aprueba.
- **Datos reales de campañas** (`/api/campaigns`, escribir solo admin): reemplazan los datos de ejemplo de los sectores del colegio.
- **Seguridad**: contraseñas con hash, límite de intentos de inicio de sesión, fotos
  re-codificadas (se borran metadatos como la ubicación GPS), fotos de reportes visibles
  solo para su autor y los admin.

## Ejecutar en tu computador

```bash
pip install -r requirements.txt
python app.py
```

Abre http://localhost:8000. La primera vez se crea la cuenta `admin` y su contraseña
aparece **una sola vez** en la terminal. Para elegirla tú:

```bash
MICROALERTA_ADMIN_PASSWORD="una-clave-larga" python app.py      # Mac/Linux
set MICROALERTA_ADMIN_PASSWORD=una-clave-larga && python app.py  # Windows
```

Agregar cuentas para colegios o vecinos:

```bash
python app.py --crear-usuario
```

## Pruebas

```bash
python -m pytest -q tests
```

## Variables de entorno

| Variable | Para qué |
|---|---|
| `MICROALERTA_SECRET` | Firma las sesiones. Obligatoria en Render. |
| `MICROALERTA_ADMIN_PASSWORD` | Contraseña de la cuenta admin inicial. |
| `MICROALERTA_DATA_DIR` | Carpeta de la base de datos y las fotos (en Render, la ruta del disco). |
| `MICROALERTA_HTTPS` | `1` en producción (cookies seguras). `app.py` lo pone en `0` en local. |

## Estructura

```
app.py              servidor local y alta de cuentas
wsgi.py             entrada para Render (waitress)
microalerta/
  __init__.py       crea la app y la cuenta admin
  config.py         protocolo, umbrales y niveles MP2,5/MP10
  science.py        part./mL, TDC, semáforo, avisos de control
  db.py             esquema SQLite
  api.py            sesión, reportes, validación, galería, Excel
  weather.py        clima de Open-Meteo para cada reporte
  excel.py          exportación
static/             la app (index.html), manifest, service worker e íconos
tests/              pruebas
```

Ver `DESPLIEGUE_RENDER.md` para publicarla en internet.
