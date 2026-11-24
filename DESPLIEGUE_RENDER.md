# Publicar MicroAlerta en Render

## Importante: dónde se guardan los datos

En el plan **Free** de Render el disco se borra cada vez que el servicio se duerme
(15 minutos sin visitas) o se vuelve a desplegar. Eso significa que **los reportes,
las cuentas y las fotos se perderían**.

Para una presentación (LIYSF, ferias) tienes dos caminos:

1. **Con datos permanentes (recomendado):** plan *Starter* + un *Disk* de 1 GB montado
   en `/var/data`, y la variable `MICROALERTA_DATA_DIR=/var/data`. Tiene costo mensual;
   revisa el precio actual en render.com.
2. **Gratis:** usa el plan Free solo para mostrar la app y el flujo. Si necesitas que los
   reportes duren, usa la versión publicada en Claude, que guarda reportes y galería en
   su base compartida.

## Pasos

1. Sube esta carpeta a un repositorio de GitHub (el `.gitignore` ya excluye `data/`).
2. En https://dashboard.render.com: **New + → Web Service** y elige el repositorio.
3. Configura:

   | Campo | Valor |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `waitress-serve --host=0.0.0.0 --port=$PORT wsgi:app` |

4. En **Environment** agrega:

   | Key | Value |
   |---|---|
   | `MICROALERTA_SECRET` | una frase larga e inventada |
   | `MICROALERTA_ADMIN_PASSWORD` | la contraseña del admin (mínimo 10 caracteres) |
   | `MICROALERTA_DATA_DIR` | `/var/data` (solo si agregaste un Disk) |

5. **Create Web Service**. Cuando diga *Live*, abre la URL y entra con `admin` desde el
   botón ⚙ de la app.

## Después

- Crear cuentas para colegios: desde la pestaña *Shell* de Render,
  `python app.py --crear-usuario`.
- Descargar los datos: con sesión de admin, abre `https://TU-APP.onrender.com/api/export.xlsx`.
- Actualizar: cada `git push` vuelve a desplegar. Con plan Free eso borra los datos.

## Antes de presentar

- Cambia cualquier contraseña de prueba y no compartas la del admin.
- La primera apertura en plan Free tarda 30–60 s: ábrela unos minutos antes.
