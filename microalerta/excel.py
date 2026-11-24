"""Exportación de reportes a Excel para el equipo y la comunidad científica."""
import io
import json

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

COLUMNAS = ["ID", "Fecha", "Punto", "Placa", "Tipo", "Grado", "Partículas", "Control",
            "Volumen (mL)", "Área (cm²)", "Horas", "part./mL", "TDC PCM/(m²·día)", "Nivel",
            "Estado", "Comentario", "Autor", "Temp. (°C)", "Humedad (%)", "Viento (km/h)",
            "Dirección (°)", "Fuente clima", "Observaciones"]


def libro(reportes) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Reportes"
    ws.append(COLUMNAS)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="13203A")
    for r in reportes:
        cl = json.loads(r["clima"]) if r["clima"] else {}
        ws.append([r["id"], r["creado"], r["punto"], r["placa"], r["tipo"], r["grado"],
                   r["particulas"], r["control"], r["volumen_ml"], r["area_cm2"], r["horas"],
                   round(r["conc"], 3), round(r["tdc"]), r["nivel"], r["estado"], r["comentario"],
                   r["autor"], cl.get("temperatura"), cl.get("humedad"), cl.get("viento"),
                   cl.get("direccion"), cl.get("fuente"), r["obs"]])
    for i, _ in enumerate(COLUMNAS, 1):
        ws.column_dimensions[get_column_letter(i)].width = 14
    ws.freeze_panes = "A2"

    res = wb.create_sheet("Resumen por punto")
    res.append(["Punto", "Reportes aprobados", "Promedio part./mL", "Máximo part./mL", "Último nivel"])
    for c in res[1]:
        c.font = Font(bold=True)
    por = {}
    for r in reportes:
        if r["estado"] == "aprobado" and "blanco" not in r["tipo"].lower():
            por.setdefault(r["punto"], []).append(r)
    for punto, rs in sorted(por.items()):
        vals = [x["conc"] for x in rs]
        res.append([punto, len(rs), round(sum(vals) / len(vals), 3), round(max(vals), 3), rs[-1]["nivel"]])
    res.column_dimensions["A"].width = 22
    nota = wb.create_sheet("Notas")
    for linea in [
        "MicroAlerta · Colegio Cervantino, Copiapó",
        "part./mL: partículas compatibles con microplásticos por mL de agua de la placa.",
        "TDC = máx(0, muestra − control) / (área × días), en PCM/(m²·día).",
        "Clasificación óptica presuntiva: solo Raman o FTIR confirman el polímero.",
        "Semáforo exploratorio: no es una escala sanitaria ni mide concentración en el aire.",
    ]:
        nota.append([linea])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
