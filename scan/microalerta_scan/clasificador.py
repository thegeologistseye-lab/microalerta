"""Clasificador tabular PCM vs. descarte (Random Forest).

Se entrena con partículas etiquetadas a mano (y, idealmente, validadas por Raman).
Etiquetas sugeridas en la columna `etiqueta`: PCM, organico, mineral, residuo_sal, otro.
Todo lo que no sea PCM cuenta como descarte, pero se conserva la etiqueta fina.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

from .rasgos import RASGOS_MODELO


def _matriz(df, rasgos):
    X = df.reindex(columns=rasgos).astype(float)
    # las modalidades ausentes (sin foto polarizada / fluorescencia) se codifican como 0 + bandera
    for col in ("indice_birrefringencia", "birrefringencia_p95", "frac_brillante_pol", "indice_fluorescencia"):
        if col in X:
            X[col + "_falta"] = X[col].isna().astype(float)
            X[col] = X[col].fillna(0.0)
    return X.fillna(0.0)


def entrenar(csv_etiquetado, salida_modelo, semilla=0):
    df = pd.read_csv(csv_etiquetado)
    df = df[df["etiqueta"].notna() & (df["etiqueta"].astype(str).str.strip() != "")]
    if len(df) < 10:
        raise ValueError(f"Solo hay {len(df)} partículas etiquetadas; se necesitan al menos ~50 para un modelo útil.")
    y = (df["etiqueta"].str.strip().str.upper() == "PCM").astype(int).values
    X = _matriz(df, RASGOS_MODELO)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                random_state=semilla, n_jobs=-1)
    informe = {"n": int(len(df)), "n_pcm": int(y.sum()), "n_descarte": int((1 - y).sum())}
    k = int(min(5, np.bincount(y).min()))
    # Validación cruzada por disco (muestra_id): las partículas de un mismo disco se parecen mucho, y
    # una misma partícula puede salir en dos campos. Si se reparten al azar, la nota sale inflada.
    grupos = df["muestra_id"].astype(str).values if "muestra_id" in df else None
    n_grupos = len(set(grupos)) if grupos is not None else 0
    if k >= 2 and n_grupos >= 2:
        k = min(k, n_grupos)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
        pred = cross_val_predict(rf, X, y, cv=cv, groups=grupos)
        informe["validacion_cruzada"] = f"por disco (muestra_id), {k} grupos"
    elif k >= 2:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
        pred = cross_val_predict(rf, X, y, cv=cv)
        informe["validacion_cruzada"] = f"por partícula, {k} grupos"
        informe["aviso"] = ("Todas las partículas vienen de un solo disco: la validación cruzada mezcla partículas "
                            "muy parecidas y sobreestima el desempeño. Use la prueba con discos aparte (probar).")
    if k >= 2:
        informe["validacion_cruzada_folds"] = k
        informe["matriz_confusion [[desc->desc, desc->PCM],[PCM->desc, PCM->PCM]]"] = confusion_matrix(y, pred).tolist()
        informe["reporte"] = classification_report(y, pred, target_names=["descarte", "PCM"], output_dict=True,
                                                   zero_division=0)
    rf.fit(X, y)
    informe["importancia_rasgos"] = dict(sorted(zip(X.columns, rf.feature_importances_.round(4).tolist()),
                                                key=lambda t: -t[1]))
    joblib.dump({"modelo": rf, "columnas": list(X.columns), "version": 1}, salida_modelo)
    return informe


def cargar_modelo(path):
    return joblib.load(path)


def predecir(paquete, filas):
    if not filas:
        return []
    df = pd.DataFrame(filas)
    X = _matriz(df, RASGOS_MODELO).reindex(columns=paquete["columnas"], fill_value=0.0)
    return paquete["modelo"].predict_proba(X)[:, 1].tolist()


def probar(modelo_path, csv_prueba, meta=0.80, umbral_p=0.5, csv_entrenamiento=None):
    """Prueba con partículas que el modelo no vio: matriz de confusión, precisión y sensibilidad.

    La meta (p. ej. 0,80 en ambas) se fija y se anota ANTES de correr la prueba. Si se da el CSV de
    entrenamiento, revisa que ningún disco (muestra_id) esté en los dos conjuntos."""
    paquete = cargar_modelo(modelo_path)
    df = pd.read_csv(csv_prueba)
    df = df[df["etiqueta"].notna() & (df["etiqueta"].astype(str).str.strip() != "")]
    y = (df["etiqueta"].str.strip().str.upper() == "PCM").astype(int).values
    p = np.array(predecir(paquete, df.to_dict("records")))
    pred = (p >= umbral_p).astype(int)
    prec = float(precision_score(y, pred, zero_division=0))
    sens = float(recall_score(y, pred, zero_division=0))
    inf = {"n": int(len(df)), "n_pcm": int(y.sum()), "n_descarte": int((1 - y).sum()),
           "matriz_confusion [[desc->desc, desc->PCM],[PCM->desc, PCM->PCM]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
           "precision": prec, "sensibilidad": sens, "meta": meta,
           "aprueba": bool(prec >= meta and sens >= meta and y.sum() >= 20 and (1 - y).sum() >= 20)}
    avisos = []
    if y.sum() < 20 or (1 - y).sum() < 20:
        avisos.append("Hay menos de 20 partículas por clase en la prueba: no alcanza para aprobar.")
    if csv_entrenamiento and "muestra_id" in df:
        ent = pd.read_csv(csv_entrenamiento)
        comunes = sorted(set(df["muestra_id"].astype(str)) & set(ent.get("muestra_id", pd.Series(dtype=str)).astype(str)))
        if comunes:
            inf["aprueba"] = False
            avisos.append(f"Los discos {', '.join(comunes)} están en el entrenamiento y en la prueba: la prueba no es independiente.")
    inf["avisos"] = avisos
    inf["informar_como"] = "PCM" if inf["aprueba"] else "candidatas detectadas por fluorescencia"
    return inf


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Entrena el clasificador PCM vs descarte, o lo prueba con discos aparte")
    ap.add_argument("csv", help="particulas.csv con la columna 'etiqueta' completada")
    ap.add_argument("--salida", default="modelo_pcm.joblib")
    ap.add_argument("--probar", metavar="MODELO", help="en vez de entrenar, probar este modelo con el CSV (discos que no vio)")
    ap.add_argument("--meta", type=float, default=0.80, help="precisión y sensibilidad mínimas, fijadas antes")
    ap.add_argument("--entrenamiento", help="CSV con que se entrenó, para revisar que no se repitan discos")
    a = ap.parse_args()
    if a.probar:
        print(json.dumps(probar(a.probar, a.csv, a.meta, csv_entrenamiento=a.entrenamiento), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(entrenar(a.csv, a.salida), indent=2, ensure_ascii=False))
