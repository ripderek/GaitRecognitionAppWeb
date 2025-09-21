
import os
import time
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import controllers.EntrenamientoController as ec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, classification_report,make_scorer, f1_score, precision_score, recall_score, balanced_accuracy_score



# Variable global para acumular logs
log_texto_RF = ""

    
#entrenamiento con matriz de confusion:
def EntrenamientoRF(numero):
    global log_texto_RF
    # Paso 1: Obtener JSON desde PostgreSQL
    resultado = ec.ObtenerDatosEntrenamiento(numero)  # El JSON como dict/list
    datos = resultado
    X = []
    y = []

    print(f"RF -> Iniciando entrenamiento Modelo_{numero}")
    log_texto_RF += f"{datetime.now()} - RF -> Iniciando entrenamiento Modelo_{numero}\n"

    # Paso 2: Convertir muestras en vectores
    for muestra in datos:
        vector = []
        puntos = muestra["puntos"]

        for clave in sorted(puntos.keys()):
            vector.append(puntos[clave]["promedio"])
            vector.append(puntos[clave]["desviacion"])

        X.append(vector)
        y.append(muestra["persona"])

    # Definir Random Forest y parámetros a optimizar
    modelo = RandomForestClassifier(random_state=42)

    valores, cuentas = np.unique(y, return_counts=True)
    print(pd.DataFrame({"Clase": valores, "Cantidad": cuentas}))
    log_texto_RF += pd.DataFrame({"Clase": valores, "Cantidad": cuentas}).to_string() + "\n"

    # Definir el espacio de búsqueda de hiperparámetros
    param_dist = {
        "n_estimators": randint(50, 500),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    # Definir métricas personalizadas
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=50,
        cv=8,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring=scoring,
        refit="f1_macro"
    )

    # Paso 3: Entrenar modelo
    inicio = time.time()
    random_search.fit(X, y)
    fin = time.time()

    mejor_modelo = random_search.best_estimator_
    mejor_params = random_search.best_params_
    mejor_score = random_search.best_score_  # F1-macro promedio
    resultados = pd.DataFrame(random_search.cv_results_)  # todos los resultados
    duracion = fin - inicio

    # Paso 4: Guardar modelo
    os.makedirs("MODELOS/RF", exist_ok=True)
    nombre_archivo = f"MODELOS/RF/modelo_{numero}.joblib"
    dump(mejor_modelo, nombre_archivo)

    # Logs
    print(f"Modelo entrenado y guardado exitosamente como: {nombre_archivo}")
    log_texto_RF += f"{datetime.now()} - Modelo entrenado y guardado exitosamente como: {nombre_archivo}\n"
    print(f"Tiempo de entrenamiento: {duracion:.2f} segundos")
    log_texto_RF += f"{datetime.now()} - Tiempo de entrenamiento: {duracion:.2f} segundos\n"
    print(f"Mejores parámetros: {mejor_params}")
    log_texto_RF += f"{datetime.now()} - Mejores parámetros: {mejor_params}\n"
    print(f"Mejor F1-macro promedio (cv=8): {mejor_score:.4f}")
    log_texto_RF += f"{datetime.now()} - Mejor F1-macro promedio (cv=8): {mejor_score:.4f}\n"

    print("\nResumen de combinaciones probadas (Top 5 ordenadas por F1-macro):")
    log_texto_RF += f"{datetime.now()} - Resumen de combinaciones probadas (Top 5 ordenadas por F1-macro):\n"

    print(resultados[["params", "mean_test_accuracy", "mean_test_f1_macro",
                      "mean_test_precision_macro", "mean_test_recall_macro",
                      "mean_test_balanced_accuracy"]]
          .sort_values(by="mean_test_f1_macro", ascending=False)
          .head())
    
    log_texto_RF += (resultados[["params", "mean_test_accuracy", "mean_test_f1_macro",
                      "mean_test_precision_macro", "mean_test_recall_macro",
                      "mean_test_balanced_accuracy"]]
          .sort_values(by="mean_test_f1_macro", ascending=False)
          .head().to_string() + "\n")

    # Obtener JSON desde PostgreSQL de evaluacion
    resultado_ = ec.ObtenerDatosEvaluacion(numero)  # El JSON como dict/list
    datos_ = resultado_
    X_entre = []
    y_entre = []

    print(f"RF -> Obteniendo muestas para evaluacion_{numero}")
    log_texto_RF += f"{datetime.now()} - RF -> Obteniendo muestas para evaluacion_{numero}\n"

    # Paso 2: Convertir muestras en vectores
    for muestra in datos_:
        vector = []
        puntos = muestra["puntos"]

        for clave in sorted(puntos.keys()):
            vector.append(puntos[clave]["promedio"])
            vector.append(puntos[clave]["desviacion"])

        X_entre.append(vector)
        y_entre.append(muestra["persona"])



    # ------------------------------
    # Paso 5: Matriz de confusión
    # ------------------------------
    # Para un resultado aproximado, usamos todo X e y
    # Idealmente usar X_test e y_test si tienes un conjunto separado de prueba

    #hacer un consulta sql para obtener las muestras de evaluacion para construir la matriz de confusion
    y_pred = mejor_modelo.predict(X_entre)

    # Matriz de confusión
    #cm = confusion_matrix(y_entre, y_pred, labels=valores)
    #print("\nMatriz de Confusión:")
    #print(pd.DataFrame(cm, index=valores, columns=valores))

    # Reporte de métricas por clase
    #print("\nReporte de clasificación (Precision, Recall, F1 por clase):")
    #print(classification_report(y_entre, y_pred, labels=valores, digits=4))

    # Matriz de confusión
    cm = confusion_matrix(y_entre, y_pred, labels=valores)
    cm_df = pd.DataFrame(cm, index=valores, columns=valores)

    # Reporte de métricas por clase
    report = classification_report(y_entre, y_pred, labels=valores, digits=4)

    # Mostrar en consola (opcional)
    print("\nMatriz de Confusión:")
    print(cm_df)
    log_texto_RF += f"{datetime.now()} - Matriz de Confusión:\n"
    log_texto_RF += cm_df.to_string() + "\n"

    print("\nReporte de clasificación (Precision, Recall, F1 por clase):")
    print(report)

    log_texto_RF += f"{datetime.now()} - Reporte de clasificación (Precision, Recall, F1 por clase):\n"
    log_texto_RF += report + "\n"

    # ------------------------------
    # Paso 6: Guardar en un txt la matriz de confusion y la tabla con las metricas obtenidas
    # ------------------------------

    with open(f"resultados_modelo_{numero}.txt", "w", encoding="utf-8") as f:
        f.write("Matriz de Confusión:\n")
        f.write(cm_df.to_string())  # convierte el DataFrame en texto
        f.write("\n\nReporte de Clasificación:\n")
        f.write(report)
    
    log_texto_RF += f"{datetime.now()} - Modelo entrenado y guardado exitosamente como: {nombre_archivo}\n"

def reset_log_RF():
    global log_texto_RF
    log_texto_RF = ""






"""

if __name__ == "__main__":
    EntrenamientoRF(1)
    EntrenamientoRF(2)
    EntrenamientoRF(3)
"""

