import time
from metrics import evaluar_modelo

def entrenar_y_evaluar_modelo(nombre, modelo, X_tr, y_tr_onehot, X_val, y_val, y_val_onehot, parametros_fit, num_classes=47):
    """
    Entrena un modelo con fit_avanzado y evalúa su performance en validation.

    Devuelve:
    - historial del entrenamiento
    - tabla de métricas
    - matriz de confusión
    """
    inicio = time.time()

    historial = modelo.fit_avanzado(X_tr, y_tr_onehot, X_val, y_val_onehot, **parametros_fit)

    tiempo = time.time() - inicio

    tabla_val, matriz_val = evaluar_modelo(modelo, X_val, y_val, y_val_onehot, nombre_conjunto=nombre, num_classes=num_classes)

    tabla_val["Modelo"] = nombre
    tabla_val["Tiempo entrenamiento (seg)"] = tiempo

    return historial, tabla_val, matriz_val