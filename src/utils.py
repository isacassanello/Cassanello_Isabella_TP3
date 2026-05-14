import numpy as np
import pandas as pd
import time
import torch
from metrics import accuracy_score, cross_entropy, matriz_confusion, f1_score_macro
from preprocessing import one_hot_encode

def evaluar_modelo(modelo, X, y_true, y_true_onehot, nombre_conjunto, num_classes=47):
    """
    Evalúa un modelo ya entrenado.

    Devuelve:
    - accuracy
    - cross-entropy
    - matriz de confusión
    - f1 macro
    """

    y_pred_proba = modelo.forward(X)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    ce = cross_entropy(y_true_onehot, y_pred_proba)
    matriz = matriz_confusion(y_true, y_pred, num_classes)
    f1_macro = f1_score_macro(y_true, y_pred, num_classes)

    tabla = pd.DataFrame({
        "Conjunto": [nombre_conjunto],
        "Modelo": [nombre_conjunto],
        "Accuracy": [acc],
        "Cross-Entropy": [ce],
        "F1 Macro": [f1_macro]
    })

    return tabla, matriz

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

def evaluar_modelo_pytorch(modelo, X, y_true, nombre_conjunto, num_classes=47, device=None, tiempo_entrenamiento=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modelo.eval()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = modelo(X_t)
        probabilidades = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = np.argmax(probabilidades, axis=1)
    y_true_onehot = one_hot_encode(y_true, num_classes)

    acc = accuracy_score(y_true, y_pred)
    ce = cross_entropy(y_true_onehot, probabilidades)
    matriz = matriz_confusion(y_true, y_pred, num_classes)
    f1_macro = f1_score_macro(y_true, y_pred, num_classes)

    tabla = pd.DataFrame({
        "Conjunto": [nombre_conjunto],
        "Modelo": [nombre_conjunto],
        "Accuracy": [acc],
        "Cross-Entropy": [ce],
        "F1 Macro": [f1_macro]
    })

    if tiempo_entrenamiento is not None:
        tabla["Tiempo entrenamiento (seg)"] = tiempo_entrenamiento

    return tabla, matriz
