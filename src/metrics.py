import numpy as np
import pandas as pd

def cross_entropy(y_true, y_pred):
    """ 
    Calcula la cross-entropy loss multiclase 
    
    Parámetros:
    - y_true: matriz one-hot de tamaño (N, K)
    - y_pred: probabilidades predichas por softmax de tamaño (N, K)
    """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / N # la división por N es para calcular el promedio por muestra
    return loss

def accuracy_score(y_true, y_pred):
    """
    Calcula accuracy

    Parámetros:
    - y_true: etiquetas reales como enteros, shape (N,)
    - y_pred: etiquetas predichas como enteros, shape (N,)
    """
    return np.mean(y_true == y_pred)

def matriz_confusion(y_true, y_pred, num_classes=None):
    """
    Calcula la matriz de confusión multiclase.

    Filas: clases reales
    Columnas: clases predichas
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    matriz = np.zeros((num_classes, num_classes), dtype=int)

    for real, predicha in zip(y_true, y_pred):
        matriz[real, predicha] += 1

    return matriz

def precision_recall_f1_por_clase(y_true, y_pred, num_classes=None):
    """ Calcula precision, recall y F1 para cada clase """
    matriz = matriz_confusion(y_true, y_pred, num_classes)

    precision = np.zeros(matriz.shape[0])
    recall = np.zeros(matriz.shape[0])
    f1 = np.zeros(matriz.shape[0])

    for clase in range(matriz.shape[0]):
        tp = matriz[clase, clase]
        fp = np.sum(matriz[:, clase]) - tp
        fn = np.sum(matriz[clase, :]) - tp

        if tp + fp == 0:
            precision[clase] = 0
        else:
            precision[clase] = tp / (tp + fp)

        if tp + fn == 0:
            recall[clase] = 0
        else:
            recall[clase] = tp / (tp + fn)

        if precision[clase] + recall[clase] == 0:
            f1[clase] = 0
        else:
            f1[clase] = 2 * precision[clase] * recall[clase] / (precision[clase] + recall[clase])

    return precision, recall, f1

def f1_score_macro(y_true, y_pred, num_classes=None):
    """
    Calcula el F1-score macro

    Macro significa:
    1. Calcular el F1 de cada clase
    2. Promediar todos los F1, dando el mismo peso a cada clase
    """
    _, _, f1 = precision_recall_f1_por_clase(y_true, y_pred, num_classes)
    return np.mean(f1)

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
        "Accuracy": [acc],
        "Cross-Entropy": [ce],
        "F1 Macro": [f1_macro]
    })

    return tabla, matriz
