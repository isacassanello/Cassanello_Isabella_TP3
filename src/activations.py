import numpy as np

def relu(x):
    """ Función de activación ReLU """
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def softmax(x):
    """ Función softmax aplicada por filas """
    # Estabilidad numérica
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(x_shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)