import numpy as np

def normalizar_imagenes(X):
    """ Normaliza las imágenes dividiendo todos los valores por 255 """
    return X.astype(np.float32) / 255.0

def flatten_imagenes(X):
    """
    Convierte imágenes de (n, 28, 28) a vectores de (n, 784)
    """
    return X.reshape(X.shape[0], -1)