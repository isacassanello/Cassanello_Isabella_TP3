import numpy as np

def normalizar_imagenes(X):
    """ Normaliza las imágenes dividiendo todos los valores por 255 """
    return X.astype(np.float32) / 255.0

def one_hot_encode(y, num_classes=None):
    """
    Convierte etiquetas enteras a formato one-hot
    """

    y = np.array(y).astype(int)

    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot

def flatten_imagenes(X):
    """
    Convierte imágenes de (n, 28, 28) a vectores de (n, 784)
    """
    return X.reshape(X.shape[0], -1)