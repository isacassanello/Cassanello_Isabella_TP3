import numpy as np

def cross_entropy(y_true, y_pred):
    """ Calcula la cross-entropy loss multiclase """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss