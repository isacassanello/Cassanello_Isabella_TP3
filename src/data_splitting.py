import numpy as np

def split_train_val_test(X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divide el dataset en Train, Validation y Test de forma aleatoria estratificada.

    La estratificación mantiene aproximadamente la misma proporción de clases
    en los tres subconjuntos.
    """
    np.random.seed(random_state)

    train_indices = []
    val_indices = []
    test_indices = []

    clases = np.unique(y)

    for clase in clases:
        indices_clase = np.where(y == clase)[0]
        np.random.shuffle(indices_clase)

        n_total = len(indices_clase)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)

        train_indices.extend(indices_clase[:n_train])
        val_indices.extend(indices_clase[n_train:n_train + n_val])
        test_indices.extend(indices_clase[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test