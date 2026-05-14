import numpy as np
import matplotlib.pyplot as plt

def visualizar_imagenes(X_images, y_images, cantidad=3):
    """
    Visualiza una cantidad determinada de imágenes del dataset junto con sus etiquetas.

    Parámetros:
    - X_imagesL Arreglo con las imágenes del dataset.
    - y_images: Arreglo con las etiquetas correspondientes a cada imagen.
    - cantidad: Cantidad de imágenes a visualizar.
    """

    fig, axes = plt.subplots(1, cantidad, figsize=(4 * cantidad, 4))

    if cantidad == 1:
        axes = [axes]

    for i in range(cantidad):
        img = X_images[i].reshape(28, 28)

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {y_images[i]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def graficar_distribucion_clases(y_images):
    """
    Grafica la cantidad de ejemplos por clase para analizar
    si el dataset está balanceado o desbalanceado.
    """
    clases, conteos = np.unique(y_images, return_counts=True)

    plt.figure(figsize=(14, 5))

    plt.bar(clases, conteos, color="#8ecae6")

    plt.xlabel("Clase")
    plt.ylabel("Cantidad de imágenes")
    plt.title("Distribución de clases en el dataset EMNIST ByMerge")

    plt.xticks(clases)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

def graficar_funcion_costo(historial):
    """
    Grafica la evolución de la función de costo en train y validation.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(historial["epoch"], historial["train_loss"], label="Entrenamiento")
    plt.plot(historial["epoch"], historial["val_loss"], label="Validación")

    plt.xlabel("Época")
    plt.ylabel("Función de costo (cross-entropy)")
    plt.title("Evolución de la función de costo")
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_matrices_confusion(matriz_train, matriz_val):
    """ Grafica las matrices de confusión de train y validation """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    matrices = [matriz_train, matriz_val]
    titulos = ["Matriz de Confusión - Train", "Matriz de Confusión - Validation"]

    for ax, matriz, titulo in zip(axes, matrices, titulos):
        im = ax.imshow(matriz, cmap="Greens")

        n_clases = matriz.shape[0]

        ax.set_title(titulo)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")

        ax.set_xticks(np.arange(n_clases))
        ax.set_yticks(np.arange(n_clases))

        ax.set_xticklabels(np.arange(n_clases), rotation=90, fontsize=7)
        ax.set_yticklabels(np.arange(n_clases), fontsize=7)

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def graficar_comparacion_modelos(tabla_comparacion):
    """
    Grafica comparación de modelos según:
    - Accuracy
    - F1 Macro
    - Cross-Entropy
    - Tiempo de entrenamiento
    """

    metricas = ["Accuracy", "F1 Macro", "Cross-Entropy", "Tiempo entrenamiento (seg)"]
    titulos = ["Comparación de Accuracy", "Comparación de F1 Macro", "Comparación de Cross-Entropy", "Comparación de Tiempo de Entrenamiento"]
    colores = ["#8ecae6", "#e5bdfe", "#fedd89", "#d481be"]

    metricas_disponibles = [
        (metrica, titulo, color)
        for metrica, titulo, color in zip(metricas, titulos, colores)
        if metrica in tabla_comparacion.columns
    ]

    modelos = tabla_comparacion["Modelo"].astype(str)

    for metrica, titulo, color in metricas_disponibles:

        plt.figure(figsize=(10, 5))

        plt.bar(modelos, tabla_comparacion[metrica], color=color, alpha=0.85)

        plt.title(titulo)
        plt.xlabel("Modelo")
        plt.ylabel(metrica)

        plt.xticks(rotation=45, ha="right")

        for i, valor in enumerate(tabla_comparacion[metrica]):
            plt.text(i, valor, f"{valor:.4f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.show()

def graficar_matrices_confusion_modelos(matrices, nombres_modelos):
    """
    Grafica en subplots las matrices de confusión de los modelos con mejoras

    Parámetros:
    - matrices: lista de matrices de confusión
    - nombres_modelos: lista con el nombre de cada modelo
    - columnas: cantidad de columnas del subplot
    """

    cantidad = len(matrices)
    filas = int(np.ceil(cantidad / 3))

    fig, axes = plt.subplots(filas, 3, figsize=(6 * 3, 5 * filas))

    axes = np.array(axes).reshape(-1)

    for i, (matriz, nombre) in enumerate(zip(matrices, nombres_modelos)):
        ax = axes[i]

        im = ax.imshow(matriz, cmap="Greens")

        n_clases = matriz.shape[0]
        ticks = np.arange(0, n_clases, 2)

        ax.set_title(f"Matriz de Confusión - {nombre}")
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels(ticks, rotation=90, fontsize=7)
        ax.set_yticklabels(ticks, fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(cantidad, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
