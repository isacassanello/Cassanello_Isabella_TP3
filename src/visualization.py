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

        ax.set_title(titulo)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
