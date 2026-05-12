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

def graficar_loss(historial):
    """
    Grafica la evolución de la cross-entropy en train y validation
    """

    plt.figure(figsize=(8, 5))

    plt.plot(historial["epoch"], historial["train_loss"], label="Train")
    plt.plot(historial["epoch"], historial["val_loss"], label="Validation")

    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy")
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
        im = ax.imshow(matriz)

        ax.set_title(titulo)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()