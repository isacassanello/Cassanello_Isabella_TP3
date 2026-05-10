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