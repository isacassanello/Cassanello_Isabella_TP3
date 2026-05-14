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

def graficar_robustez(tabla_robustez):
    """
    Grafica Accuracy, Cross-Entropy y F1 Macro frente a distintos niveles de ruido
    en una única figura.
    """

    metricas = ["Accuracy", "Cross-Entropy", "F1 Macro"]
    titulos = ["Accuracy frente a ruido", "Cross-Entropy frente a ruido", "F1 Macro frente a ruido",]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metrica, titulo in zip(axes, metricas, titulos):
        for modelo in tabla_robustez["Modelo"].unique():
            datos_modelo = tabla_robustez[tabla_robustez["Modelo"] == modelo]
            datos_modelo = datos_modelo.sort_values("Ruido sigma")

            ax.plot(
                datos_modelo["Ruido sigma"],
                datos_modelo[metrica],
                marker="o",
                label=modelo,
            )

        ax.set_xlabel("Nivel de ruido (sigma)")
        ax.set_ylabel(metrica)
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="best")
    plt.tight_layout()
    plt.show()

def graficar_robustez_metricas_juntas(tabla_robustez):
    tabla = tabla_robustez.copy()

    metricas = ["Accuracy", "F1 Macro", "Cross-Entropy"]

    for modelo in tabla["Modelo"].unique():
        datos_limpios = tabla[
            (tabla["Modelo"] == modelo) & (tabla["Ruido sigma"] == 0.0)
        ].iloc[0]

        for metrica in metricas:
            valor_limpio = datos_limpios[metrica]

            if metrica == "Cross-Entropy":
                tabla.loc[tabla["Modelo"] == modelo, f"Variación {metrica}"] = (
                    tabla.loc[tabla["Modelo"] == modelo, metrica] - valor_limpio
                )
            else:
                tabla.loc[tabla["Modelo"] == modelo, f"Variación {metrica}"] = (
                    valor_limpio - tabla.loc[tabla["Modelo"] == modelo, metrica]
                )

    plt.figure(figsize=(11, 6))

    estilos = {
        "Accuracy": "-",
        "F1 Macro": "--",
        "Cross-Entropy": ":",
    }

    for modelo in tabla["Modelo"].unique():
        datos_modelo = tabla[tabla["Modelo"] == modelo].sort_values("Ruido sigma")

        for metrica in metricas:
            plt.plot(
                datos_modelo["Ruido sigma"],
                datos_modelo[f"Variación {metrica}"],
                linestyle=estilos[metrica],
                marker="o",
                label=f"{modelo} - {metrica}"
            )

    plt.xlabel("Nivel de ruido (sigma)")
    plt.ylabel("Degradación respecto al test sin ruido")
    plt.title("Robustez frente a ruido")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
