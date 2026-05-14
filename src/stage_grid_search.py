import numpy as np
import pandas as pd
import time

try:
    from pytorch_models import entrenar_modelo_pytorch
except ImportError:
    from src.pytorch_models import entrenar_modelo_pytorch

def normalizar_config_para_fit(config, epochs):
    """ Traduce los nombres usados en la búsqueda a los nombres que espera fit_avanzado """
   
    parametros_fit = {
        "epochs": epochs,
        "learning_rate": config.get("learning_rate", 0.1),
        "batch_size": config.get("batch_size", None),
        "optimizador": config.get("optimizer", "sgd"),
        "schedule": config.get("scheduler", None),
        "l2_lambda": config.get("lambda_l2", 0.0),
        "early_stopping": config.get("early_stopping", False),
        "patience": config.get("patience", 10),
    }

    scheduler = config.get("scheduler", None)
    scheduler_params = config.get("scheduler_params", {}) or {}

    if scheduler == "linear":
        parametros_fit["lr_final"] = scheduler_params.get(
            "lr_final",
            scheduler_params.get("lr_min", parametros_fit["learning_rate"] * 0.1),
        )
        parametros_fit["k"] = scheduler_params.get("k", epochs)

    elif scheduler == "exponential":
        parametros_fit["c"] = scheduler_params.get(
            "c",
            scheduler_params.get("decay_rate", 0.95),
        )
        parametros_fit["s"] = scheduler_params.get("s", 1)

    return parametros_fit


def entrenar_configuracion(
    modelo_class,
    X_train,
    y_train,
    X_val,
    y_val,
    arquitectura,
    input_dim,
    output_dim,
    config,
    epochs=30,
    seed=42,
):
    """ Entrena una configuración particular del modelo y devuelve métricas relevantes """
    
    layers = [input_dim] + list(arquitectura) + [output_dim]
    modelo = modelo_class(layers=layers, random_state=seed)

    parametros_fit = normalizar_config_para_fit(config, epochs)

    inicio = time.time()
    history = modelo.fit_avanzado(X_train, y_train, X_val=X_val, y_val=y_val, **parametros_fit,)
    tiempo = time.time() - inicio

    val_losses = history["val_loss"].to_numpy()
    mejor_epoch = int(np.argmin(val_losses))

    resultado = {
        "arquitectura": arquitectura,
        "layers": layers,
        "optimizer": config.get("optimizer", "sgd"),
        "learning_rate": config.get("learning_rate", None),
        "batch_size": config.get("batch_size", None),
        "scheduler": config.get("scheduler", None),
        "scheduler_params": config.get("scheduler_params", None),
        "lambda_l2": config.get("lambda_l2", 0.0),
        "early_stopping": config.get("early_stopping", False),
        "patience": config.get("patience", None),
        "epochs": len(history),
        "mejor_epoch": mejor_epoch,
        "train_loss_final": history["train_loss"].iloc[-1],
        "val_loss_final": history["val_loss"].iloc[-1],
        "mejor_val_loss": np.min(val_losses),
        "tiempo_segundos": tiempo,
    }

    return resultado, modelo, history


def agregar_resultado_stage(nombre_stage, resultados_stage, resultados_totales):
    df_stage = pd.DataFrame(resultados_stage)
    df_stage = df_stage.sort_values("mejor_val_loss").reset_index(drop=True)
    df_stage.insert(0, "stage", nombre_stage)
    resultados_totales.append(df_stage)
    return df_stage, df_stage.iloc[0].to_dict()


def stage_grid_search(modelo_class, X_train, y_train, X_val, y_val, input_dim, output_dim, epochs=30, seed=42,):
    """
    Realiza una búsqueda escalonada de hiperparámetros para encontrar M1.

    La búsqueda se ordena así:
    1. Arquitectura
    2. Optimizador, batch size y learning rate
    3. Learning rate schedule
    4. Regularización L2

    Devuelve las tablas de cada etapa, una tabla final combinada y la configuración M1.
    """

    resultados_totales = []
    mejores_modelos = {}
    mejores_histories = {}

    # STAGE 1: Arquitectura
    arquitecturas = [
        [128, 64],
        [256, 128],
        [256, 128, 64]
    ]

    config_base_arquitectura = {
        "optimizer": "sgd",
        "batch_size": 256,
        "learning_rate": 0.1,
        "scheduler": None,
        "lambda_l2": 0.0,
        "early_stopping": True,
        "patience": 10,
    }

    resultados_stage_1 = []

    for i, arquitectura in enumerate(arquitecturas):
        resultado, modelo, history = entrenar_configuracion(
            modelo_class,
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura,
            input_dim,
            output_dim,
            config_base_arquitectura,
            epochs=epochs,
            seed=seed,
        )

        resultados_stage_1.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_1, mejor_stage_1 = agregar_resultado_stage("stage_1_arquitectura", resultados_stage_1, resultados_totales)
    arquitectura_elegida = mejor_stage_1["arquitectura"]

    # STAGE 2: Optimizer + Batch Size + Learning Rate
    configs_stage_2 = []

    for lr in [0.05, 0.1, 0.2]:
        configs_stage_2.append({
            "optimizer": "sgd",
            "batch_size": 256,
            "learning_rate": lr,
            "scheduler": None,
            "lambda_l2": 0.0,
            "early_stopping": True,
            "patience": 10,
        })

    for lr in [0.001, 0.002]:
        configs_stage_2.append({
            "optimizer": "adam",
            "batch_size": 256,
            "learning_rate": lr,
            "scheduler": None,
            "lambda_l2": 0.0,
            "early_stopping": True,
            "patience": 10,
        })

    resultados_stage_2 = []

    for i, config in enumerate(configs_stage_2):
        resultado, modelo, history = entrenar_configuracion(
            modelo_class,
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
        )

        resultados_stage_2.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_2, mejor_stage_2 = agregar_resultado_stage("stage_2_optimizer_batch_lr", resultados_stage_2, resultados_totales,)

    # STAGE 3: Scheduling
    config_base_stage_3 = {
        "optimizer": mejor_stage_2["optimizer"],
        "batch_size": int(mejor_stage_2["batch_size"]),
        "learning_rate": mejor_stage_2["learning_rate"],
        "lambda_l2": 0.0,
        "early_stopping": True,
        "patience": 10,
    }

    configs_stage_3 = [
        {
            **config_base_stage_3,
            "scheduler": None,
            "scheduler_params": None,
        },
        {
            **config_base_stage_3,
            "scheduler": "linear",
            "scheduler_params": {
                "lr_final": config_base_stage_3["learning_rate"] * 0.1,
                "k": epochs,
            },
        },
        {
            **config_base_stage_3,
            "scheduler": "exponential",
            "scheduler_params": {
                "c": 0.95,
                "s": 1,
            },
        }
    ]

    resultados_stage_3 = []

    for i, config in enumerate(configs_stage_3):
        resultado, modelo, history = entrenar_configuracion(
            modelo_class,
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
        )

        resultados_stage_3.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_3, mejor_stage_3 = agregar_resultado_stage("stage_3_schedule", resultados_stage_3, resultados_totales)

    # STAGE 4: Regularización
    config_base_stage_4 = {
        "optimizer": mejor_stage_3["optimizer"],
        "batch_size": int(mejor_stage_3["batch_size"]),
        "learning_rate": mejor_stage_3["learning_rate"],
        "scheduler": mejor_stage_3["scheduler"],
        "scheduler_params": mejor_stage_3["scheduler_params"],
    }

    configs_stage_4 = []

    for lambda_l2 in [0.0, 1e-4, 1e-3]:
        configs_stage_4.append({
            **config_base_stage_4,
            "lambda_l2": lambda_l2,
            "early_stopping": True,
            "patience": 10,
        })

    resultados_stage_4 = []

    for i, config in enumerate(configs_stage_4):
        resultado, modelo, history = entrenar_configuracion(
            modelo_class,
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
        )

        resultados_stage_4.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_4, mejor_stage_4 = agregar_resultado_stage("stage_4_regularizacion", resultados_stage_4, resultados_totales,)

    resultados_completos = pd.concat(resultados_totales, ignore_index=True)
    resultados_completos = resultados_completos.sort_values("mejor_val_loss").reset_index(drop=True)

    mejor_global = resultados_completos.iloc[0].to_dict()
    mejor_val_loss_global = mejor_global["mejor_val_loss"]

    mejor_config_m1 = {
        "arquitectura": mejor_global["arquitectura"],
        "layers": mejor_global["layers"],
        "optimizer": mejor_global["optimizer"],
        "batch_size": int(mejor_global["batch_size"]),
        "learning_rate": mejor_global["learning_rate"],
        "scheduler": mejor_global["scheduler"],
        "scheduler_params": mejor_global["scheduler_params"],
        "lambda_l2": mejor_global["lambda_l2"],
        "early_stopping": mejor_global["early_stopping"],
        "patience": mejor_global["patience"],
        "mejor_val_loss": mejor_global["mejor_val_loss"],
    }

    return {
        "resultados_stage_1": df_stage_1,
        "resultados_stage_2": df_stage_2,
        "resultados_stage_3": df_stage_3,
        "resultados_stage_4": df_stage_4,
        "resultados_completos": resultados_completos,
        "mejor_config_m1": mejor_config_m1,
        "mejor_modelo_m1": mejores_modelos[mejor_val_loss_global],
        "mejor_history_m1": mejores_histories[mejor_val_loss_global],
    }


def entrenar_configuracion_pytorch(
    X_tr,
    y_tr,
    X_val,
    y_val,
    arquitectura,
    input_dim,
    output_dim,
    config,
    epochs=30,
    seed=42,
    device=None,
):
    layers = [input_dim] + list(arquitectura) + [output_dim]
    config_pytorch = {
        **config,
        "activacion": config.get("activacion", "relu"),
        "dropout": config.get("dropout", 0.0),
    }

    modelo, history, tiempo = entrenar_modelo_pytorch(
        layers,
        X_tr,
        y_tr,
        X_val,
        y_val,
        config_pytorch,
        epochs=epochs,
        device=device,
        seed=seed,
    )

    val_losses = history["val_loss"].to_numpy()
    mejor_epoch = int(np.argmin(val_losses))

    resultado = {
        "arquitectura": arquitectura,
        "layers": layers,
        "optimizer": config_pytorch.get("optimizer", "adam"),
        "learning_rate": config_pytorch.get("learning_rate", 0.001),
        "batch_size": config_pytorch.get("batch_size", X_tr.shape[0]),
        "scheduler": config_pytorch.get("scheduler", None),
        "scheduler_params": config_pytorch.get("scheduler_params", None),
        "lambda_l2": config_pytorch.get("lambda_l2", 0.0),
        "early_stopping": config_pytorch.get("early_stopping", True),
        "patience": config_pytorch.get("patience", 10),
        "activacion": config_pytorch.get("activacion", "relu"),
        "dropout": config_pytorch.get("dropout", 0.0),
        "epochs": len(history),
        "mejor_epoch": mejor_epoch,
        "train_loss_final": history["train_loss"].iloc[-1],
        "val_loss_final": history["val_loss"].iloc[-1],
        "mejor_val_loss": np.min(val_losses),
        "tiempo_segundos": tiempo,
    }

    return resultado, modelo, history


def stage_grid_search_pytorch(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    output_dim,
    config_base=None,
    epochs=30,
    seed=42,
    device=None,
):
    """
    Realiza una búsqueda escalonada de hiperparámetros en PyTorch para encontrar M3.

    La búsqueda se ordena así:
    1. Arquitectura
    2. Función de activación
    3. Dropout
    4. Regularización L2

    Devuelve las tablas de cada etapa, una tabla final combinada y la configuración M3.
    """

    if config_base is None:
        config_base = {}

    scheduler_params_base = config_base.get("scheduler_params", None)

    if scheduler_params_base is not None and not isinstance(scheduler_params_base, dict) and pd.isna(scheduler_params_base):
        scheduler_params_base = None

    config_base_pytorch = {
        "optimizer": config_base.get("optimizer", "adam"),
        "batch_size": config_base.get("batch_size", 256),
        "learning_rate": config_base.get("learning_rate", 0.001),
        "scheduler": config_base.get("scheduler", None),
        "scheduler_params": scheduler_params_base,
        "lambda_l2": config_base.get("lambda_l2", 0.0),
        "early_stopping": config_base.get("early_stopping", True),
        "patience": config_base.get("patience", 10),
        "activacion": "relu",
        "dropout": 0.0,
    }

    resultados_totales = []
    mejores_modelos = {}
    mejores_histories = {}

    # STAGE 1: Arquitectura
    arquitecturas = [
        [256, 128],
        [512, 256],
        [256, 128, 64],
        [512, 256, 128],
        [512, 256, 128, 64],
    ]

    resultados_stage_1 = []

    for i, arquitectura in enumerate(arquitecturas):
        resultado, modelo, history = entrenar_configuracion_pytorch(
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura,
            input_dim,
            output_dim,
            config_base_pytorch,
            epochs=epochs,
            seed=seed,
            device=device,
        )

        resultados_stage_1.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_1, mejor_stage_1 = agregar_resultado_stage(
        "stage_1_arquitectura_pytorch",
        resultados_stage_1,
        resultados_totales,
    )

    arquitectura_elegida = mejor_stage_1["arquitectura"]

    # STAGE 2: Activación
    configs_stage_2 = []

    for activacion in ["relu", "leaky_relu", "gelu", "silu"]:
        configs_stage_2.append({
            **config_base_pytorch,
            "activacion": activacion,
        })

    resultados_stage_2 = []

    for i, config in enumerate(configs_stage_2):
        resultado, modelo, history = entrenar_configuracion_pytorch(
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
            device=device,
        )

        resultados_stage_2.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_2, mejor_stage_2 = agregar_resultado_stage(
        "stage_2_activacion",
        resultados_stage_2,
        resultados_totales,
    )

    # STAGE 3: Dropout
    config_base_stage_3 = {
        **config_base_pytorch,
        "activacion": mejor_stage_2["activacion"],
    }

    configs_stage_3 = []

    for dropout in [0.0, 0.1, 0.2, 0.3]:
        configs_stage_3.append({
            **config_base_stage_3,
            "dropout": dropout,
        })

    resultados_stage_3 = []

    for i, config in enumerate(configs_stage_3):
        resultado, modelo, history = entrenar_configuracion_pytorch(
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
            device=device,
        )

        resultados_stage_3.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_3, mejor_stage_3 = agregar_resultado_stage(
        "stage_3_dropout",
        resultados_stage_3,
        resultados_totales,
    )

    # STAGE 4: Regularización L2
    config_base_stage_4 = {
        **config_base_stage_3,
        "dropout": mejor_stage_3["dropout"],
    }

    configs_stage_4 = []

    for lambda_l2 in [0.0, 1e-4, 1e-3]:
        configs_stage_4.append({
            **config_base_stage_4,
            "lambda_l2": lambda_l2,
        })

    resultados_stage_4 = []

    for i, config in enumerate(configs_stage_4):
        resultado, modelo, history = entrenar_configuracion_pytorch(
            X_train,
            y_train,
            X_val,
            y_val,
            arquitectura_elegida,
            input_dim,
            output_dim,
            config,
            epochs=epochs,
            seed=seed,
            device=device,
        )

        resultados_stage_4.append(resultado)
        mejores_modelos[resultado["mejor_val_loss"]] = modelo
        mejores_histories[resultado["mejor_val_loss"]] = history

    df_stage_4, mejor_stage_4 = agregar_resultado_stage(
        "stage_4_l2",
        resultados_stage_4,
        resultados_totales,
    )

    resultados_completos = pd.concat(resultados_totales, ignore_index=True)
    resultados_completos = resultados_completos.sort_values("mejor_val_loss").reset_index(drop=True)

    mejor_global = resultados_completos.iloc[0].to_dict()
    mejor_val_loss_global = mejor_global["mejor_val_loss"]

    mejor_config_m3 = {
        "arquitectura": mejor_global["arquitectura"],
        "layers": mejor_global["layers"],
        "optimizer": mejor_global["optimizer"],
        "batch_size": int(mejor_global["batch_size"]),
        "learning_rate": mejor_global["learning_rate"],
        "scheduler": mejor_global["scheduler"],
        "scheduler_params": mejor_global["scheduler_params"],
        "lambda_l2": mejor_global["lambda_l2"],
        "early_stopping": mejor_global["early_stopping"],
        "patience": mejor_global["patience"],
        "activacion": mejor_global["activacion"],
        "dropout": mejor_global["dropout"],
        "mejor_val_loss": mejor_global["mejor_val_loss"],
    }

    return {
        "resultados_stage_1": df_stage_1,
        "resultados_stage_2": df_stage_2,
        "resultados_stage_3": df_stage_3,
        "resultados_stage_4": df_stage_4,
        "resultados_completos": resultados_completos,
        "mejor_config_m3": mejor_config_m3,
        "mejor_modelo_m3": mejores_modelos[mejor_val_loss_global],
        "mejor_history_m3": mejores_histories[mejor_val_loss_global],
    }
