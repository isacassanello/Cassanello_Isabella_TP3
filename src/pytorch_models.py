import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

def crear_activacion_pytorch(nombre_activacion):
    activaciones = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }

    if nombre_activacion not in activaciones:
        raise ValueError(f"Activación no soportada: {nombre_activacion}")

    return activaciones[nombre_activacion]()


def crear_modelo_pytorch(layers, activacion="relu", dropout=0.0):
    capas = []

    for i in range(len(layers) - 2):
        capas.append(nn.Linear(layers[i], layers[i + 1]))
        capas.append(crear_activacion_pytorch(activacion))

        if dropout > 0:
            capas.append(nn.Dropout(p=dropout))

    capas.append(nn.Linear(layers[-2], layers[-1]))
    # no agrego softmax porque nn.CrossEntropyLoss() combina softmax + cross entropy
    return nn.Sequential(*capas)


class MLP_PyTorch(nn.Module): # nn.Module es la clase base de PyTorch para construir modelos
    def __init__(self, layers):
        super().__init__() # llama al constructor de nn.Module
        self.model = crear_modelo_pytorch(layers) # convierto la lista de capas en un modelo secuencial

    def forward(self, x):
        return self.model(x)


def calcular_lr_pytorch(epoch, lr_inicial, scheduler=None, scheduler_params=None):
    scheduler_params = scheduler_params or {}

    if scheduler == "linear":
        lr_final = scheduler_params.get("lr_final", lr_inicial * 0.1)
        k = scheduler_params.get("k", 30)
        progreso = min(epoch / k, 1)
        return lr_inicial + progreso * (lr_final - lr_inicial)

    if scheduler == "exponential":
        c = scheduler_params.get("c", 0.95)
        s = scheduler_params.get("s", 1)
        return lr_inicial * (c ** (epoch / s))

    return lr_inicial


def crear_optimizer_pytorch(modelo, config):
    lr = config.get("learning_rate", 0.001)
    lambda_l2 = config.get("lambda_l2", 0.0)
    optimizer_name = config.get("optimizer", "adam")

    if optimizer_name == "sgd":
        return optim.SGD(modelo.parameters(), lr=lr, weight_decay=lambda_l2)

    return optim.Adam(modelo.parameters(), lr=lr, weight_decay=lambda_l2)


def entrenar_modelo_pytorch(layers, X_tr, y_tr, X_val, y_val, config, epochs=30, device=None, seed=None):
    """
    Entrena una MLP en PyTorch. Sirve tanto para M2 como para las configuraciones
    que se prueban en la búsqueda de M3.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)

    modelo = crear_modelo_pytorch(
        layers,
        activacion=config.get("activacion", "relu"),
        dropout=config.get("dropout", 0.0),
    ).to(device)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = crear_optimizer_pytorch(modelo, config)

    lr_inicial = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", None)
    scheduler = config.get("scheduler", None)
    scheduler_params = config.get("scheduler_params", None)
    early_stopping = config.get("early_stopping", False)
    patience = config.get("patience", 10)

    if batch_size is None:
        batch_size = X_tr.shape[0]

    historial = []
    mejor_val_loss = float("inf")
    mejor_estado = None
    epochs_sin_mejora = 0
    inicio = time.time()

    for epoch in range(epochs):
        lr_epoch = calcular_lr_pytorch(epoch, lr_inicial, scheduler, scheduler_params)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_epoch

        modelo.train()
        indices = torch.randperm(X_tr_t.shape[0], device=device)
        train_loss_acum = 0.0
        cantidad_batches = 0

        for inicio_batch in range(0, X_tr_t.shape[0], batch_size):
            fin_batch = inicio_batch + batch_size
            batch_idx = indices[inicio_batch:fin_batch]

            X_batch = X_tr_t[batch_idx]
            y_batch = y_tr_t[batch_idx]

            logits = modelo(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_acum += loss.item()
            cantidad_batches += 1

        train_loss = train_loss_acum / cantidad_batches

        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(X_val_t)
            val_loss = loss_fn(logits_val, y_val_t).item()

        historial.append({
            "epoch": epoch,
            "learning_rate": lr_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_estado = {
                nombre: parametro.detach().cpu().clone()
                for nombre, parametro in modelo.state_dict().items()
            }
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if early_stopping and epochs_sin_mejora >= patience:
            break

    tiempo = time.time() - inicio

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    historial_df = pd.DataFrame(historial)
    return modelo, historial_df, tiempo


def entrenar_pytorch_m2(layers, X_tr, y_tr, X_val, y_val, config_m1, epochs=30, device=None):
    """
    Entrena el modelo M2 en PyTorch usando la misma arquitectura
    e hiperparámetros encontrados para M1.
    """
    config_m2 = {
        **config_m1,
        "activacion": "relu",
        "dropout": 0.0,
    }

    modelo, historial_df, tiempo = entrenar_modelo_pytorch(
        layers,
        X_tr,
        y_tr,
        X_val,
        y_val,
        config_m2,
        epochs=epochs,
        device=device,
    )

    historial_df = historial_df.rename(columns={"train_loss": "tr_loss"})
    return modelo, historial_df, tiempo
