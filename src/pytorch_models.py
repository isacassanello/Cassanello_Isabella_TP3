import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

class MLP_PyTorch(nn.Module): # nn.Module es la clase base de PyTorch para construir modelos
    def __init__(self, layers):
        super().__init__() # llama al constructor de nn.Module

        capas = []

        for i in range(len(layers) - 2):
            capas.append(nn.Linear(layers[i], layers[i + 1]))
            capas.append(nn.ReLU())

        capas.append(nn.Linear(layers[-2], layers[-1]))
        # no agrego softmax porque cuando use nn.CrossEntropyLoss(), la funcion combina softmax + cross entropy

        self.model = nn.Sequential(*capas) # convierto la lista de capas en un modelo secuencial -> ejecuta estas capas una después de la otra, en orden

    def forward(self, x):
        return self.model(x)
    
def entrenar_pytorch_m2(layers, X_tr, y_tr, X_val, y_val, config_m1, epochs=30, device=None):
    """
    Entrena el modelo M2 en PyTorch usando la misma arquitectura
    e hiperparámetros encontrados para M1
    """
    # device es dónde va a correr el modelo
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = MLP_PyTorch(layers).to(device)

    # convierte los datos de NumPy a tensores de PyTorch para calcular la cross entropy multiclase
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    loss_fn = nn.CrossEntropyLoss()

    lr = config_m1["learning_rate"]
    optimizer_name = config_m1["optimizer"]
    lambda_l2 = config_m1.get("lambda_l2", 0.0)

    # M2 usa el mismo optimizador que M1
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            modelo.parameters(), # toma todos los pesos y biases entrenables del modelo
            lr=lr,
            weight_decay=lambda_l2
        )
    else:
        optimizer = optim.SGD(
            modelo.parameters(),
            lr=lr,
            weight_decay=lambda_l2
        )

    batch_size = config_m1.get("batch_size", None)

    # si no se definió batch size, usa todo el conjunto de train como un único batch
    if batch_size is None:
        batch_size = X_tr.shape[0]

    historial = []
    inicio = time.time()

    for epoch in range(epochs):
        modelo.train()

        # crea una permutación aleatoria de los índices del dataset para los mini-batches
        indices = torch.randperm(X_tr_t.shape[0])

        tr_loss_acum = 0     # guarda la suma de las losses de todos los batches
        cantidad_batches = 0 # cuenta cuántos batches hubo

        for inicio_batch in range(0, X_tr_t.shape[0], batch_size):
            fin_batch = inicio_batch + batch_size
            batch_idx = indices[inicio_batch:fin_batch] # selecciona los índices aleatorios correspondientes a ese batch

            X_batch = X_tr_t[batch_idx]
            y_batch = y_tr_t[batch_idx]

            logits = modelo(X_batch) # forward propagation
            loss = loss_fn(logits, y_batch)

            # en PyTorch, los gradientes se acumulan por defecto -> antes de calcular los gradientes nuevos, hay que ponerlos en cero
            optimizer.zero_grad() # impia los gradientes anteriores
            loss.backward()
            optimizer.step() # actualiza los pesos usando los gradientes calculados

            tr_loss_acum += loss.item()
            cantidad_batches += 1

        tr_loss = tr_loss_acum / cantidad_batches

        modelo.eval() # pone el modelo en modo evaluación
        with torch.no_grad(): # no gaurda operaciones para calcular gradientes porque como en validation no se actualizan pesos, no hace falta backpropagation
            logits_val = modelo(X_val_t)
            val_loss = loss_fn(logits_val, y_val_t).item()

        historial.append({"epoch": epoch, "tr_loss": tr_loss, "val_loss": val_loss})

    tiempo = time.time() - inicio
    historial_df = pd.DataFrame(historial)
    return modelo, historial_df, tiempo