import numpy as np
import pandas as pd
from activations import relu, relu_derivada, softmax
from metrics import cross_entropy

class MLP:
    def __init__(self, layers, random_state=42, kappa=0.01):
        np.random.seed(random_state)
        self.layers = layers
        self.num_layers = len(layers) - 1

        self.weights = []
        self.biases = []

        # inicializacion de pesos y biases
        for layer in range(self.num_layers):
            n_input = layers[layer]
            n_output = layers[layer + 1]

            # inicialización He para los pesos
            W = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(n_input, n_output)).astype(np.float32)

            # biases pequeños y positivos
            b = np.random.uniform(0, kappa**2, size=(1, n_output)).astype(np.float32)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """ Realiza forward propagation """

        self.activaciones = [X]
        self.pre_activaciones = []

        Z = X

        # capas ocultas
        for layer in range(self.num_layers - 1):
            A = Z @ self.weights[layer] + self.biases[layer]
            Z = relu(A)

            self.pre_activaciones.append(A)
            self.activaciones.append(Z)

        # capa de salida
        A = Z @ self.weights[-1] + self.biases[-1]
        Y_hat = softmax(A)

        self.pre_activaciones.append(A)
        self.activaciones.append(Y_hat)   

        return Y_hat 
    
    def backward(self, y_true):
        """ Calcula los gradientes usando backpropagation """
        N = y_true.shape[0]

        self.dW = [None] * self.num_layers
        self.db = [None] * self.num_layers

        # gradiente de la capa de salida
        dA = self.activaciones[-1] - y_true

        for layer in reversed(range(self.num_layers)):
            # gradiente de los pesos
            z_anterior = self.activaciones[layer]
            self.dW[layer] = (z_anterior.T @ dA) / N

            # gradiente de los biases
            self.db[layer] = np.sum(dA, axis=0, keepdims=True) / N

            # propago hacia atras si no llegue a la primera capa
            if layer > 0:
                dZ_anterior = dA @ self.weights[layer].T
                pre_act_anterior = self.pre_activaciones[layer - 1]
                dA = dZ_anterior * relu_derivada(pre_act_anterior)

    def update_params_sgd(self, learning_rate):
        """ Actualiza pesos y biases usando gradiente descendente """
        for layer in range(self.num_layers):
            self.weights[layer] -= learning_rate * self.dW[layer]
            self.biases[layer] -= learning_rate * self.db[layer]

    def calcular_lr(self, epoch, lr_inicial, schedule=None, lr_final=0.0001, k=50, c=0.70, s=1):
        if schedule is None:
            return lr_inicial
        
        elif schedule=="linear":
            if epoch >= k:
                lr = lr_final
            else:
                lr = (1 - epoch/k) * lr_inicial + (epoch/k) * lr_final
            return lr
        
        elif schedule=="exponential":
            lr = lr_inicial * (c ** (epoch/s))
            return lr
        
    def inicializar_adam(self):
        """
        s = promedio móvil del gradiente
        r = promedio móvil del gradiente al cuadrado
        """
        self.sW = [np.zeros_like(W) for W in self.weights]
        self.rW = [np.zeros_like(W) for W in self.weights]

        self.sb = [np.zeros_like(b) for b in self.biases]
        self.rb = [np.zeros_like(b) for b in self.biases]

        self.t = 0

    def update_params_adam(self, learning_rate, beta1=0.9, beta2=0.99, delta=1e-8):
        """ Actualiza pesos y biases usando Adam """
        self.t += 1

        for layer in range(self.num_layers):
            self.sW[layer] = beta1 * self.sW[layer] + (1 - beta1) * self.dW[layer]
            self.rW[layer] = beta2 * self.rW[layer] + (1 - beta2) * (self.dW[layer] ** 2)

            self.sb[layer] = beta1 * self.sb[layer] + (1 - beta1) * self.db[layer]
            self.rb[layer] = beta2 * self.rb[layer] + (1 - beta2) * (self.db[layer] ** 2)

            sW_hat = self.sW[layer] / (1 - beta1 ** self.t)
            rW_hat = self.rW[layer] / (1 - beta2 ** self.t)

            sb_hat = self.sb[layer] / (1 - beta1 ** self.t)
            rb_hat = self.rb[layer] / (1 - beta2 ** self.t)

            self.weights[layer] -= learning_rate * sW_hat / (np.sqrt(rW_hat) + delta)
            self.biases[layer] -= learning_rate * sb_hat / (np.sqrt(rb_hat) + delta)

    def calcular_penalizacion_l2(self, l2_lambda):
        """
        Calcula la penalización L2 que se suma a la loss.
        Solo penaliza pesos, no biases
        """
        suma_pesos = 0

        for W in self.weights:
            suma_pesos += np.sum(W ** 2)

        return 0.5 * l2_lambda * suma_pesos

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, learning_rate=0.1):
        """
        Entrena la red usando descenso por gradiente estándar

        En cada época:
        1. Forward propagation
        2. Cálculo de la loss
        3. Backpropagation
        4. Actualización de pesos y biases
        """
        historial = []

        for epoch in range(epochs):
            # 1. forward en train
            y_pred_train = self.forward(X_train)

            # 2. loss de train
            train_loss = cross_entropy(y_train, y_pred_train)

            # 3. backpropagation
            self.backward(y_train)

            # 4. actualizar los parametros
            self.update_params_sgd(learning_rate)

            # loss de validación
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = cross_entropy(y_val, y_pred_val)
            else:
                val_loss = None

            historial.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

        historial_df = pd.DataFrame(historial)

        return historial_df
    
    def fit_avanzado(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        learning_rate=0.01,

        # mejoras
        schedule=None,
        lr_final=0.0001,
        k=50,
        c=0.95,
        s=1,

        batch_size=None,

        optimizador="sgd",

        l2_lambda=0.0,

        early_stopping=False,
        patience=10
    ):

        """
        Entrenamiento avanzado para M1.

        Incluye
        - learning rate schedule lineal y exponencial
        - mini-batch SGD
        - Adam
        - regularización L2
        - early stopping
        """

        historial = []
        mejor_val_loss = np.inf
        mejor_weights = None
        mejor_biases = None
        epochs_sin_mejora = 0

        n = X_train.shape[0]

        if optimizador == "adam":
            self.inicializar_adam()

        for epoch in range(epochs):
            lr_actual = self.calcular_lr(epoch, learning_rate, schedule=schedule, lr_final=lr_final, k=k, c=c, s=s)

            # si no se pasa batch_size del mini-batch, usa todo el dataset como en GD clásico
            if batch_size is None:
                batch_size_actual = n
            else:
                batch_size_actual = batch_size

            # mezclar el dataset antes de armar los mini batches
            indices = np.random.permutation(n)
            X_tr_mezclado = X_train[indices]
            y_tr_mezclado = y_train[indices]

            # loop de batches -> recorre el dataset por partes
            for inicio in range(0, n, batch_size_actual):
                fin = inicio + batch_size_actual

                X_batch = X_tr_mezclado[inicio:fin]
                y_batch = y_tr_mezclado[inicio:fin]

                # entrena sobre un batch
                y_pred_batch = self.forward(X_batch)
                self.backward(y_batch)

                # regularizacion L2. Si l2_lambda ≤ 0, no hace nada
                if l2_lambda > 0:
                    for layer in range(self.num_layers):
                        self.dW[layer] += l2_lambda * self.weights[layer] # ∇L̃ = ∇L + λw

                if optimizador == "sgd":
                    self.update_params_sgd(lr_actual)

                elif optimizador == "adam":
                    self.update_params_adam(lr_actual)

            # registro cómo le fue al modelo en toda la época
            y_pred_train = self.forward(X_train)
            train_loss = cross_entropy(y_train, y_pred_train)

            if l2_lambda > 0:
                train_loss += self.calcular_penalizacion_l2(l2_lambda)

            if X_val is not None and y_val is not None:
                # para ver si el modelo generaliza
                y_pred_val = self.forward(X_val)
                val_loss = cross_entropy(y_val, y_pred_val)

                if l2_lambda > 0:
                    val_loss += self.calcular_penalizacion_l2(l2_lambda)
            else:
                val_loss = None

            historial.append({"epoch": epoch, "learning_rate": lr_actual, "train_loss": train_loss, "val_loss": val_loss})

            # early stopping -> mira si la loss en validation mejora
            if early_stopping and val_loss is not None:
                if val_loss < mejor_val_loss:
                    mejor_val_loss = val_loss
                    mejor_weights = [W.copy() for W in self.weights]
                    mejor_biases = [b.copy() for b in self.biases]
                    epochs_sin_mejora = 0
                else:
                    epochs_sin_mejora += 1

                # si validation loss deja de mejorar, seguir entrenando puede hacer que el modelo memorice el train.
                if epochs_sin_mejora >= patience:
                    print(f"Early stopping en epoch {epoch}")

                    self.weights = mejor_weights
                    self.biases = mejor_biases

                    break

        historial_df = pd.DataFrame(historial)

        return historial_df
