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
        
    def adam(self, epoch, B1=0.9, B2=0.99):
        
        return 0
    def update_params_adam(self, learning_rate):
        return 0

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
    

