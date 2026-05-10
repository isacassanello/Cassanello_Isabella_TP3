import numpy as np
from activations import relu, softmax

class MLP:
    def __init__(self, layers, random_state=42, kappa=0.01):
        np.random.seed(random_state)
        self.layers = layers
        self.num_layers = len(layers) - 1

        self.weights = []
        self.biases = []

        # inicializacion de pesos y biases
        for neurona in range(self.num_layers):
            input_size = layers[neurona]
            output_size = layers[neurona + 1]

            # inicialización He para los pesos
            W = np.random.normal(loc=0, scale=np.sqrt(2 / input_size), size=(input_size, output_size))

            # biases pequeños y positivos
            b = np.random.uniform(0, kappa**2, size=(1, output_size))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """ Realiza forward propagation """

        self.activaciones = [X]
        self.pre_activaciones = []

        Z = X

        # capas ocultas
        for neurona in range(self.num_layers - 1):
            A = Z @ self.weights[neurona] + self.biases[neurona]
            Z = relu(A)

            self.pre_activaciones.append(A)
            self.activaciones.append(Z)

        # capa de salida
        A = Z @ self.weights[-1] + self.biases[-1]
        Y_hat = softmax(A)

        self.pre_activaciones.append(A)
        self.activaciones.append(Y_hat)   

        return Y_hat 


