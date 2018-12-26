import FileIO.file_io

import numpy as np
import datetime
import dill as pickle

import re


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def l2_cost_derivative(output_activations, y):
    return output_activations - y


def l2_cost_function(output_activations, y):
    return np.linalg.norm(output_activations - y) ** 2 / (2 * y.shape[1])


class Activation:
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative


class Cost:
    def __init__(self, cost_function, cost_derivative):
        self.cost_function = cost_function
        self.cost_derivative = cost_derivative


sigma_activation = Activation(sigmoid, sigmoid_prime)
l2_cost = Cost(l2_cost_function, l2_cost_derivative)


class Network(object):

    def __init__(self, sizes, activation=sigma_activation, cost=l2_cost, name=None):
        self.activation = activation.activation_function
        self.activation_derivative = activation.activation_derivative
        self.cost_derivative = cost.cost_derivative
        self.cost_function = cost.cost_function
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights, self.biases = self.random_weights_and_biases()

        self.name = name if name else re.sub("[-.:; ]", "_", str(datetime.datetime.now()))

    def random_weights_and_biases(self):
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return weights, biases

    def feed_forward(self, x, weights=None, biases=None):
        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases
        for w, b in zip(weights, biases):
            x = self.activation(w @ x + b)
        return x

    def loss_function(self, x, y, weights=None, biases=None):
        return self.cost_function(self.feed_forward(x, weights, biases), y)

    def back_propagation(self, x, y, weights=None, biases=None):
        n = y.shape[1]

        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases

        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        activations = [x]
        zs = []
        for w, b in zip(weights, biases):
            zs.append(w @ activations[-1] + b)
            activations.append(self.activation(zs[-1]))

        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(zs[-1])

        nabla_b[-1] = np.sum(delta, 1).reshape(nabla_b[-1].shape) / n
        nabla_w[-1] = (delta @ activations[-2].T) / n

        for l in range(2, self.num_layers):
            delta = (weights[-l + 1].T @ delta) * self.activation_derivative(zs[-l])

            nabla_b[-l] = np.sum(delta, 1).reshape(nabla_b[-l].shape) / n
            nabla_w[-l] = (delta @ activations[-l - 1].T) / n

        return self.cost_function(activations[-1], y), nabla_w, nabla_b

    def save_to_file(self, file_path=None):
        file_path = file_path if file_path is not None else self.name + ".netsv"
        FileIO.file_io.save(self.representation(), file_path)

    @classmethod
    def load_from_file(cls, file_path):
        return cls.from_str(FileIO.file_io.load(file_path))

    @classmethod
    def from_str(cls, string):
        return pickle.loads(string)

    def representation(self):
        return pickle.dumps(self)
