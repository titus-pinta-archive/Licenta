import MNIST.mnist_loader
import MachineLearning.network
import MachineLearning.network_problem

import numpy as np


class MNISTProblem:

    def __init__(self, size_train_data=0, hidden_layers=[100], weights=None, biases=None, cond="gradinf", optim="GD",
                 eps=10 ** (-5), cond_options=None, optim_options=None, disp="print", name=None, ret_info="iter",
                 disp_info="iter norminf", iter_print_gap=1, test_accuracy=0, iter_stop=2500, eta=2,
                 activation=MachineLearning.network.sigma_activation, net=None,
                 cost=MachineLearning.network.cross_entropy_cost, l2_regularization=0):
        self.size_train_data = size_train_data

        self.x_tdata, self.y_tdata = MNIST.mnist_loader.load_train_data(size_train_data)
        self.name = name

        self.layout = [784] + hidden_layers + [10]
        self.net = MachineLearning.network.Network(self.layout, activation=activation, cost=cost, name=name,
                                                   l2_regularization=l2_regularization) if net is None else net

        if "acc" == cond:
            if cond_options is not None:
                cond_options["acc"] = lambda x: self.accuracy_validation(0, *self.net.from_vector(x))
            else:
                cond_options = {"acc": lambda x: self.accuracy_validation(0, *self.net.from_vector(x))}

        self.problem = MachineLearning.network_problem.NetworkProblem(self.net, self.x_tdata, self.y_tdata, weights,
                                                                      biases, cond=cond, optim=optim, eps=eps,
                                                                      cond_options=cond_options,
                                                                      optim_options=optim_options, disp=disp,
                                                                      disp_info=disp_info, ret_info=ret_info,
                                                                      iter_print_gap=iter_print_gap,
                                                                      iter_stop=iter_stop, eta=eta)

        self.test_accuracy = test_accuracy
        self.x_test_data, self.y_test_data = None, None

    def get_hipper_params(self):
        return self.net.weights, self.net.biases, self.layout

    def accuracy_validation(self, test_accuracy=0, weights=None, biases=None):

        if self.x_test_data is None and self.y_test_data is None:
            self.x_test_data, self.y_test_data = MNIST.mnist_loader.load_test_data(test_accuracy)

        activation = self.net.feed_forward(self.x_test_data, weights=weights, biases=biases)
        accuracy = 0
        for a, y in zip(activation.T, self.y_test_data.T):
            if np.argmax(a) == np.argmax(y):
                accuracy = accuracy + 1

        return accuracy / len(self.y_test_data.T)

    def optimize(self):
        self.problem.optimize()

        if self.test_accuracy or self.test_accuracy == 0:
            return self.accuracy_validation(self.test_accuracy)

    def save(self):
        self.problem.save()

    def save_network(self):
        self.problem.save_network()
