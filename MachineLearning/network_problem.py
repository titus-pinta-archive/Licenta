import numpy as np
import TestProblems.problem


class NetworkProblem:

    def __init__(self, net, x_tdata, y_tdata, weights=None, biases=None, cond="gradinf", optim="GD", eps=10 ** (-5),
                 cond_options=None, optim_options=None, disp="print", name=None, disp_info="iter norminf",
                 ret_info="iter", iter_print_gap=1, iter_stop=2500, eta=2):

        self.net = net
        self.x_tdata = x_tdata
        self.y_tdata = y_tdata

        self.obj_func = self.construct_obj_function_handler()

        if weights is not None:
            net.weights = weights
        if biases is not None:
            net.biases = biases

        start_x = self.to_vector()

        self.problem = TestProblems.problem.Problem(None, None, start_x, optim=optim, cond=cond, name=name, eps=eps,
                                                    optim_options=optim_options, cond_options=cond_options, disp=disp,
                                                    op_func=self.obj_func, disp_info=disp_info, ret_info=ret_info,
                                                    iter_print_gap=iter_print_gap, iter_stop=iter_stop, eta=eta)

        self.name = self.net.name if self.net.name is not None else self.problem.name
        self.net.name = self.name

    def from_vector(self, x):
        return self.net.from_vector(x)

    def to_vector(self, weights=None, biases=None):
        return self.net.to_vector(weights=weights, biases=biases)

    def construct_obj_function_handler(self):
        def obj_func(x):
            f, w, b = self.net.back_propagation(self.x_tdata, self.y_tdata, *self.from_vector(x))
            return f, self.to_vector(w, b)

        return obj_func

    def optimize(self):
        self.problem.solve()
        self.net.weights, self.net.biases = self.from_vector(self.problem.result.data["x_min"])

    def save(self):
        if not self.net.name:
            self.net.name = self.name

        self.net.save_to_file()

        if not self.problem.name:
            self.problem.name = self.name

        self.problem.save()

    def save_network(self):
        self.net.save_to_file()