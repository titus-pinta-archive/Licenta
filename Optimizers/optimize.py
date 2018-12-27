#
# call:
#   df(x) = gradient of f at x as numpy array
#   x0 = initial condition as numpy array
#   condition = stopping condition function (callable)
#       the condition function is called condition(x0, x1, g, **condition_parameters)
#   condition_params = parameters for the condition function, if op_func is supplied then f(x_n) is added as f=f(x_n)
#   eta = learning rate
#   gamma = momentum parameter
#   op_func = returns tuple of gradient at x and function value at same point
#   ret_info = specification for the format of the return
#   disp_info = specification for the format of the display
#       possible values for ret_info and disp_info are:
#           ""          only the approximate minimizer
#           "iter"      number of iterations
#           "trace"     all points visited by the algorithm
#           "grad"      the gradient in all points visited
#       multiple diso_info values can be combined for a combined result
#       disp_info = "iter" doesn't do anything
#   iter_print_gap = number of iterations between relevant information is displayed
#   disp = callable function to handle data display, by default it is print
#   iter_stop = number of iterations until the algorithm fails
#
#   returns the specified n-tuple (x_approx, it_number, trace, grads)
#
###################################################################################################

import numpy as np


class Optimizer:
    def __init__(self, df, x0, condition, condition_params=None, eta=3 * 10 ** (-3),
                 op_func=None, ret_info="", disp_info="", iter_print_gap=0, disp=None, iter_stop=10 ** 6):

        self.df = df
        self.x0 = x0
        self.op_function = op_func
        self.iter_stop = iter_stop

        self.ret_state = 0
        self.disp_state = 0

        self.disp = disp
        self.eta = eta

        self.ret_trace_list = []
        self.ret_grad_list = []
        self.ret_func_list = []

        self.condition = condition

        self.accuracy = False
        self.accuracy_function = None

        if "iter" in ret_info:
            self.ret_state = self.ret_state | 1

        if "trace" in ret_info:
            self.ret_state = self.ret_state | 2

        if "grad" in ret_info:
            self.ret_state = self.ret_state | 4

        if "obj" in ret_info:
            self.ret_state = self.ret_state | 8

        self.iter_print_gap = int(iter_print_gap)
        if iter_print_gap != 0:
            if "iter" in disp_info:
                self.disp_state = self.disp_state | 1

            if "trace" in disp_info:
                self.disp_state = self.disp_state | 2

            if "grad" in disp_info:
                self.disp_state = self.disp_state | 4

            if "norminf" in disp_info:
                self.disp_state = self.disp_state | 8

            if "norm2" in disp_info:
                self.disp_state = self.disp_state | 16

            if "obj" in disp_info:
                self.disp_state = self.disp_state | 32

            if "acc" in disp_info:
                self.disp_state = self.disp_state | 64
                self.accuracy = True
                self.accuracy_function = condition_params["acc"]

        if condition_params is None:
            self.condition_params = {}
        else:
            self.condition_params = condition_params

    def print_step(self, it_number, x=None, g=None, f=None, acc=None):
        disp_str = ""
        disp_dict = {}

        if self.disp_state & 1:
            disp_str = str(it_number) + " iteration: \n"
            disp_dict["it"] = it_number

        if self.disp_state & 2:
            disp_str = disp_str + "    position: " + str(x) + "\n"
            disp_dict["x"] = x

        if self.disp_state & 4:
            disp_str = disp_str + "    gradient: " + str(g) + "\n"
            disp_dict["g"] = g

        if self.disp_state & 8:
            disp_str = disp_str + "    gradient inf norm: " + str(np.linalg.norm(g, np.inf)) + "\n"
            disp_dict["ginf"] = np.linalg.norm(g, np.inf)

        if self.disp_state & 16:
            disp_str = disp_str + "    gradient 2 norm: " + str(np.linalg.norm(g)) + "\n"
            disp_dict["gl2"] = np.linalg.norm(g)

        if self.disp_state & 32:
            disp_str = disp_str + "    function: " + str(f) + "\n"
            disp_dict["f"] = f

        if self.disp_state & 64:
            disp_str = disp_str + "    accuracy: " + str(acc) + "\n"
            disp_dict["acc"] = acc

        if it_number % self.iter_print_gap == 0:
            if self.disp is None:
                print(disp_str)
            else:
                self.disp(**disp_dict)

    def add_to_return(self, x=None, g=None, f=None):
        if self.ret_state & 2:
            self.ret_trace_list.append(x)

        if self.ret_state & 4:
            self.ret_grad_list.append(g)

        if self.ret_state & 8:
            self.ret_func_list.append(f)

    def construct_return_dictionary(self, it, x):
        ret_dict = {"x_min": x}
        if self.ret_state & 1:
            ret_dict["it"] = it

        if self.ret_state & 2:
            ret_dict["x"] = self.ret_trace_list

        if self.ret_state & 4:
            ret_dict["g"] = self.ret_grad_list

        if self.ret_state & 8:
            ret_dict["f"] = self.ret_func_list

        return ret_dict
