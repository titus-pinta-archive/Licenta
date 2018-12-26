# TODO documentatie

import Optimizers.gradient_descent
import Optimizers.polyak
import Optimizers.mod_polyak
import Optimizers.nesterov
import Optimizers.mod_nesterov
import Optimizers.conditions

import DataVizualizer.optimizer_animate as plt

import Data.result_data as result

import FileIO.file_io

import dill as pickle

import numpy as np
import datetime

import re


class Problem:
    def __init__(self, func, grad, x0, cond="grad", optim="GD", eps=10 ** (-5), cond_options=None, optim_options=None,
                 disp="print", ret_info="iter trace grad", disp_info="iter trace grad", x_range=np.array([-10, 10]),
                 y_range=np.array([-10, 10]), name=None, iter_print_gap=1, op_func=None, iter_stop=10 ** 6, eta=3):

        self.f = func
        self.g = grad
        self.x0 = x0
        self.eps = eps

        self.cond_options = cond_options
        self.optim_options = optim_options if optim_options is not None else {}

        self.disp_info = disp_info
        self.ret_info = ret_info

        if name is not None:
            self.name = name
        else:
            self.name = re.sub("[-.:; ]", "_", str(datetime.datetime.now()))

        self.iter_print_gap = iter_print_gap

        self.result = None

        self.cond = Optimizers.conditions.Conditions(cond, eps).cond

        self.obj_cond = False

        self.op_func = op_func

        if disp == "print":
            self.disp = None
        elif disp == "surf" or disp == "mesh" or disp == "contour":
            pf1 = plt.OptimizerAnimate(func, disp, x_range=x_range, y_range=y_range)
            self.disp = pf1.construct_display_animation()
        elif disp == "none":
            self.disp = None
            self.iter_print_gap = 0
        elif callable(disp):
            self.disp = disp
        else:
            raise ValueError("Invalid display mode")

        if optim == "GD":
            self.optim = Optimizers.gradient_descent.GradientDescent(grad, x0, self.cond,
                                                                     condition_params=cond_options,
                                                                     op_func=op_func, ret_info=ret_info,
                                                                     disp_info=disp_info, iter_print_gap=iter_print_gap,
                                                                     disp=self.disp,
                                                                     iter_stop=iter_stop, eta=eta)
        elif optim == "Polyak":
            self.optim = Optimizers.polyak.Polyak(grad, x0, self.cond, condition_params=cond_options,
                                                  op_func=op_func, ret_info=ret_info,
                                                  disp_info=disp_info, iter_print_gap=iter_print_gap,
                                                  disp=self.disp, iter_stop=iter_stop, eta=eta, **self.optim_options)
        elif optim == "ModPolyak":
            self.optim = Optimizers.mod_polyak.ModPolyak(grad, x0, self.cond, condition_params=cond_options,
                                                         op_func=op_func, ret_info=ret_info,
                                                         disp_info=disp_info, iter_print_gap=iter_print_gap,
                                                         disp=self.disp, iter_stop=iter_stop, eta=eta)
        elif optim == "Nesterov":
            self.optim = Optimizers.nesterov.Nesterov(grad, x0, self.cond,
                                                      condition_params=cond_options,
                                                      op_func=op_func, ret_info=ret_info,
                                                      disp_info=disp_info, iter_print_gap=iter_print_gap,
                                                      disp=self.disp,
                                                      iter_stop=iter_stop, eta=eta, **self.optim_options)
        elif optim == "ModNesterov":
            self.optim = Optimizers.mod_nesterov.ModNesterov(grad, x0, self.cond,
                                                             condition_params=cond_options,
                                                             op_func=op_func, ret_info=ret_info,
                                                             disp_info=disp_info, iter_print_gap=iter_print_gap,
                                                             disp=self.disp,
                                                             iter_stop=iter_stop, eta=eta)
        else:
            raise ValueError("Undefined Optimizer")

    def solve(self):
        self.optim_options = {} if self.optim_options is None else self.optim_options
        ret_list = self.optim.optimize()
        if "error" in ret_list.keys():
            print("Optimizer failed to converge in: " + str(ret_list["it"]))
            del ret_list["error"]

        self.result = result.ResultData(**ret_list)
        print("Problem solved")

    @classmethod
    def from_result(cls, r, name=""):
        ret = cls(None, None, None, name=name)
        ret.solve = None
        ret.result = r
        return ret

    @classmethod
    def from_file(cls, file_path):
        return cls.from_str(FileIO.file_io.load(file_path))

    @classmethod
    def from_str(cls, string):
        return pickle.loads(string)

    def representation(self):
        return pickle.dumps(self)

    def save_result(self, file_path=None):
        if self.result:
            file_path = file_path if file_path is not None else self.name + ".sv"
            FileIO.file_io.save(self.result.representation(), file_path)
        else:
            print("Problem not yet solved")

    def save(self, file_path=None):
        if self.result:
            file_path = file_path if file_path is not None else self.name + ".sv"
            FileIO.file_io.save(self.representation(), file_path)
        else:
            print("Problem not yet solved")
