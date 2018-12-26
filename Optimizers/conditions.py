# 11/4/2018
#
# A class for the different stopping conditions
# TODO de terminat documentatia
#
###################################################################################################

import numpy as np


class Conditions:

    def __init__(self, cond="grad", eps=10 ** (-5)):
        if cond == "grad":
            def grad_cond(_, __, g):
                return np.linalg.norm(g) < eps

            self.cond = grad_cond

        elif cond == "gradinf":
            def grad_cond_inf(_, __, g):
                return np.linalg.norm(g, np.inf) < eps

            self.cond = grad_cond_inf

        elif cond == "obj":
            self._f_ant = None

            def obj_cond(_, __, ___, f):
                if self._f_ant:
                    _aux = np.abs(f - self._f_ant) < eps
                    self._f_ant = f
                    return _aux
                else:
                    self._f_ant = f
                    return False

            self.cond = obj_cond

        elif cond == "x":
            def x_cond(x0, x1, _):
                return np.linalg.norm(x0 - x1) < eps

            self.cond = x_cond

        elif cond == "xinf":
            def xinf_cond(x0, x1, _):
                return np.linalg.norm(x0 - x1, np.inf) < eps

            self.cond = xinf_cond

        else:
            raise ValueError("Invalid condition type")
