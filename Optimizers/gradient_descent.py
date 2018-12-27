import Optimizers.optimize


class GradientDescent(Optimizers.optimize.Optimizer):

    def __init__(self, df, x0, condition, condition_params=None, eta=3 * 10 ** (-3),
                 op_func=None, ret_info="", disp_info="", iter_print_gap=0, disp=None, iter_stop=10 ** 6):
        super().__init__(df, x0, condition, condition_params, eta, op_func, ret_info, disp_info,
                         iter_print_gap, disp, iter_stop)

    def optimize(self):
        df = self.df
        x = self.x0
        op_func = self.op_function
        iter_stop = self.iter_stop
        iter_print_gap = self.iter_print_gap

        eta = self.eta

        accuracy = self.accuracy
        acc_function = self.accuracy_function

        condition_params = self.condition_params
        condition = self.condition

        if op_func:
            f, g = op_func(x)
            if condition == "obj":
                condition_params["f"] = f
        else:
            f = None
            g = df(x)

        if accuracy:
            acc = acc_function(x)
            condition_params["acc"] = acc

        # return initial conditions
        self.add_to_return(x=x, g=g, f=f)

        # display initial conditions
        if iter_print_gap:
            self.print_step(0, x=x, g=g, f=f, acc=acc)

        # update position for first iteration
        it_number = 1
        x_ant = x
        x = x - eta * g

        # display
        if iter_print_gap:
            self.print_step(1, x=x, g=g, f=f, acc=acc)
        self.add_to_return(x=x, g=g, f=f)

        # loop until the condition is met
        while not condition(x_ant, x, g, **condition_params):
            it_number = it_number + 1

            if op_func:
                f, g = op_func(x)
                if condition == "obj":
                    condition_params["f"] = f
            else:
                g = df(x)

            if accuracy:
                acc = acc_function(x)
                condition_params["acc"] = acc

            # update position
            x_ant = x
            x = x - eta * g

            # return info
            self.add_to_return(x=x, g=g, f=f)

            # display
            if iter_print_gap:
                self.print_step(it_number, x=x, g=g, f=f, acc=acc)

            if it_number == iter_stop:
                return {"error": True, "it": it_number, "x_min": x}

        return self.construct_return_dictionary(it_number, x)
