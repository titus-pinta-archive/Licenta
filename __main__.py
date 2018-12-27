import MNIST.mnist_problem
import MNIST.mnist_loader
import DataVizualizer.disp_to_file
import Data.result_data
import TestProblems.problem
import MachineLearning.network
import Optimizers.gradient_descent

import numpy as np

# p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="GD", eta=0.003)
# p.solve()

# p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="Polyak", eta=0.003)
# p.solve()

# p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="ModPolyak", eta=0.003)
# p.solve()

# p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="Nesterov", eta=0.003)
# p.solve()

# p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="ModNesterov", eta=0.003)
# p.solve()

# print(p.result.data["x_min"])
# print(p.result.data["it"])


appender = DataVizualizer.disp_to_file.DisplayFile("nesterov.sv", print_to_screen_info="iter acc norminf")

mpb = MNIST.mnist_problem.MNISTProblem(eta=0.5, eps=0.95, iter_stop=5000, l2_regularization=0.1,
                                       optim="ModNesterov", name="nesterov", disp=appender.disp,
                                       disp_info="iter trace grad norminf obj acc", cond="acc")

print(mpb.optimize())
mpb.save_network()
appender = None

# t = TestProblems.problem.Problem.from_file("nesterov.sv")
# print(t.result.data)

# TODO l1 regularization
# TODO Restart nesterov 3 metode
# TODO under damped nesterov
# TODO adaptative dampping
# TODO start from small constant nesterov
# TODO beta n n / n+ 3 la alg cristi
# TODO foloseste iterate mai des sa nu uiti
