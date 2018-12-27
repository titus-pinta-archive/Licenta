import MNIST.mnist_problem
import MNIST.mnist_loader
import DataVizualizer.disp_to_file
import Data.result_data
import TestProblems.problem
import MachineLearning.network
import Optimizers.gradient_descent

import numpy as np

#p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="GD", eta=0.003)
#p.solve()

#p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="Polyak", eta=0.003)
#p.solve()

#p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="ModPolyak", eta=0.003)
#p.solve()

#p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="Nesterov", eta=0.003)
#p.solve()

#p = TestProblems.problem.Problem(lambda x: 0.5 * x ** 2, lambda x: x, 1, optim="ModNesterov", eta=0.003)
#p.solve()

#print(p.result.data["x_min"])
#print(p.result.data["it"])




appender = DataVizualizer.disp_to_file.DisplayFile("nesterov.sv", print_to_screen_info="iter acc norminf")


#n = MachineLearning.network.Network.load_from_file("start_net.netsv")
# n.name = "nesterov"
# mpb = MNIST.mnist_problem.MNISTProblem(eta=0.3, eps=0.7, optim_options={"gamma": 0.9}, iter_stop=500,
#                                        optim="ModNesterov", name="nesterov", disp=appender.disp,
#                                        disp_info="iter trace grad norminf obj acc", cond="acc")
#
# print(mpb.optimize())
# mpb.save_network()
# appender = None
#t = TestProblems.problem.Problem.from_file("nesterov.sv")
#print(t.result.data)

ti, tl = MNIST.mnist_loader.load_test_data()

n = MachineLearning.network.Network([784, 100, 10], cost=MachineLearning.network.cross_entropy_cost)
w, b = n.random_weights_and_biases()


_, nw, nb = n.back_propagation(ti, tl, weights=w, biases=b)

for i in range(len(w)):
    for j in range(len(w[i])):
        for k in range(len(w[i][j])):
            w[i][j][k] = w[i][j][k] + 0.000001
            aux1 = n.loss_function(ti, tl, weights=w, biases=b)
            w[i][j][k] = w[i][j][k] - 0.000002
            aux2 = n.loss_function(ti, tl, weights=w, biases=b)
            w[i][j][k] = w[i][j][k] + 0.000001
            print(nw[i][j][k], (aux1 - aux2) / 0.000002, nw[i][j][k] - (aux1 - aux2) / 0.000002)
            if np.abs(nw[i][j][k] - (aux1 - aux2) / 0.000002) > 10**(-5):
                print(i, j, k)
                raise ValueError("Nu mere!")


print(n.loss_function(ti, tl))
print(n.back_propagation(ti, tl))


# TODO Cross entropy
# TODO l2 regularization
# TODO Redo all optimizers with new stuff
# TODO Restart nesterov 3 metode
# TODO under damped nesterov
# TODO adaptative dampping
# TODO start from small constant nesterov
# TODO beta n n / n+ 3 la alg cristi
# TODO foloseste iterate mai des sa nu uiti
