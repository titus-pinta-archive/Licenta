import MNIST.mnist_problem
import MNIST.mnist_loader
import DataVizualizer.disp_to_file
import Data.result_data
import TestProblems.problem
import MachineLearning.network
import Optimizers.gradient_descent

import numpy as np
from PIL import Image


#appender = DataVizualizer.disp_to_file.DisplayFile("nesterov.sv", print_to_screen_info="iter acc norminf")

mpb = MNIST.mnist_problem.MNISTProblem(eta=0.5, eps=0.75, iter_stop=250, l2_regularization=0.05,
                                       optim="ModNesterov", name="nesterov", disp="print",
                                       disp_info="iter  norminf acc", cond="acc")

print(mpb.optimize())
mpb.save_network()
#appender = None

#n = MachineLearning.network.Network.load_from_file("nesterov.netsv")
n = mpb.net

img = Image.open("Tests/test0.png")
arr = np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255
print(0, np.argmax(n.feed_forward(arr)))

img = Image.open("Tests/test1.png")
print(1, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test2.png")
print(2, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test3.png")
print(3, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test4.png")
print(4, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test5.png")
print(5, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test6.png")
print(6, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test7.png")
print(7, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test8.png")
print(8, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

img = Image.open("Tests/test9.png")
print(9, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((3, 28, 28))[0].reshape(784, 1) / 255)))

ti, tl = MNIST.mnist_loader.load_test_data()
img = Image.fromarray(np.array(ti.T[:][0].reshape((28, 28)) * 255, np.uint8))
#img.save("train.png")
#img.show()
print(7, np.argmax(n.feed_forward(np.array(img.getdata(), np.uint8).reshape((1, 28, 28))[0].reshape(784, 1) / 255)))
print(7, np.argmax(n.feed_forward(ti[:, 0].reshape((784, 1)))))


arr1 = np.array(img.getdata(), np.uint8).reshape((1, 28, 28))[0].reshape(784, 1) / 255
arr2 = ti[:, 0].reshape((784, 1))
print(np.max(np.abs(arr1 - arr2)))



# t = TestProblems.problem.Problem.from_file("nesterov.sv")
# print(t.result.data)

# TODO early stopping
# TODO Daemon si python daemon scripting language
# TODO vezi varianta cu daemon manager si cu salvare la inchidere
# TODO defineste gradul de incredere

# TODO l1 regularization
# TODO Restart nesterov 3 metode
# TODO under damped nesterov
# TODO adaptative dampping
# TODO start from small constant nesterov
# TODO beta n n / n+ 3 la alg cristi
# TODO foloseste iterate mai des sa nu uiti
