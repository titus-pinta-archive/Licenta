import matplotlib.pyplot as plt
import numpy as np


def plot_grad(result, style="r"):

    if result.data["it"] is not None:
        it = result.data["it"]
    else:
        print("Result has no it")
        return

    if result.data["grad"] is not None:
        grad = np.array([np.linalg.norm(g) for g in result.data["grad"]])

    else:
        print("Result has no grad")
        return

    print(it)
    print(grad)
    plt.plot(np.array(range(it + 1)), grad, style)
    plt.show()
