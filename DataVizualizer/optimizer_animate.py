# 11/4/2018
#
# A class that allows vizualization of the steps of gradient optimizers
# Constructor:
#   func = function to minimize
#   plot_type = type of plot
#       possible values for type:
#           "contour"   2D contour plot
#           "surf"      3D surface plot
#           "mesh"      3D mesh plot
#
#
#
#
# TODO documentatia de utilizare
#
#
#
#
#
#   throws exception if function is not f:R^2 -> R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class OptimizerAnimate:

    def __init__(self, func, plot_type="contour", x_range=np.array([-1, 1]), y_range=np.array([-1, 1]), x_density=50,
                 y_density=50, contour_density=50, plt_title="", x_label="", y_label="", z_label=""):
        # initialize data
        self._func = func

        self._dim = 2

        self._x_range = x_range
        self._y_range = y_range

        self._x_density = x_density
        self._y_density = y_density

        x, y = np.meshgrid(np.linspace(x_range[0], x_range[1], x_density),
                           np.linspace(y_range[0], y_range[1], y_density))
        z = func([x, y])

        self.ax = None

        if "contour" == plot_type:
            self._dim = 2

            plt.figure()
            self.ax = plt.axes()
            self.ax.contour(x, y, z, contour_density, cmap=plt.cm.jet)

        elif "mesh" == plot_type or "surf" == plot_type:
            self._dim = 3

            plt.figure()
            self.ax = plt.axes(projection="3d")
            if "mesh" == plot_type:
                self.ax.plot_wireframe(x, y, z, cmap=plt.cm.jet)
            elif "surf" == plot_type:
                self.ax.plot_surface(x, y, z, cmap=plt.cm.jet)

        if plt_title:
            plt.title(plt_title)

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if z_label:
            plt.xlabel(z_label)

        plt.ion()
        plt.show()
        plt.pause(0.001)

        self._p_ant = None

    def construct_display_animation(self, style="r*"):

        def disp(it=None, x=None, g=None):

            disp_str = ""

            if it is not None:
                disp_str = str(it) + " iteration: \n"

            if x is not None:
                disp_str = disp_str + "    position: " + str(x) + "\n"

            if g is not None:
                disp_str = disp_str + "    gradient: " + str(g) + "\n"

            print(disp_str)

            if self._p_ant is not None:
                plt.setp(self._p_ant, markersize=2)

            if self._dim == 2:
                self._p_ant = self.ax.plot(*x, style, markersize=10)
            else:
                self._p_ant = self.ax.plot(np.array([x[0]]), np.array([x[1]]), np.array([self._func(x)]), style,
                                           markersize=10)

            plt.show()
            plt.pause(0.001)

        return disp
