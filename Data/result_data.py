# 11/4/2018
#
# result data
#
# A class that encapsulates the output of a optimization problem
# The data inside can be converted to a string via str()
# Conversion from string is possible via from_str()
#
# call:
#   x: minimum point
#   it: number of iterations before convergence
#   grad: vector with gradient values
#   trace: vector of points visited by the algorithm
#
###################################################################################################

import FileIO.file_io

import dill as pickle

class ResultData:

    def __init__(self, x_min=None, it=None, g=None, x=None, f=None):
        self.data = {"x_min": x_min}

        if it is not None:
            self.data["it"] = it

        if g is not None:
            self.data["g"] = g

        if x is not None:
            self.data["x"] = x

        if f is not None:
            self.data["f"] = f

    @classmethod
    def from_str(cls, string):
        return pickle.loads(string)

    @classmethod
    def from_file(cls, path):
        return cls.from_str(FileIO.file_io.load(path))

    @classmethod
    def from_trace_file(cls, path):
        string = FileIO.file_io.load(path)
        if not string:
            raise ValueError("Empty file!")

        data = {"g": [], "x": []}
        for entry in string.split(bytearray("\n\n --NEW ENTRY-- \n\n", "ascii")):
            if entry:
                _dict = pickle.loads(entry)

                if "x" in _dict.keys():
                    data["x"].append(_dict["x"])

                if "g" in _dict.keys():
                    data["g"].append(_dict["g"])

        iterations = 0
        x0 = None
        if "x" in data.keys():
            iterations = len(data["x"])
            x0 = data["x"][-1]
        elif "g" in data.keys():
            iterations = len(data["g"])

        return cls(**data, it=iterations, x_min=x0)

    def representation(self):
        return pickle.dumps(self)
