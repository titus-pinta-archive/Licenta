import FileIO.file_io


import dill as pickle


class DisplayFile:

    def __init__(self, path, print_to_screen_info="iter"):
        self.append = FileIO.file_io.Appender(path)

        def write(it=None, g=None, x=None, ginf=None, gl2=None, f=None, acc=None):

            _dict = {"g": g, "x": x, "f": f}
            self.append.append(pickle.dumps(_dict) + bytearray("\n\n --NEW ENTRY-- \n\n", "ascii"))

            disp_str = ""
            if "iter" in print_to_screen_info and it is not None:
                disp_str = str(it) + " iteration: \n"

            if "trace" in print_to_screen_info and x is not None:
                disp_str = disp_str + "    position: " + str(x) + "\n"

            if "grad" in print_to_screen_info and g is not None:
                disp_str = disp_str + "    gradient: " + str(g) + "\n"

            if "norminf" in print_to_screen_info and ginf is not None:
                disp_str = disp_str + "    gradient inf norm: " + str(ginf) + "\n"

            if "norm2" in print_to_screen_info and gl2 is not None:
                disp_str = disp_str + "    gradient 2 norm: " + str(gl2) + "\n"

            if "obj" in print_to_screen_info and f is not None:
                disp_str = disp_str + "    function: " + str(f) + "\n"

            if "acc" in print_to_screen_info and acc is not None:
                disp_str = disp_str + "    accuracy: " + str(acc) + "\n"

            if disp_str:
                print(disp_str)

        self.disp = write
