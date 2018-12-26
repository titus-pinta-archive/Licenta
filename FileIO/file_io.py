import os

class Appender:
    def __init__(self, path):
        self.f = open(path, "wb")

    def append(self,  string):
        self.f.write(string)
        self.f.flush()
        os.fsync(self.f.fileno())

    def __del__(self):
        self.f.close()


def save(string, file_name):
    with open(file_name, "wb") as f:
        f.write(string)
        f.close()


def load(file_name):
    with open(file_name, "rb") as f:
        string = f.read()
        return string

