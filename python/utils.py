import os


def almost_equal(a, b):
    return abs(a - b) < 10e-6


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass
