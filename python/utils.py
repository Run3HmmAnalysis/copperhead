import os


def almost_equal(a, b, precision=10e-6):
    return abs(a - b) < precision


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass
