import itertools
from functools import partial


def parallelize(func, argset, client, parameters={}, timer=None):
    """
    `argset` is a dictionary of possible values of arguments for `func`.
    All combinations of parameters are computed, and `func` is executed for each combination.

    """

    argset = [
        dict(zip(argset.keys(), values))
        for values in itertools.product(*argset.values())
    ]
    futures = client.map(partial(func, parameters=parameters), argset)
    results = client.gather(futures)

    return results
