import itertools
from functools import partial


def argset_product(arglist):
    return [
        dict(zip(arglist.keys(), values))
        for values in itertools.product(*arglist.values())
    ]


def parallelize(
    func,
    argsets,
    client,
    parameters={},
    timer=None,
):
    """
    arg_exec is a dictionary of possible values of arguments for func_exec.
    All combinations of parameters are computed, and func_exec is executed for each combination.

    """

    argsets = argset_product(argsets)
    futures = client.map(partial(func, parameters=parameters), argsets)
    results = client.gather(futures)

    return results
