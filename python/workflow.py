import itertools
from functools import partial


def parallelize(func, argset, client, parameters={}, seq=False):
    """
    `func`: the function to be executed in parallel.
    `argset`: a dictionary that contains a list of possible values for each of the `func` arguments.
    If there is only one possible value for some argument, pass it as a list with a single element.
    All combinations of arguments will be computed, and `func` will be executed for each combination.
    `client`: Dask client connected to a cluster of workers.
    `parameters`: global parameters shared by different methods across the framework, almost every
    function needs some of them.

    Set `seq=True` to force sequential processing for debugging.

    """

    argset = [
        dict(zip(argset.keys(), values))
        for values in itertools.product(*argset.values())
    ]
    if seq:
        results = []
        for args in argset:
            results.append(func(args, parameters))
    else:
        map_futures = client.scatter(argset)
        futures = client.map(partial(func, parameters=parameters), map_futures)
        results = client.gather(futures)

    return results
