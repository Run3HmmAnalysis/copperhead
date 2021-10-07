import pytest
import asyncio
import dask
from dask.distributed import Client
from dask.distributed import Scheduler, Worker
from dask_jobqueue import SLURMCluster
from coffea.processor import dask_executor
from python.processor import DimuonProcessor

dask.config.set({"temporary-directory": "/tmp/dask-temp/"})
dask.config.set({"distributed.worker.timeouts.connect": "60s"})

__all__ = [
    "pytest",
    "asyncio",
    "dask",
    "Client",
    "Scheduler",
    "Worker",
    "SLURMCluster",
    "dask_executor",
    "DimuonProcessor",
]

print("Dask version:", dask.__version__)


async def f(scheduler_address):
    r = await Worker(
        scheduler_address,
        resources={"processor": 0, "reducer": 1},
        ncores=1,
        nthreads=1,
        memory_limit="64GB",
    )
    await r.finished()
