from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import pytest
from coffea.processor.executor import dask_executor
import dask
dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})
dask.config.set({'distributed.worker.timeouts.connect': '60s'})
from python.dimuon_processor import DimuonProcessor

import asyncio
from dask.distributed import Scheduler, Worker

async def f(scheduler_address):    
    r = await Worker(scheduler_address, resources={'processor': 0, 'reducer': 1}, ncores=1, nthreads=1, memory_limit='64GB')
    await r.finished()




