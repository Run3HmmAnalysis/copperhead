## Dask client initialization
### Local cluster
```python
from dask.distributed import Client
client = Client(
    processes=True,
    n_workers=40,
    threads_per_worker=1,
    memory_limit='2.9GB',
)
```

### Slurm cluster
Slurm cluster is created from an `iPython` session which should be started in a new bash session / screen.
```
dkondra@hammer-c000:~/hmumu-coffea $ ipython -i slurm_cluster_prep.py
Python 3.7.9 | packaged by conda-forge | (default, Dec  9 2020, 21:08:20)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.
Dask version: 2021.03.0

In [1]: cluster = SLURMCluster( project='cms', cores=1, memory='3.9GB',walltime='14-00:00:00', job_extra=['--qos=normal', '-o /tmp/dask_job.%j.%N.out','-e /tmp/dask_job.%j.%N.error'])
In [2]: cluster.scale(100)
In [3]: print(cluster)
SLURMCluster(346dd8d0, 'tcp://128.211.149.133:37608', workers=92, threads=92, memory=358.80 GB)
```
The IP address (in this example `128.211.149.133:37608`) can be used to create a Dask client that would connect to this cluster:

```python
from dask.distributed import Client
client = Client(128.211.149.133:37608)
```

Number of workers in the cluster can be adjusted using `cluster.scale()`, which specifies maximum number of workers. The workers will be added as soon as the required resources (CPUs and memory) are available.

Parameters of `SLURMCluster`:
- `project` - corresponds to queue name at Purdue Tier-2
- `cores` - number of CPUs per job (1 is enough for our purposes)
- `memory` - memory limit per worker. If a job exceeds the memory limit, it gets resubmitted a few times and then fails
- `walltime` - how long the cluster will be running
- `job_extra` - corresponds to arguments passed to `srun`
    - note that location for `-o` and `-e` arguments should be an existing directory, otherwise the cluster will not start
    - if a dedicated reservation has been created by a system administrator, it can be specified using additional argument, e.g `'--reservation=TEST'`
