# H→µµ analysis framework

![Flake8](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/flake8.yml/badge.svg)
![Validation](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/validation.yml/badge.svg)

The framework was originally designed for the [Run 2 H→µµ search](https://inspirehep.net/literature/1815813) by CMS Collaboration. The results in the channel targeting the VBF Higgs production mode were reproduced with 1% precision.

The data processing is implemented using columnar approach, making use of the functionality introduced in [Coffea](https://github.com/CoffeaTeam/coffea) package. The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster, or a distributed `Slurm` cluster. Job distribution can also be done using `Apache Spark`, but it leads to failures when processing large datasets due to unresolved memory issues.

### Installation instructions (Purdue Hammer cluster):
```bash
git clone https://github.com/kondratyevd/hmumu-coffea
cd hmumu-coffea
module load anaconda/5.3.1-py37
conda create --name hmumu python=3.7
source activate hmumu
pip install --user coffea matplotlib==3.4.2 dask_jobqueue mplhep
conda install -c conda-forge pytest dask xrootd 
source /cvmfs/cms.cern.ch/cmsset_default.sh
mkdir dask_logs
. setup_proxy.sh
```

### Example of Dask+Slurm cluster initialization
The `ipython` session that creates the cluster should be started in a different bash session / screen.
```bash
dkondra@hammer-c000:~/hmumu-coffea $ ipython -i slurm_cluster_prep.py
Python 3.7.9 | packaged by conda-forge | (default, Dec  9 2020, 21:08:20)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.
Dask version: 2021.03.0

In [1]: cluster = SLURMCluster( project='cms', cores=1, memory='3.9GB',walltime='14-00:00:00', job_extra=['--qos=normal', '-o dask_logs/dask_job.%j.%N.out','-e dask_logs/dask_job.%j.%N.error', '--reservation=DASKTEST'])

In [2]: cluster.scale(300)

In [3]: print(cluster)
SLURMCluster(346dd8d0, 'tcp://128.211.149.133:37608', workers=92, threads=92, memory=358.80 GB)
```
The IP address (in this example `128.211.149.133:37608`) can be used to create a Dask client that would connect to this cluster. 

### Examples of Dask client initialization
Local cluster:
```python
from dask.distributed import Client
client = Client(
    processes=True,
    n_workers=40,
    threads_per_worker=1,
    memory_limit='2.9GB',
)
```
Connecting to an existing Slurm cluster:
```python
from dask.distributed import Client
client = Client(128.211.149.133:37608)
```
