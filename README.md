# H→µµ analysis framework

![Flake8](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/flake8.yml/badge.svg)
![Validation](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/validation.yml/badge.svg)

The framework was originally designed for the [Run 2 H→µµ search](https://inspirehep.net/literature/1815813) by the CMS Collaboration. The results in the channel targeting the VBF Higgs production mode were reproduced with 1% precision.

The data processing is implemented via [columnar approach](https://indico.cern.ch/event/759388/contributions/3306852/attachments/1816027/2968106/ncsmith-how2019-columnar.pdf), making use of the tools implemented in [Coffea](https://github.com/CoffeaTeam/coffea) package. The inputs to the framework are datasets in `NanoAOD` format; there is a separate simplified workflow for `Delphes` datasets (work in progress).

The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster, or a distributed `Slurm` cluster. Job distribution can also be done using [Apache Spark](https://github.com/apache/spark), but it leads to failures when processing large datasets due to unresolved memory issues.

### Installation instructions (Purdue Hammer cluster):
```bash
module load anaconda/5.3.1-py37
conda create --name hmumu python=3.7
source activate hmumu
conda install -c conda-forge pytest dask xrootd
pip install --user coffea matplotlib==3.4.2 dask_jobqueue mplhep
source /cvmfs/cms.cern.ch/cmsset_default.sh

git clone https://github.com/kondratyevd/hmumu-coffea
cd hmumu-coffea
. setup_proxy.sh
```

### Test run
Should take around 1.5 minutes to complete.
```bash
python3 -W ignore tests/test_submit.py
```

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
Connecting to an existing Slurm cluster (see next section about where to get the IP address):
```python
from dask.distributed import Client
client = Client(128.211.149.133:37608)
```

### Example of Dask+Slurm cluster initialization
The `iPython` session that creates the cluster should be started in a new bash session / screen.
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
The IP address (in this example `128.211.149.133:37608`) can be used to create a Dask client that would connect to this cluster. 

Number of workers in the cluster can be adjusted using `cluster.scale()`, which specifies maximum number of workers. The workers will be added as soon as the required resources (CPUs and memory) are available.

Parameters of `SLURMCluster`:
- `project` - corresponds to queue name at Purdue Tier-2
- `cores` - number of CPUs per job (1 is enough for our purposes)
- `memory` - memory limit per worker. If a job exceeds the memory limit, it gets resubmitted a few times and then fails
- `walltime` - how long the cluster will be running
- `job_extra` - corresponds to arguments passed to `srun`
    - note that location for `-o` and `-e` arguments should be an existing directory, otherwise the cluster will not start
    - if a dedicated reservation has been created by a system administrator, it can be specified using additional argument, e.g `'--reservation=TEST'`
