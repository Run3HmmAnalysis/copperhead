# H→µµ analysis framework

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Nanoaod](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/nanoaod.yml/badge.svg)
![Delphes](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/delphes.yml/badge.svg)

The framework was originally designed for the [Run 2 H→µµ search](https://inspirehep.net/literature/1815813) by the CMS Collaboration. The results in the channel targeting the VBF Higgs production mode were reproduced with 1% precision.

The data processing is implemented via [columnar approach](https://indico.cern.ch/event/759388/contributions/3306852/attachments/1816027/2968106/ncsmith-how2019-columnar.pdf), making use of the tools implemented in [Coffea](https://github.com/CoffeaTeam/coffea) package. The inputs to the framework can be in `NanoAOD` format (for LHC Run 2 and Run 3 analyses), or in `Delphes` format (for HL-LHC predictions).

The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster, or a distributed `Slurm` cluster. Job distribution can also be done using [Apache Spark](https://github.com/apache/spark), but it leads to failures when processing large datasets due to unresolved memory issues.

### Installation instructions (tested at Purdue Hammer cluster):
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
Should take around 1.5-2 minutes to complete.
```bash
python3 -W ignore tests/test_nanoaod_continuous.py
```

### Dask client initialization
The framework can be used either via a local Dask client (using CPUs on the same mode where the job is launched), or using a Slurm cluster initialized over multiple computing nodes. The instructions for both modes can be found [here](docs/dask_client.md).
