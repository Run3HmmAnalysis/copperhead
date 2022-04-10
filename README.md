# H→µµ analysis framework

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI/CD](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/ci.yml/badge.svg)

The framework was originally designed for the [search for the Higgs boson decays into two muons](https://inspirehep.net/literature/1815813) with the CMS detector at the Large Hadron Collider. The published results in the channel targeting the VBF Higgs production mode were reproduced with 1% precision.

Currently the framework is under development to integrate both of the main Higgs production modes (ggH and VBF), and predict the sensitivity of the CMS H→µµ analysis in LHC Run 3 and at HL-LHC.

## Data formats, packages and tools used in the implementation
The data processing is implemented via [columnar approach](https://indico.cern.ch/event/759388/contributions/3306852/attachments/1816027/2968106/ncsmith-how2019-columnar.pdf), making use of the tools provided by [coffea](https://github.com/CoffeaTeam/coffea) package. The inputs to the framework should be in `NanoAOD` format.

- The first stage of the processing includes event and object selection, application of corrections, and construction of new variables. The data columns are handled via coffea's `NanoEvents` format which relies on *jagged arrays* implemented in [Awkward Array](https://github.com/scikit-hep/awkward-1.0) package. After event selection, the jagged arrays are converted to flat [pandas](https://github.com/pandas-dev/pandas) dataframes and saved as [Apache Parquet](https://github.com/apache/parquet-format) files.
- The second stage of the processing (WIP) contains / will contain training and evaluation of MVA methods (boosted decision trees, deep neural networks), as well as event categorization. The second stage mainly relies on Pandas dataframes, and uses [scikit-hep/hist](https://github.com/scikit-hep/hist) for histogramming.
- The third stage (WIP) contains / will contain plotting, parametric fits, preparation of datacards for statistical analysis. The plotting is done via [scikit-hep/mplhep](https://github.com/scikit-hep/mplhep).

The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster, or a distributed `Slurm` cluster.

## Installation instructions
Work from a `conda` environment to avoid version conflicts:
```bash
module load anaconda/5.3.1-py37
conda create --name hmumu python=3.7
source activate hmumu
```
Installation:
```bash
git clone https://github.com/kondratyevd/hmumu-coffea
cd hmumu-coffea
python3 -m pip install --user --upgrade -r requirements.txt
```
If accessing datasets via `xRootD` will be needed:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
. setup_proxy.sh
```

## Test run
Running full analysis workflow on a single input file.

**NanoAOD workflow** (should take around 1.5-2 minutes to complete):
```bash
python3 -W ignore tests/test_continuous.py
```

## Dask client initialization
The job distribution can be performed either using a local cluster (uses CPUs on the same node where the job is launched), or using a `Slurm` cluster initialized over multiple computing nodes. The instructions for the Dask client initialization in both modes can be found [here](docs/dask_client.md).
