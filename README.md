# Run3 H→µµ analysis framework

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI/CD](https://github.com/kondratyevd/hmumu-coffea/actions/workflows/ci.yml/badge.svg)

The framework was originally designed for the [search for the Higgs boson decays into two muons](https://inspirehep.net/literature/1815813) with the CMS detector at the Large Hadron Collider. The published results in the channel targeting the VBF Higgs production mode were reproduced with 1% precision. The framework was also used for the [projections of the H&rarr;µµ search sensitivity to HL-LHC](https://cds.cern.ch/record/2804002/).

Currently the framework is under development to integrate both of the main Higgs production modes (ggH and VBF), and to prepare for analyzing the Run3 data when it becomes available.

## Framework structure, data formats, used packages
The input data for the framework should be in `NanoAOD` format.

The analysis workflow contains three stages:
- **Stage 1** includes event and object selection, application of corrections, and construction of new variables. The data processing is implemented via [columnar approach](https://indico.cern.ch/event/759388/contributions/3306852/attachments/1816027/2968106/ncsmith-how2019-columnar.pdf), making use of the tools provided by [coffea](https://github.com/CoffeaTeam/coffea) package. The data columns are handled via `coffea`'s `NanoEvents` format which relies on *jagged arrays* implemented in [Awkward Array](https://github.com/scikit-hep/awkward-1.0) package. After event selection, the jagged arrays are converted to flat [pandas](https://github.com/pandas-dev/pandas) dataframes and saved into [Apache Parquet](https://github.com/apache/parquet-format) files.
- **Stage 2** (WIP) contains / will contain evaluation of MVA methods (boosted decision trees, deep neural networks), event categorization, and producing histograms. The stage 2 workflow is structured as follows:
  - Outputs of Stage 1 (`Parquet` files) are loaded as partitions of a [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html) (similar to Pandas DF, but partitioned and "lazy").
  - The Dask DataFrame is (optionally) re-partitioned to decrease number of partitions.
  - The partitions are processed in parallel; for each partition, the following sequence is executed:
    - Partition of the Dask DataFrame is "computed" (converted to a Pandas Dataframe).
    - Evaluation of MVA models (can also be done after categorization). Current inmplementation includes evaluation of `PyTorch` DNN models and/or `XGBoost` BDT models. Other methods can be implemented, but one has to verify that they would work well in a distributed environment (e.g. Tensorflow sessions are not very good for that).
    - Definition of event categories and/or MVA bins.
    - Creating histograms using [scikit-hep/hist](https://github.com/scikit-hep/hist).
    - Saving histograms.
    - (Optionally) Saving individual columns (can be used later for unbinned fits).

- **Stage 3** (WIP) contains / will contain plotting, parametric fits, preparation of datacards for statistical analysis. The plotting is done via [scikit-hep/mplhep](https://github.com/scikit-hep/mplhep).

## Job parallelization
The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster (uses CPUs on the same node where the job is launched), or a distributed `Slurm` cluster initialized over multiple computing nodes. The instructions for the Dask client initialization in both modes can be found [here](docs/dask_client.md).

It is possible to create a cluster with other batch submission systems (`HTCondor`, `PBS`, etc., see full list in [Dask-Jobqueue API](https://jobqueue.dask.org/en/latest/api.html#)).

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

## Test runs
Run each stage individually, or the full analysis workflow on a single input file.
```bash
python3 -W ignore tests/test_stage1.py
python3 -W ignore tests/test_stage2.py
python3 -W ignore tests/test_stage3.py
python3 -W ignore tests/test_continuous.py
```
---
## Credits
- **Original developer:** [Dmitry Kondratyev](https://github.com/kondratyevd)
- **Contributors:** [Arnab Purohit](https://github.com/ArnabPurohit), [Stefan Piperov](https://github.com/piperov)
