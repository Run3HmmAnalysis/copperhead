import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import coffea.processor as processor
from nanoaod.processor import DimuonProcessor
from nanoaod.preprocessor import SamplesInfo

import pyspark.sql
from pyarrow.util import guid
from coffea.processor.spark.detail import _spark_initialize, _spark_stop

os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

from coffea.processor import run_spark_job
from coffea.processor.spark.spark_executor import spark_executor


# This example would only work if some variables in coffea/nanoevents/transforms.py
# are manually changed from 64bit to 32bit (counts2nestedindex_form, counts2offsets_form)
# itemsize changed from 8 to 4, etc.
# Also, dataset name should be changed from pyarrow.StringScalar to python string.

# In addition, custom Spark branch is needed, otherwise it is too slow
# https://github.com/lgray/spark/tree/v2.4.4_arrowhacks

# So, a lot of trouble for questionable reward.
# This code will not work out of the box, and even if all hacks are implemented, there
# are unexplained memory problems when processing large datasets.

if __name__ == "__main__":
    tick = time.time()

    spark_config = (
        pyspark.sql.SparkSession.builder.appName("spark-executor-test-%s" % guid())
        .master("local[1]")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.executor.memory", "7g")
        .config("spark.executor.cores", "1")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", 100000)
        .config("spark.cores.max", "1")
    )

    spark = _spark_initialize(
        config=spark_config,
        log_level="ERROR",
        spark_progress=False,
        laurelin_version="1.0.0",
    )
    print("Spark initialized")

    file_name = "vbf_powheg_dipole_NANOV10_2018.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    dataset = {"test": file_path}

    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = "2018"
    samp_info.load("test", use_dask=False)
    samp_info.lumi_weights["test"] = 1.0

    executor = spark_executor
    executor_args = {"schema": processor.NanoAODSchema, "file_type": "root"}
    processor_args = {"samp_info": samp_info, "do_timer": False, "do_btag_syst": False}
    print(samp_info.fileset)

    output = run_spark_job(
        samp_info.fileset,
        DimuonProcessor(**processor_args),
        executor,
        spark=spark,
        thread_workers=32,
        partitionsize=100000,
        executor_args=executor_args,
    )
    _spark_stop(spark)

    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
