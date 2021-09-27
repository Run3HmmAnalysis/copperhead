import dask
from dask.distributed import Client

import dask.dataframe as dd
import pandas as pd
import numpy as np
import glob


def load_data(path):
    if len(path) > 0:
        df = dd.read_parquet(path)
    else:
        df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    return df


def workflow(client, paths, parameters):
    # Load dataframes
    df_future = client.map(load_data, paths)
    df_future = client.gather(df_future)
    
    # Select only one column and concatenate
    df_future = [d[['dimuon_mass']] for d in df_future]
    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    df = df.compute()
    df.reset_index(inplace=True, drop=True)

    background_fit(df)


def background_fit(column):
    # Add your code here
    print(column)


if __name__ == '__main__':
    parameters = {'ncpus': 40}
    paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_*/*.parquet')

    client = Client(
        processes=True,
        n_workers=parameters['ncpus'],
        threads_per_worker=1,
        memory_limit='4GB'
    )

    workflow(client, paths, parameters)
