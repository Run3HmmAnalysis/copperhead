from __future__ import print_function, division
from functools import partial
from itertools import repeat
import time
import uproot
import pickle
import math
import copy
import json
import cloudpickle
from collections import defaultdict
from cachetools import LRUCache
import lz4.frame as lz4f
from coffea.processor.processor import ProcessorABC
from coffea.processor.accumulator import (
    AccumulatorABC,
    value_accumulator,
    set_accumulator,
    dict_accumulator,
)
from coffea.processor.dataframe import (
    LazyDataFrame,
)
from coffea.nanoaod import NanoEvents
from coffea.util import _hash

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE = LRUCache(100000)


# instrument xrootd source
if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):
    def _read(self, chunkindex):
        self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
        return self._read_real(chunkindex)

    uproot.source.xrootd.XRootDSource._read_real =\
        uproot.source.xrootd.XRootDSource._read
    uproot.source.xrootd.XRootDSource._read = _read


class FileMeta(object):
    __slots__ = ['dataset', 'filename', 'treename', 'metadata']

    def __init__(self, dataset, filename, treename, metadata=None):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.metadata = metadata

    def __hash__(self):
        # As used to lookup metadata, no need for dataset
        return _hash((self.filename, self.treename))

    def __eq__(self, other):
        # In case of hash collisions
        return self.filename == other.filename and\
            self.treename == other.treename

    def maybe_populate(self, cache):
        if cache and self in cache:
            self.metadata = cache[self]

    def populated(self, clusters=False):
        '''Return true if metadata is populated

        By default, only require bare minimum metadata (numentries, uuid)
        If clusters is True, then require cluster metadata to be populated
        '''
        if self.metadata is None:
            return False
        elif clusters and 'clusters' not in self.metadata:
            return False
        return True

    def chunks(self, target_chunksize, align_clusters):
        if not self.populated(clusters=align_clusters):
            raise RuntimeError
        if align_clusters:
            chunks = [0]
            for c in self.metadata['clusters']:
                if c >= chunks[-1] + target_chunksize:
                    chunks.append(c)
            if self.metadata['clusters'][-1] != chunks[-1]:
                chunks.append(self.metadata['clusters'][-1])
            for start, stop in zip(chunks[:-1], chunks[1:]):
                yield WorkItem(self.dataset, self.filename,
                               self.treename, start, stop,
                               self.metadata['uuid'])
        else:
            n = max(round(self.metadata['numentries'] /
                          target_chunksize), 1)
            actual_chunksize = math.ceil(self.metadata['numentries'] / n)
            for index in range(n):
                start, stop = actual_chunksize * index,\
                    min(self.metadata['numentries'],
                        actual_chunksize * (index + 1))
                yield WorkItem(self.dataset, self.filename,
                               self.treename, start, stop,
                               self.metadata['uuid'])


class WorkItem(object):
    __slots__ = ['dataset', 'filename', 'treename',
                 'entrystart', 'entrystop', 'fileuuid']

    def __init__(self, dataset, filename, treename, entrystart,
                 entrystop, fileuuid):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.entrystart = entrystart
        self.entrystop = entrystop
        self.fileuuid = fileuuid


def _iadd(output, result):
    try:
        import pandas as pd
        output['out'] = pd.concat([output['out'], result['out']])
        output['metrics'] += result['metrics']
    except Exception:
        output += result


class _reduce(object):
    def __init__(self):
        pass

    def __str__(self):
        return "reduce"

    def __call__(self, items):
        if len(items) == 0:
            raise ValueError("Empty list provided to reduction")
        out = items.pop()
        if isinstance(out, AccumulatorABC):
            # if dask has a cached result, we cannot alter it,
            # so make a copy
            out = copy.deepcopy(out)
        while items:
            _iadd(out, items.pop())
        return out


def dask_executor(items, function, accumulator, **kwargs):
    if len(items) == 0:
        return accumulator
    client = kwargs.pop('client')
    ntree = kwargs.pop('treereduction', 20)
    status = kwargs.pop('status', True)
    priority = kwargs.pop('priority', 0)
    retries = kwargs.pop('retries', 3)
    heavy_input = kwargs.pop('heavy_input', None)
    function_name = kwargs.pop('function_name', None)
    reducer = _reduce()
    # secret options
    direct_heavy = kwargs.pop('direct_heavy', None)
    worker_affinity = kwargs.pop('worker_affinity', False)

    if heavy_input is not None:
        heavy_token = client.scatter(heavy_input, broadcast=True,
                                     hash=False, direct=direct_heavy)
        items = list(zip(items, repeat(heavy_token)))

    work = []
    if worker_affinity:
        workers = list(client.run(lambda: 0))

        def belongsto(workerindex, item):
            if heavy_input is not None:
                item = item[0]
            hashed = _hash((item.fileuuid, item.treename,
                            item.entrystart, item.entrystop))
            return hashed % len(workers) == workerindex

        for workerindex, worker in enumerate(workers):
            work.extend(client.map(
                function,
                [item for item in items if belongsto(workerindex, item)],
                pure=(heavy_input is not None),
                priority=priority,
                retries=retries,
                workers={worker},
                allow_other_workers=False,
            ))
    else:
        work = client.map(
            function,
            items,
            pure=(heavy_input is not None),
            priority=priority,
            retries=retries,
            key=function_name,
        )

    if (function_name == 'processor'):
        if status:
            from distributed import progress
            progress(work, multi=True, notebook=False)
        import dask.dataframe as dd
        df = dd.from_delayed(work, verify_meta=True)
        accumulator['out'] = df
        return accumulator
    else:
        while len(work) > 1:
            work = client.map(
                reducer,
                [work[i:i + ntree] for i in range(0, len(work), ntree)],
                pure=True,
                priority=priority,
                retries=retries,
            )
        work = work[0]
        if status:
            from distributed import progress
            progress(work, multi=True, notebook=False)
        accumulator += work.result()
        return accumulator


def _work_function(item, processor_instance, flatten=False,
                   savemetrics=False, mmap=False, nano=False,
                   cachestrategy=None, skipbadfiles=False,
                   retries=0, xrootdtimeout=None):
    if processor_instance == 'heavy':
        item, processor_instance = item
    if not isinstance(processor_instance, ProcessorABC):
        processor_instance = cloudpickle.loads(
            lz4f.decompress(processor_instance))
    if mmap:
        localsource = {}
    else:
        opts = dict(uproot.FileSource.defaults)
        opts.update({'parallel': None})

        def localsource(path):
            return uproot.FileSource(path, **opts)

    import warnings
    try:
        out = processor_instance.accumulator.identity()
    except Exception:
        import pandas as pd
        out = pd.DataFrame()
    retry_count = 0
    while retry_count <= retries:
        try:
            from uproot.source.xrootd import XRootDSource
            xrootdsource = XRootDSource.defaults
            xrootdsource['timeout'] = xrootdtimeout
            file = uproot.open(item.filename,
                               localsource=localsource,
                               xrootdsource=xrootdsource)
            if nano:
                cache = None
                if cachestrategy == 'dask-worker':
                    from distributed import get_worker
                    from .dask import ColumnCache
                    worker = get_worker()
                    try:
                        cache = worker.plugins[ColumnCache.name]
                    except KeyError:
                        # emit warning if not found?
                        pass
                df = NanoEvents.from_file(
                    file=file,
                    treename=item.treename,
                    entrystart=item.entrystart,
                    entrystop=item.entrystop,
                    metadata={
                        'dataset': item.dataset,
                        'filename': item.filename
                    },
                    cache=cache,
                )
            else:
                tree = file[item.treename]
                df = LazyDataFrame(tree, item.entrystart,
                                   item.entrystop, flatten=flatten)
                df['dataset'] = item.dataset
                df['filename'] = item.filename
            tic = time.time()
            out = processor_instance.process(df)
            toc = time.time()
            metrics = dict_accumulator()
            if savemetrics:
                if isinstance(file.source,
                              uproot.source.xrootd.XRootDSource):
                    metrics['bytesread'] =\
                        value_accumulator(int, file.source.bytesread)
                    metrics['dataservers'] = set_accumulator(
                        {file.source._source.get_property('DataServer')})
                metrics['columns'] = set_accumulator(df.materialized)
                metrics['entries'] = value_accumulator(int, df.size)
                metrics['processtime'] = value_accumulator(float,
                                                           toc - tic)
            # wrapped_out = dict_accumulator({'out': out,
            #                                 'metrics': metrics})
            file.source.close()
            break
        # catch xrootd errors and optionally skip
        # or retry to read the file
        except OSError as e:
            if not skipbadfiles:
                raise e
            else:
                w_str = 'Bad file source %s.' % item.filename
                if retries:
                    w_str += ' Attempt %d of %d.' %\
                        (retry_count + 1, retries + 1)
                    if retry_count + 1 < retries:
                        w_str += ' Will retry.'
                    else:
                        w_str += ' Skipping.'
                else:
                    w_str += ' Skipping.'
                warnings.warn(w_str)
            metrics = dict_accumulator()
            if savemetrics:
                metrics['bytesread'] = value_accumulator(int, 0)
                metrics['dataservers'] = set_accumulator({})
                metrics['columns'] = set_accumulator({})
                metrics['entries'] = value_accumulator(int, 0)
                metrics['processtime'] = value_accumulator(float, 0)
            # wrapped_out = dict_accumulator({'out': out,
            #                                 'metrics': metrics})
        except Exception as e:
            if retries == retry_count:
                raise e
            w_str = 'Attempt %d of %d. Will retry.' %\
                (retry_count + 1, retries + 1)
            warnings.warn(w_str)
        retry_count += 1
    return out
#    return wrapped_out


def _normalize_fileset(fileset, treename):
    if isinstance(fileset, str):
        with open(fileset) as fin:
            fileset = json.load(fin)
    for dataset, filelist in fileset.items():
        if isinstance(filelist, dict):
            local_treename = filelist['treename'] if 'treename' in\
                filelist else treename
            filelist = filelist['files']
        elif isinstance(filelist, list):
            if treename is None:
                raise ValueError('treename must be specified if the'
                                 ' fileset does not contain tree names')
            local_treename = treename
        else:
            raise ValueError(
                'list of filenames in fileset must be a list or a dict')
        for filename in filelist:
            yield FileMeta(dataset, filename, local_treename)


def _get_metadata(item, skipbadfiles=False, retries=0,
                  xrootdtimeout=None, align_clusters=False):
    import warnings
    out = set_accumulator()
    retry_count = 0
    while retry_count <= retries:
        try:
            # add timeout option according to
            # modified uproot numentries defaults
            xrootdsource = {"timeout": xrootdtimeout,
                            "chunkbytes": 32 * 1024,
                            "limitbytes": 1024**2,
                            "parallel": False}
            file = uproot.open(item.filename, xrootdsource=xrootdsource)
            tree = file[item.treename]
            metadata = {'numentries': tree.numentries,
                        'uuid': file._context.uuid}
            if align_clusters:
                metadata['clusters'] = [0] +\
                    list(c[1] for c in tree.clusters())
            out = set_accumulator([FileMeta(item.dataset,
                                            item.filename,
                                            item.treename,
                                            metadata)])
            break
        except OSError as e:
            if not skipbadfiles:
                raise e
            else:
                w_str = 'Bad file source %s.' % item.filename
                if retries:
                    w_str += ' Attempt %d of %d.' %\
                        (retry_count + 1, retries + 1)
                    if retry_count + 1 < retries:
                        w_str += ' Will retry.'
                    else:
                        w_str += ' Skipping.'
                else:
                    w_str += ' Skipping.'
                warnings.warn(w_str)
        except Exception as e:
            if retries == retry_count:
                raise e
            w_str = 'Attempt %d of %d. Will retry.' %\
                (retry_count + 1, retries + 1)
            warnings.warn(w_str)
        retry_count += 1
    return out


def run_uproot_job(fileset,
                   treename,
                   processor_instance,
                   executor,
                   executor_args={},
                   pre_executor=None,
                   pre_args=None,
                   chunksize=100000,
                   maxchunks=None,
                   metadata_cache=None,
                   ):
    if not isinstance(fileset, (Mapping, str)):
        raise ValueError(
            "Expected fileset to be a mapping dataset: "
            "list(files) or filename")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError(
            "Expected processor_instance to derive from ProcessorABC")

    if pre_executor is None:
        pre_executor = executor
    if pre_args is None:
        pre_args = dict(executor_args)
    if metadata_cache is None:
        metadata_cache = DEFAULT_METADATA_CACHE

    fileset = list(_normalize_fileset(fileset, treename))
    for filemeta in fileset:
        filemeta.maybe_populate(metadata_cache)

    # pop _get_metdata args here (also sent to _work_function)
    skipbadfiles = executor_args.pop('skipbadfiles', False)
    retries = executor_args.pop('retries', 0)
    xrootdtimeout = executor_args.pop('xrootdtimeout', None)
    align_clusters = executor_args.pop('align_clusters', False)
    metadata_fetcher = partial(_get_metadata,
                               skipbadfiles=skipbadfiles,
                               retries=retries,
                               xrootdtimeout=xrootdtimeout,
                               align_clusters=align_clusters,
                               )

    chunks = []
    if maxchunks is None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = set(
            filemeta for filemeta in fileset if not filemeta.populated(
                clusters=align_clusters))
        if len(to_get) > 0:
            out = set_accumulator()
            pre_arg_override = {
                'desc': 'Preprocessing',
                'unit': 'file',
                'tailtimeout': None,
                'worker_affinity': False,
            }
            pre_args.update(pre_arg_override)
            pre_executor(to_get, metadata_fetcher, out, **pre_args)
            while out:
                item = out.pop()
                metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(metadata_cache)
        while fileset:
            filemeta = fileset.pop()
            if skipbadfiles and not filemeta.populated(
              clusters=align_clusters):
                continue
            for chunk in filemeta.chunks(chunksize, align_clusters):
                chunks.append(chunk)
    else:
        # get just enough file info to compute chunking
        nchunks = defaultdict(int)
        while fileset:
            filemeta = fileset.pop()
            if nchunks[filemeta.dataset] >= maxchunks:
                continue
            if not filemeta.populated(clusters=align_clusters):
                filemeta.metadata = metadata_fetcher(
                    filemeta).pop().metadata
                metadata_cache[filemeta] = filemeta.metadata
            if skipbadfiles and not filemeta.populated(
              clusters=align_clusters):
                continue
            for chunk in filemeta.chunks(chunksize, align_clusters):
                chunks.append(chunk)
                nchunks[filemeta.dataset] += 1
                if nchunks[filemeta.dataset] >= maxchunks:
                    break

    # pop all _work_function args here
    savemetrics = executor_args.pop('savemetrics', False)
    flatten = executor_args.pop('flatten', False)
    mmap = executor_args.pop('mmap', False)
    nano = executor_args.pop('nano', False)
    cachestrategy = executor_args.pop('cachestrategy', None)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(
            cloudpickle.dumps(processor_instance),
            compression_level=pi_compression)
    closure = partial(
        _work_function,
        flatten=flatten,
        savemetrics=savemetrics,
        mmap=mmap,
        nano=nano,
        cachestrategy=cachestrategy,
        skipbadfiles=skipbadfiles,
        retries=retries,
        xrootdtimeout=xrootdtimeout,
    )
    # hack around dask/dask#5503 which is really a
    # silly request but here we are
    if executor is dask_executor:
        executor_args['heavy_input'] = pi_to_send
        executor_args['function_name'] = 'processor'
        closure = partial(closure, processor_instance='heavy')
    else:
        closure = partial(closure, processor_instance=pi_to_send)

    # out = processor_instance.accumulator.identity()
    import pandas as pd
    out = pd.DataFrame()

    wrapped_out = dict_accumulator({'out': out,
                                    'metrics': dict_accumulator()})
    exe_args = {
        'unit': 'chunk',
        'function_name': type(processor_instance).__name__,
    }
    exe_args.update(executor_args)
    executor(chunks, closure, wrapped_out, **exe_args)
    wrapped_out['metrics']['chunks'] = value_accumulator(int, len(chunks))
    processor_instance.postprocess(out)

    if savemetrics:
        return wrapped_out['out'], wrapped_out['metrics']
    return wrapped_out['out']
