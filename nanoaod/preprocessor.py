# import time
import subprocess
import glob
import tqdm

import uproot

from nanoaod.config.parameters import parameters
from nanoaod.config.cross_sections import cross_sections

DEBUG = False


def load_sample(dataset, parameters):
    xrootd = parameters["xrootd"]
    args = {
        "year": parameters["year"],
        "server": parameters["server"],
        "datasets_from": "purdue",
        "debug": DEBUG,
        "xrootd": xrootd,
        "timeout": 120,
    }
    samp_info = SamplesInfo(**args)
    samp_info.load(dataset, use_dask=True, client=parameters["client"])
    samp_info.finalize()
    return {dataset: samp_info}


def load_samples(datasets, parameters):
    args = {
        "year": parameters["year"],
        "server": parameters["server"],
        "datasets_from": "purdue",
        "debug": DEBUG,
    }
    samp_info_total = SamplesInfo(**args)
    print("Loading lists of paths to ROOT files for these datasets:", datasets)
    for d in tqdm.tqdm(datasets):
        if d in samp_info_total.samples:
            continue
        si = load_sample(d, parameters)[d]
        if "files" not in si.fileset[d].keys():
            continue
        if si.fileset[d]["files"] == {}:
            continue
        samp_info_total.data_entries += si.data_entries
        samp_info_total.fileset.update(si.fileset)
        samp_info_total.metadata.update(si.metadata)
        samp_info_total.lumi_weights.update(si.lumi_weights)
        samp_info_total.samples.append(si.sample)
    return samp_info_total


def read_via_xrootd(server, path, from_das=True):
    if from_das:
        command = f'dasgoclient --query=="file dataset={path}"'
    else:
        command = f"xrdfs {server} ls -R {path} | grep '.root'"
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    result = proc.stdout.readlines()
    if proc.stderr.readlines():
        print("Loading error! This may help:")
        print("    voms-proxy-init --voms cms")
        print("    source /cvmfs/cms.cern.ch/cmsset_default.sh")
    result = [server + r.rstrip().decode("utf-8") for r in result]
    return result


class SamplesInfo(object):
    def __init__(self, **kwargs):
        self.year = kwargs.pop("year", "2016")
        self.xrootd = kwargs.pop("xrootd", True)
        self.server = kwargs.pop("server", "root://xrootd.rcac.purdue.edu/")
        self.timeout = kwargs.pop("timeout", 60)
        self.debug = kwargs.pop("debug", False)
        datasets_from = kwargs.pop("datasets_from", "purdue")

        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.is_mc = True

        if "purdue" in datasets_from:
            from nanoaod.config.datasets import datasets
        elif "pisa" in datasets_from:
            from nanoaod.config.datasets_pisa import datasets

        self.paths = datasets[self.year]

        if "2016" in self.year:
            self.lumi = 35900.0
        elif "2017" in self.year:
            self.lumi = 41530.0
        elif "2018" in self.year:
            self.lumi = 59970.0
        # print('year: ', self.year)
        # print('Default lumi: ', self.lumi)

        self.data_entries = 0
        self.sample = ""
        self.samples = []

        self.fileset = {}
        self.metadata = {}

        self.lumi_weights = {}

    def load(self, sample, use_dask, client=None):
        if "data" in sample:
            self.is_mc = False

        res = self.load_sample(sample, use_dask, client)

        self.sample = sample
        self.samples = [sample]
        self.fileset = {sample: res["files"]}

        self.metadata = res["metadata"]
        self.data_entries = res["data_entries"]

    def load_sample(self, sample, use_dask=False, client=None):
        if (sample not in self.paths) or (self.paths[sample] == ""):
            # print(f"Couldn't load {sample}! Skipping.")
            return {
                "sample": sample,
                "metadata": {},
                "files": {},
                "data_entries": 0,
                "is_missing": True,
            }

        all_files = []
        metadata = {}
        data_entries = 0

        if self.xrootd:
            all_files = read_via_xrootd(self.server, self.paths[sample])
        elif self.paths[sample].endswith(".root"):
            all_files = [self.paths[sample]]
        else:
            all_files = glob.glob(self.server + self.paths[sample] + "/**/**/**/*.root")
            all_files = all_files + glob.glob(
                self.server + self.paths[sample] + "/**/**/*.root"
            )

        if self.debug:
            all_files = [all_files[0]]

        # print(f"Loading {sample}: {len(all_files)} files")

        sumGenWgts = 0
        nGenEvts = 0

        if use_dask:
            from dask.distributed import get_client

            if not client:
                client = get_client()
            if "data" in sample:
                work = client.map(self.get_data, all_files, priority=100)
            else:
                work = client.map(self.get_mc, all_files, priority=100)
            for w in work:
                ret = w.result()
                if "data" in sample:
                    data_entries += ret["data_entries"]
                else:
                    sumGenWgts += ret["sumGenWgts"]
                    nGenEvts += ret["nGenEvts"]
        else:
            for f in all_files:
                if "data" in sample:
                    data_entries += self.get_data(f)["data_entries"]
                else:
                    ret = self.get_mc(f)
                    sumGenWgts += ret["sumGenWgts"]
                    nGenEvts += ret["nGenEvts"]

        metadata["sumGenWgts"] = sumGenWgts
        metadata["nGenEvts"] = nGenEvts

        files = {"files": all_files, "treename": "Events"}
        return {
            "sample": sample,
            "metadata": metadata,
            "files": files,
            "data_entries": data_entries,
            "is_missing": False,
        }

    def get_data(self, f):
        ret = {}
        file = uproot.open(f, timeout=self.timeout)
        tree = file["Events"]
        ret["data_entries"] = tree.num_entries
        return ret

    def get_mc(self, f):
        ret = {}
        tree = uproot.open(f, timeout=self.timeout)["Runs"]
        if ("NanoAODv6" in f) or ("NANOV10" in f):
            ret["sumGenWgts"] = tree["genEventSumw_"].array()[0]
            ret["nGenEvts"] = tree["genEventCount_"].array()[0]
        else:
            ret["sumGenWgts"] = tree["genEventSumw"].array()[0]
            ret["nGenEvts"] = tree["genEventCount"].array()[0]
        return ret

    def finalize(self):
        if self.is_mc:
            if len(self.metadata) == 0:
                return 0
            N = self.metadata["sumGenWgts"]
            numevents = self.metadata["nGenEvts"]
            if isinstance(cross_sections[self.sample], dict):
                xsec = cross_sections[self.sample][self.year]
            else:
                xsec = cross_sections[self.sample]
            print(self.lumi, xsec, N)
            if N > 0:
                self.lumi_weights[self.sample] = xsec * self.lumi / N
            else:
                self.lumi_weights[self.sample] = 0
            # print(f"{self.sample}: events={numevents}")
            return numevents
        else:
            return self.data_entries
