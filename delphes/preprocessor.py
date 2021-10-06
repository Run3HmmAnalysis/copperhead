import glob
import awkward as ak
from coffea.nanoevents import DelphesSchema, NanoEventsFactory

from delphes.cross_sections import cross_sections


def get_sum_wgts(file):
    events = NanoEventsFactory.from_root(
        file,
        "Delphes",
        schemaclass=DelphesSchema,
    ).events()
    return ak.sum(events.Event.Weight)


def get_fileset(datasets, parameters):
    fileset = {}
    for sample, path in datasets.items():
        if sample not in cross_sections.keys():
            print(f"Cross section for {sample} missing, skipping")
        filelist = [glob.glob(parameters["server"] + path + "/*.root")[0]]
        # filelist = glob.glob(parameters['server'] + path + '/*.root')
        futures = parameters["client"].map(get_sum_wgts, filelist)
        nEvts = sum(parameters["client"].gather(futures))
        mymetadata = {
            "lumi_wgt": parameters["lumi"] * cross_sections[sample] / nEvts,
            "regions": ["z-peak", "h-sidebands", "h-peak"],
            "channels": ["ggh_01j", "ggh_2j", "vbf", "vbf_01j", "vbf_2j"],
        }
        fileset[sample] = {
            "treename": "Delphes",
            "files": filelist,
            "metadata": mymetadata,
        }
    return fileset
