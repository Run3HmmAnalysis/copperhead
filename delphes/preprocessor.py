import glob
import tqdm
import json
import awkward as ak
from coffea.nanoevents import DelphesSchema, NanoEventsFactory

from delphes.config.cross_sections import cross_sections


def get_sum_wgts(file):
    try:
        events = NanoEventsFactory.from_root(
            file,
            "Delphes",
            schemaclass=DelphesSchema,
        ).events()
        result = (file, ak.sum(events.Event.Weight))
    except Exception:
        result = (file, 0)
    return result


def get_fileset(datasets, parameters, save_to=None, load_from=None):
    if load_from:
        with open(load_from, "r") as fp:
            fileset = json.load(fp)
    else:
        fileset = {}
        for sample, path in tqdm.tqdm(datasets.items()):
            if sample not in cross_sections.keys():
                print(f"Cross section for {sample} missing, skipping")
            if len(glob.glob(path)) > 0:
                filelist = glob.glob(path)
            else:
                filelist = glob.glob(parameters["server"] + path + "/*.root")
                # filelist = [glob.glob(parameters["server"] + path + "/*.root")[0]]
            futures = parameters["client"].map(get_sum_wgts, filelist)
            results = parameters["client"].gather(futures)
            cleaned_filelist = [r[0] for r in results if r[1] > 0]
            # bad_files = [r[0] for r in results if r[1] <= 0]
            nEvts = sum([r[1] for r in results if r[1] > 0])
            mymetadata = {
                "lumi_wgt": str(
                    parameters["lumi"] * cross_sections[sample] / float(nEvts)
                ),
                "regions": ["z-peak", "h-sidebands", "h-peak"],
                "channels": ["ggh_01j", "ggh_2j", "vbf", "vbf_01j", "vbf_2j"],
            }
            fileset[sample] = {
                "treename": "Delphes",
                "files": cleaned_filelist,
                "metadata": mymetadata,
            }
        if save_to:
            with open(save_to, "w") as fp:
                json.dump(fileset, fp)
    return fileset
