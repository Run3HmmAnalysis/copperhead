import json

uncertainty_v1 = {}


def append_unc(uncertainty_dict, Name, Type, ggh_unc, qqh_unc, bkg_unc):
    uncertainty_dict[Name] = {
        "type": Type,
        "ggh": ggh_unc,
        "qqh": qqh_unc,
        "bkg": bkg_unc,
    }


# uncertainty_v1["lumi_13TeV_2016"] = []
uncertainty_v1["lumi_13TeV_2016"] = {
    "type": "lnN",
    "ggh": "1.007",
    "qqh": "1.007",
    "bkg": "-",
}

uncertainty_v1["lumi_13TeV_2017"] = {
    "type": "lnN",
    "ggh": "1.007",
    "qqh": "1.007",
    "bkg": "-",
}

uncertainty_v1["lumi_13TeV_2018"] = {
    "type": "lnN",
    "ggh": "1.007",
    "qqh": "1.007",
    "bkg": "-",
}

append_unc(uncertainty_v1, "QCDscale_qqH", "lnN", "-", "0.997/1.004", "-")

with open("uncertainty_v1.json", "w") as outfile:
    json.dump(uncertainty_v1, outfile, indent=4)
