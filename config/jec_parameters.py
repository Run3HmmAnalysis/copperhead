def for_all_years(value):
    out = {k: value for k in ["2016", "2017", "2018"]}
    return out


def get_variations(sources):
    result = []
    for v in sources:
        result.append(v + "_up")
        result.append(v + "_down")
    return result


jec_parameters = {}

jec_unc_to_consider = {
    "2016": [
        "Absolute",
        # "Absolute2016",
        # "BBEC1",
        # "BBEC12016",
        # "EC2",
        # "EC22016",
        "HF",
        # "HF2016",
        "RelativeBal",
        # "RelativeSample2016",
        # "FlavorQCD",
    ],
    "2017": [
        "Absolute",
        # "Absolute2017",
        # "BBEC1",
        # "BBEC12017",
        # "EC2",
        # "EC22017",
        "HF",
        # "HF2017",
        "RelativeBal",
        # "RelativeSample2017",
        # "FlavorQCD",
    ],
    "2018": [
        "Absolute",
        # "Absolute2018",
        # "BBEC1",
        # "BBEC12018",
        # "EC2",
        # "EC22018",
        "HF",
        # "HF2018",
        "RelativeBal",
        # "RelativeSample2018",
        # "FlavorQCD",
    ],
}

jec_parameters["jec_variations"] = {
    year: get_variations(jec_unc_to_consider[year]) for year in ["2016", "2017", "2018"]
}

jec_parameters["runs"] = {
    "2016": ["B", "C", "D", "E", "F", "G", "H"],
    "2017": ["B", "C", "D", "E", "F"],
    "2018": ["A", "B", "C", "D"],
}

jec_parameters["jec_levels_mc"] = for_all_years(
    ["L1FastJet", "L2Relative", "L3Absolute"]
)
jec_parameters["jec_levels_data"] = for_all_years(
    ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]
)

jec_parameters["jec_tags"] = {
    "2016": "Summer16_07Aug2017_V11_MC",
    "2017": "Fall17_17Nov2017_V32_MC",
    "2018": "Autumn18_V19_MC",
}

jec_parameters["jer_tags"] = {
    "2016": "Summer16_25nsV1_MC",
    "2017": "Fall17_V3_MC",
    "2018": "Autumn18_V7_MC",
}

jec_parameters["jec_data_tags"] = {
    "2016": {
        "Summer16_07Aug2017BCD_V11_DATA": ["B", "C", "D"],
        "Summer16_07Aug2017EF_V11_DATA": ["E", "F"],
        "Summer16_07Aug2017GH_V11_DATA": ["G", "H"],
    },
    "2017": {
        "Fall17_17Nov2017B_V32_DATA": ["B"],
        "Fall17_17Nov2017C_V32_DATA": ["C"],
        "Fall17_17Nov2017DE_V32_DATA": ["D", "E"],
        "Fall17_17Nov2017F_V32_DATA": ["F"],
    },
    "2018": {
        "Autumn18_RunA_V19_DATA": ["A"],
        "Autumn18_RunB_V19_DATA": ["B"],
        "Autumn18_RunC_V19_DATA": ["C"],
        "Autumn18_RunD_V19_DATA": ["D"],
    },
}

# jer_variations = ["jer1", "jer2", "jer3", "jer4", "jer5", "jer6"]
jer_variations = []
jec_parameters["jer_variations"] = {
    year: get_variations(jer_variations) for year in ["2016", "2017", "2018"]
}
