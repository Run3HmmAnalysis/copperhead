import pandas as pd
from python.io import mkdir

rename_regions = {
    "h-peak": "SR",
    "h-sidebands": "SB",
    "z-peak": "Z",
}
signal_groups = ["VBF", "ggH"]
rate_syst_lookup = {
    "2016": {
        "XsecAndNorm2016DY": {"DY": 1.1291},
        "XsecAndNorm2016EWK": {"EWK": 1.06131},
        "XsecAndNormTT+ST": {"TT+ST": 1.182},
        "XsecAndNormVV": {"VV": 1.13203},
        "XsecAndNormggH": {"ggH": 1.38206},
    },
    "2017": {
        "XsecAndNorm2017DYJ2": {"DY": 1.13020},
        "XsecAndNorm2017EWK": {"EWK": 1.05415},
        "XsecAndNormTT+ST": {"TT+ST": 1.18406},
        "XsecAndNormVV": {"VV": 1.05653},
        "XsecAndNormggH": {"ggH": 1.37126},
    },
    "2018": {
        "XsecAndNorm2018DYJ2": {"DY": 1.12320},
        "XsecAndNorm2018EWK": {"EWK": 1.05779},
        "XsecAndNormTT+ST": {"TT+ST": 1.18582},
        "XsecAndNormVV": {"VV": 1.05615},
        "XsecAndNormggH": {"ggH": 1.38313},
    },
}
lumi_syst = {
    "2016": {
        "uncor2016": 2.2,
        "xyfac": 0.9,
        "len": 0.0,
        "bb": 0.4,
        "beta": 0.5,
        "calib": 0.0,
        "ghost": 0.4,
    },
    "2017": {
        "uncor2017": 2.0,
        "xyfac": 0.8,
        "len": 0.3,
        "bb": 0.4,
        "beta": 0.5,
        "calib": 0.3,
        "ghost": 0.1,
    },
    "2018": {
        "uncor2018": 1.5,
        "xyfac": 2.0,
        "len": 0.2,
        "bb": 0.0,
        "beta": 0.0,
        "calib": 0.2,
        "ghost": 0.0,
    },
}


def build_datacards(var_name, yield_df, parameters):
    channels = parameters["channels"]
    regions = parameters["regions"]
    years = parameters["years"]
    global_path = parameters["global_path"]
    label = parameters["label"]

    templates_path = f"{global_path}/{label}/stage3_templates/{var_name}"
    datacard_path = f"{global_path}/{label}/stage3_datacards/"
    mkdir(datacard_path)
    datacard_path += "/" + var_name
    mkdir(datacard_path)

    for year in years:
        for channel in channels:
            for region in regions:
                region_new = rename_regions[region]
                bin_name = f"{channel}_{region_new}_{year}"
                datacard_name = (
                    f"{datacard_path}/datacard_{channel}_{region_new}_{year}.txt"
                )
                templates_file = f"{templates_path}/{channel}_{region}_{year}.root"
                datacard = open(datacard_name, "w")
                datacard.write("imax 1\n")
                datacard.write("jmax *\n")
                datacard.write("kmax *\n")
                datacard.write("---------------\n")
                datacard.write(
                    f"shapes * {bin_name} {templates_file} $PROCESS $PROCESS_$SYSTEMATIC\n"
                )
                datacard.write("---------------\n")
                data_str = print_data(
                    yield_df, var_name, region, channel, year, bin_name
                )
                datacard.write(data_str)
                datacard.write("---------------\n")
                mc_str = print_mc(yield_df, var_name, region, channel, year, bin_name)
                datacard.write(mc_str)
                datacard.write("---------------\n")
                # shape_syst = print_shape_syst(yield_df, mc_df)
                # datacard.write("---------------\n")
                # datacard.write(systematics)
                datacard.write(f"{bin_name} autoMCStats 0 1 1\n")
                datacard.write("---------------\n")
                datacard.close()
                print(f"Saved datacard to {datacard_name}")
    return


def print_data(yield_df, var_name, region, channel, year, bin_name):
    if "Data" in yield_df.group.unique():
        data_yield = yield_df.loc[
            (yield_df.var_name == var_name)
            & (yield_df.region == region)
            & (yield_df.channel == channel)
            & (yield_df.year == year)
            & (yield_df.variation == "nominal")
            & (yield_df.group == "Data"),
            "yield",
        ].values[0]
        data_str = "{:<20} {:>20}\n".format("bin", bin_name) + "{:<20} {:>20}\n".format(
            "observation", int(data_yield)
        )
        return data_str

    else:
        return ""


def print_mc(yield_df, var_name, region, channel, year, bin_name):
    # get yields for each process
    sig_counter = 0
    bkg_counter = 0
    mc_rows = []
    nuisances = {}
    all_nuisances = []
    nuisance_lines = {}

    groups = [g for g in yield_df.group.unique() if g != "Data"]
    for group in groups:
        if group in signal_groups:
            sig_counter -= 1
            igroup = sig_counter
        else:
            bkg_counter += 1
            igroup = bkg_counter

        mc_yield = yield_df.loc[
            (yield_df.var_name == var_name)
            & (yield_df.region == region)
            & (yield_df.channel == channel)
            & (yield_df.year == year)
            & (yield_df.variation == "nominal")
            & (yield_df.group == group),
            "yield",
        ].values[0]

        mc_row = {"group": group, "igroup": igroup, "yield": mc_yield}

        nuisances[group] = []
        variations = yield_df.loc[
            ((yield_df.group == group) & (yield_df.year == year)), "variation"
        ].unique()
        for v in variations:
            if v == "nominal":
                continue
            v_name = v.replace("Up", "").replace("Down", "")
            if v_name not in all_nuisances:
                all_nuisances.append(v_name)
                nuisance_lines[v_name] = "{:<20} {:<9}".format(v_name, "shape")
            if v_name not in nuisances[group]:
                nuisances[group].append(v_name)

        mc_rows.append(mc_row)

    mc_df = pd.DataFrame(mc_rows).sort_values(by="igroup")
    for group, gr_nuis in nuisances.items():
        for nuisance in gr_nuis:
            mc_df.loc[mc_df.group == group, nuisance] = "1.0"

    for rate_unc, apply_to in rate_syst_lookup[year].items():
        if rate_unc not in all_nuisances:
            all_nuisances.append(rate_unc)
            nuisance_lines[rate_unc] = "{:<20} {:<9}".format(rate_unc, "lnN")
        for group, value in apply_to.items():
            mc_df.loc[mc_df.group == group, rate_unc] = str(value)

    for lumi_unc, value in lumi_syst[year].items():
        if lumi_unc not in all_nuisances:
            all_nuisances.append(lumi_unc)
            nuisance_lines[lumi_unc] = "{:<20} {:<9}".format(lumi_unc, "lnN")
            mc_df.loc[:, lumi_unc] = str(1 + value / 100)

    mc_df = mc_df.fillna("-")

    # prepare datacard lines
    mc_str_1 = "{:<30}".format("bin")
    mc_str_2 = "{:<30}".format("process")
    mc_str_3 = "{:<30}".format("process")
    mc_str_4 = "{:<30}".format("rate")

    for group in groups:
        group_df = mc_df[mc_df.group == group]
        mc_str_1 += "{:<20}".format(bin_name)
        mc_str_2 += "{:<20}".format(group)
        mc_str_3 += "{:<20}".format(group_df["igroup"].values[0])
        mc_str_4 += "{:<20}".format(group_df["yield"].values[0])
        for nuisance in all_nuisances:
            nuisance_lines[nuisance] += "{:<20}".format(group_df[nuisance].values[0])

    process_lines = f"{mc_str_1}\n{mc_str_2}\n{mc_str_3}\n{mc_str_4}\n"

    mc_str = process_lines + "---------------\n"
    for nuisance in all_nuisances:
        mc_str += nuisance_lines[nuisance] + "\n"

    return mc_str
