import numpy as np
import pandas as pd
import awkward as ak
from python.math_tools import p4_sum, delta_r, rapidity


def prepare_jets(df, is_mc):
    # Initialize missing fields (needed for JEC)
    df["Jet", "pt_raw"] = (1 - df.Jet.rawFactor) * df.Jet.pt
    df["Jet", "mass_raw"] = (1 - df.Jet.rawFactor) * df.Jet.mass
    df["Jet", "rho"] = ak.broadcast_arrays(df.fixedGridRhoFastjetAll, df.Jet.pt)[0]

    if is_mc:
        df["Jet", "pt_gen"] = ak.values_astype(
            ak.fill_none(df.Jet.matched_gen.pt, 0), np.float32
        )


def fill_jets(output, variables, jet1, jet2):
    variable_names = [
        "jet1_pt",
        "jet1_eta",
        "jet1_rap",
        "jet1_phi",
        "jet1_qgl",
        "jet1_jetId",
        "jet1_puId",
        "jet2_pt",
        "jet2_eta",
        "jet2_rap",
        "jet2_phi",
        "jet2_qgl",
        "jet2_jetId",
        "jet2_puId",
        "jj_mass",
        "jj_mass_log",
        "jj_pt",
        "jj_eta",
        "jj_phi",
        "jj_dEta",
        "jj_dPhi",
        "mmj1_dEta",
        "mmj1_dPhi",
        "mmj1_dR",
        "mmj2_dEta",
        "mmj2_dPhi",
        "mmj2_dR",
        "mmj_min_dEta",
        "mmj_min_dPhi",
        "mmjj_pt",
        "mmjj_eta",
        "mmjj_phi",
        "mmjj_mass",
        "rpt",
        "zeppenfeld",
        "ll_zstar_log",
        "nsoftjets2",
        "nsoftjets5",
        "htsoft2",
        "htsoft5",
        "selection",
    ]

    for v in variable_names:
        variables[v] = -999.0

    # Fill single jet variables
    for v in ["pt", "eta", "phi", "qgl", "jetId", "puId"]:
        variables[f"jet1_{v}"] = jet1[v]
        variables[f"jet2_{v}"] = jet2[v]

    variables.jet1_rap = rapidity(jet1)
    variables.jet2_rap = rapidity(jet2)

    # Fill dijet variables
    jj = p4_sum(jet1, jet2)
    for v in ["pt", "eta", "phi", "mass"]:
        variables[f"jj_{v}"] = jj[v]

    variables.jj_mass_log = np.log(variables.jj_mass)

    variables.jj_dEta, variables.jj_dPhi, _ = delta_r(
        variables.jet1_eta, variables.jet2_eta, variables.jet1_phi, variables.jet2_phi
    )

    # Fill dimuon-dijet system variables
    mm_columns = ["dimuon_pt", "dimuon_eta", "dimuon_phi", "dimuon_mass"]
    jj_columns = ["jj_pt", "jj_eta", "jj_phi", "jj_mass"]

    dimuons = output.loc[:, mm_columns]
    dijets = variables.loc[:, jj_columns]

    # careful with renaming
    dimuons.columns = ["mass", "pt", "eta", "phi"]
    dijets.columns = ["pt", "eta", "phi", "mass"]

    mmjj = p4_sum(dimuons, dijets)
    for v in ["pt", "eta", "phi", "mass"]:
        variables[f"mmjj_{v}"] = mmjj[v]

    variables.zeppenfeld = output.dimuon_eta - 0.5 * (
        variables.jet1_eta + variables.jet2_eta
    )

    variables.rpt = variables.mmjj_pt / (
        output.dimuon_pt + variables.jet1_pt + variables.jet2_pt
    )

    ll_ystar = output.dimuon_rap - (variables.jet1_rap + variables.jet2_rap) / 2

    ll_zstar = abs(ll_ystar / (variables.jet1_rap - variables.jet2_rap))

    variables.ll_zstar_log = np.log(ll_zstar)

    variables.mmj1_dEta, variables.mmj1_dPhi, variables.mmj1_dR = delta_r(
        output.dimuon_eta, variables.jet1_eta, output.dimuon_phi, variables.jet1_phi
    )

    variables.mmj2_dEta, variables.mmj2_dPhi, variables.mmj2_dR = delta_r(
        output.dimuon_eta, variables.jet2_eta, output.dimuon_phi, variables.jet2_phi
    )

    variables.mmj_min_dEta = np.where(
        variables.mmj1_dEta,
        variables.mmj2_dEta,
        (variables.mmj1_dEta < variables.mmj2_dEta),
    )

    variables.mmj_min_dPhi = np.where(
        variables.mmj1_dPhi,
        variables.mmj2_dPhi,
        (variables.mmj1_dPhi < variables.mmj2_dPhi),
    )


def jet_id(jets, parameters, year):
    pass_jet_id = np.ones_like(jets.jetId, dtype=bool)
    if "loose" in parameters["jet_id"]:
        pass_jet_id = jets.jetId >= 1
    elif "tight" in parameters["jet_id"]:
        if "2016" in year:
            pass_jet_id = jets.jetId >= 3
        else:
            pass_jet_id = jets.jetId >= 2
    return pass_jet_id


def jet_puid(jets, parameters, year):
    jet_puid_opt = parameters["jet_puid"]
    puId = jets.puId17 if year == "2017" else jets.puId
    jet_puid_wps = {
        "loose": (puId >= 4) | (jets.pt > 50),
        "medium": (puId >= 6) | (jets.pt > 50),
        "tight": (puId >= 7) | (jets.pt > 50),
    }
    pass_jet_puid = np.ones_like(jets.pt.values)
    if jet_puid_opt in ["loose", "medium", "tight"]:
        pass_jet_puid = jet_puid_wps[jet_puid_opt]
    elif "2017corrected" in jet_puid_opt:
        eta_window = (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)
        pass_jet_puid = (eta_window & (puId >= 7)) | (
            (~eta_window) & jet_puid_wps["loose"]
        )
    return pass_jet_puid


def fill_gen_jets(df, output):
    gjets = df.GenJet
    gleptons = df.GenPart[
        (abs(df.GenPart.pdgId) == 13)
        | (abs(df.GenPart.pdgId) == 11)
        | (abs(df.GenPart.pdgId) == 15)
    ]
    gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
    _, _, dr_gl = delta_r(
        gl_pair["jet"].eta,
        gl_pair["lepton"].eta,
        gl_pair["jet"].phi,
        gl_pair["lepton"].phi,
    )
    isolated = ak.all((dr_gl > 0.3), axis=-1)
    gjet1 = ak.to_pandas(gjets[isolated]).loc[
        pd.IndexSlice[:, 0], ["pt", "eta", "phi", "mass"]
    ]
    gjet2 = ak.to_pandas(gjets[isolated]).loc[
        pd.IndexSlice[:, 1], ["pt", "eta", "phi", "mass"]
    ]
    gjet1.index = gjet1.index.droplevel("subentry")
    gjet2.index = gjet2.index.droplevel("subentry")
    gjsum = p4_sum(gjet1, gjet2)
    for var in ["pt", "eta", "phi", "mass"]:
        output[f"gjet1_{var}"] = gjet1[var]
        output[f"gjet2_{var}"] = gjet2[var]
        output[f"gjj_{var}"] = gjsum[var]
    output["gjj_dEta"], output["gjj_dPhi"], output["gjj_dR"] = delta_r(
        output.gjet1_eta, output.gjet2_eta, output.gjet1_phi, output.gjet2_phi
    )
    return output


def gen_jet_pair_mass(df):
    gjmass = None
    gjets = df.GenJet
    gleptons = df.GenPart[
        (abs(df.GenPart.pdgId) == 13)
        | (abs(df.GenPart.pdgId) == 11)
        | (abs(df.GenPart.pdgId) == 15)
    ]
    gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
    _, _, dr_gl = delta_r(
        gl_pair["jet"].eta,
        gl_pair["lepton"].eta,
        gl_pair["jet"].phi,
        gl_pair["lepton"].phi,
    )
    isolated = ak.all((dr_gl > 0.3), axis=-1)
    if ak.count(gjets[isolated], axis=None) > 0:
        # TODO: convert only relevant fields!
        gjet1 = ak.to_pandas(gjets[isolated]).loc[
            pd.IndexSlice[:, 0], ["pt", "eta", "phi", "mass"]
        ]
        gjet2 = ak.to_pandas(gjets[isolated]).loc[
            pd.IndexSlice[:, 1], ["pt", "eta", "phi", "mass"]
        ]
        gjet1.index = gjet1.index.droplevel("subentry")
        gjet2.index = gjet2.index.droplevel("subentry")

        gjsum = p4_sum(gjet1, gjet2)
        gjmass = gjsum.mass
    return gjmass


def fill_softjets(df, output, variables, cutoff):
    saj_df = ak.to_pandas(df.SoftActivityJet)
    saj_df["mass"] = 0.0
    nj_name = f"SoftActivityJetNjets{cutoff}"
    ht_name = f"SoftActivityJetHT{cutoff}"
    res = ak.to_pandas(df[[nj_name, ht_name]])

    res["to_correct"] = output.two_muons | (variables.njets > 0)
    _, _, dR_m1 = delta_r(saj_df.eta, output.mu1_eta, saj_df.phi, output.mu1_phi)
    _, _, dR_m2 = delta_r(saj_df.eta, output.mu2_eta, saj_df.phi, output.mu2_phi)
    _, _, dR_j1 = delta_r(
        saj_df.eta, variables.jet1_eta, saj_df.phi, variables.jet1_phi
    )
    _, _, dR_j2 = delta_r(
        saj_df.eta, variables.jet2_eta, saj_df.phi, variables.jet2_phi
    )
    saj_df["dR_m1"] = dR_m1 < 0.4
    saj_df["dR_m2"] = dR_m2 < 0.4
    saj_df["dR_j1"] = dR_j1 < 0.4
    saj_df["dR_j2"] = dR_j2 < 0.4
    dr_cols = ["dR_m1", "dR_m2", "dR_j1", "dR_j2"]
    saj_df[dr_cols] = saj_df[dr_cols].fillna(False)
    saj_df["to_remove"] = saj_df[dr_cols].sum(axis=1).astype(bool)

    saj_df_filtered = saj_df[(~saj_df.to_remove) & (saj_df.pt > cutoff)]
    footprint = saj_df[(saj_df.to_remove) & (saj_df.pt > cutoff)]
    res["njets_corrected"] = (
        saj_df_filtered.reset_index().groupby("entry")["subentry"].nunique()
    )
    res["njets_corrected"] = res["njets_corrected"].fillna(0).astype(int)
    res["footprint"] = footprint.pt.groupby(level=[0]).sum()
    res["footprint"] = res["footprint"].fillna(0.0)
    res["ht_corrected"] = res[ht_name] - res.footprint
    res.loc[res.ht_corrected < 0, "ht_corrected"] = 0.0

    res.loc[res.to_correct, nj_name] = res.loc[res.to_correct, "njets_corrected"]

    res.loc[res.to_correct, ht_name] = res.loc[res.to_correct, "ht_corrected"]

    variables[f"nsoftjets{cutoff}"] = res[f"SoftActivityJetNjets{cutoff}"]
    variables[f"htsoft{cutoff}"] = res[f"SoftActivityJetHT{cutoff}"]
