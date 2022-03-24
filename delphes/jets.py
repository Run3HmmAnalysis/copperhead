import numpy as np
from python.math_tools import p4_sum, delta_r, rapidity
import awkward as ak
import pandas as pd


def fill_jets(output, jet1, jet2):
    variable_names = [
        "jet1_pt",
        "jet1_eta",
        "jet1_rap",
        "jet1_phi",
        "jet2_pt",
        "jet2_eta",
        "jet2_rap",
        "jet2_phi",
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
    ]

    for v in variable_names:
        output[v] = -999.0

    # Fill single jet variables
    for v in ["pt", "eta", "phi"]:
        output[f"jet1_{v}"] = jet1[v]
        output[f"jet2_{v}"] = jet2[v]

    output.jet1_rap = rapidity(jet1)
    output.jet2_rap = rapidity(jet2)

    # Fill dijet variables
    jj = p4_sum(jet1, jet2)
    for v in ["pt", "eta", "phi", "mass"]:
        output[f"jj_{v}"] = jj[v]

    output.jj_mass_log = np.log(output.jj_mass)

    output.jj_dEta, output.jj_dPhi, _ = delta_r(
        output.jet1_eta, output.jet2_eta, output.jet1_phi, output.jet2_phi
    )

    # Fill dimuon-dijet system variables
    mm_columns = ["dimuon_pt", "dimuon_eta", "dimuon_phi", "dimuon_mass"]
    jj_columns = ["jj_pt", "jj_eta", "jj_phi", "jj_mass"]

    dimuons = output.loc[:, mm_columns]
    dijets = output.loc[:, jj_columns]

    # careful with renaming
    dimuons.columns = ["mass", "pt", "eta", "phi"]
    dijets.columns = ["pt", "eta", "phi", "mass"]

    mmjj = p4_sum(dimuons, dijets)
    for v in ["pt", "eta", "phi", "mass"]:
        output[f"mmjj_{v}"] = mmjj[v]

    output.zeppenfeld = output.dimuon_eta - 0.5 * (output.jet1_eta + output.jet2_eta)
    output.rpt = output.mmjj_pt / (output.dimuon_pt + output.jet1_pt + output.jet2_pt)
    ll_ystar = output.dimuon_rap - (output.jet1_rap + output.jet2_rap) / 2
    ll_zstar = abs(ll_ystar / (output.jet1_rap - output.jet2_rap))

    output.ll_zstar_log = np.log(ll_zstar)

    output.mmj1_dEta, output.mmj1_dPhi, output.mmj1_dR = delta_r(
        output.dimuon_eta, output.jet1_eta, output.dimuon_phi, output.jet1_phi
    )

    output.mmj2_dEta, output.mmj2_dPhi, output.mmj2_dR = delta_r(
        output.dimuon_eta, output.jet2_eta, output.dimuon_phi, output.jet2_phi
    )

    output.mmj_min_dEta = np.where(
        output.mmj1_dEta, output.mmj2_dEta, (output.mmj1_dEta < output.mmj2_dEta)
    )

    output.mmj_min_dPhi = np.where(
        output.mmj1_dPhi, output.mmj2_dPhi, (output.mmj1_dPhi < output.mmj2_dPhi)
    )

    output[variable_names] = output[variable_names].fillna(-999.0)


def fill_gen_jets(df, output):
    features = ["PT", "Eta", "Phi", "Mass"]
    gjets = df.GenJet[features]
    print(df.GenJet.fields)
    gleptons = df.MuonMedium
    # gleptons = df.GenPart[
    #    (abs(df.GenPart.pdgId) == 13)
    #    | (abs(df.GenPart.pdgId) == 11)
    #    | (abs(df.GenPart.pdgId) == 15)
    # ]
    gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
    _, _, dr_gl = delta_r(
        gl_pair["jet"].Eta,
        gl_pair["lepton"].Eta,
        gl_pair["jet"].Phi,
        gl_pair["lepton"].Phi,
    )
    isolated = ak.all((dr_gl > 0.3), axis=-1)

    gjet1 = ak.to_pandas(gjets[isolated]).loc[pd.IndexSlice[:, 0], features]
    gjet2 = ak.to_pandas(gjets[isolated]).loc[pd.IndexSlice[:, 1], features]
    gjet1.index = gjet1.index.droplevel("subentry")
    gjet2.index = gjet2.index.droplevel("subentry")
    feat_map = {"pt": "PT", "eta": "Eta", "phi": "Phi", "mass": "Mass"}
    for var in ["pt", "eta", "phi", "mass"]:
        gjet1[var] = gjet1[feat_map[var]]
        gjet2[var] = gjet2[feat_map[var]]
    gjsum = p4_sum(gjet1, gjet2)

    for var in ["pt", "eta", "phi", "mass"]:
        output[f"gjet1_{var}"] = gjet1[var]
        output[f"gjet2_{var}"] = gjet2[var]
        output[f"gjj_{var}"] = gjsum[var]
    output["gjj_dEta"], output["gjj_dPhi"], output["gjj_dR"] = delta_r(
        output.gjet1_eta, output.gjet2_eta, output.gjet1_phi, output.gjet2_phi
    )
    return output
