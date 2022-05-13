import numpy as np

from python.math_tools import p4_sum, delta_r, cs_variables


def fill_muons(processor, output, mu1, mu2, is_mc):
    mu1_variable_names = ["mu1_pt", "mu1_pt_over_mass", "mu1_eta", "mu1_phi", "mu1_iso"]
    mu2_variable_names = ["mu2_pt", "mu2_pt_over_mass", "mu2_eta", "mu2_phi", "mu2_iso"]
    dimuon_variable_names = [
        "dimuon_mass",
        "dimuon_ebe_mass_res",
        "dimuon_ebe_mass_res_rel",
        "dimuon_pt",
        "dimuon_pt_log",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_dEta",
        "dimuon_dPhi",
        "dimuon_dR",
        "dimuon_rap",
        "dimuon_cos_theta_cs",
        "dimuon_phi_cs",
    ]
    v_names = mu1_variable_names + mu2_variable_names + dimuon_variable_names

    # Initialize columns for muon variables
    for n in v_names:
        output[n] = 0.0

    # Fill single muon variables
    for v in ["pt", "ptErr", "eta", "phi"]:
        output[f"mu1_{v}"] = mu1[v]
        output[f"mu2_{v}"] = mu2[v]

    output["mu1_iso"] = mu1.pfRelIso04_all
    output["mu2_iso"] = mu2.pfRelIso04_all
    output["mu1_pt_over_mass"] = output.mu1_pt / output.dimuon_mass
    output["mu2_pt_over_mass"] = output.mu2_pt / output.dimuon_mass

    # Fill dimuon variables
    mm = p4_sum(mu1, mu2)
    for v in ["pt", "eta", "phi", "mass", "rap"]:
        name = f"dimuon_{v}"
        output[name] = mm[v]
        output[name] = output[name].fillna(-999.0)

    output["dimuon_pt_log"] = np.log(output.dimuon_pt)

    mm_deta, mm_dphi, mm_dr = delta_r(mu1.eta, mu2.eta, mu1.phi, mu2.phi)

    output["dimuon_dEta"] = mm_deta
    output["dimuon_dPhi"] = mm_dphi
    output["dimuon_dR"] = mm_dr

    output["dimuon_ebe_mass_res"] = mass_resolution(
        is_mc, processor.evaluator, output, processor.year
    )
    output["dimuon_ebe_mass_res_rel"] = output.dimuon_ebe_mass_res / output.dimuon_mass

    output["dimuon_pisa_mass_res_rel"] = mass_resolution_pisa(
        processor.evaluator, output
    )
    output["dimuon_pisa_mass_res"] = (
        output.dimuon_pisa_mass_res_rel * output.dimuon_mass
    )

    output["dimuon_cos_theta_cs"], output["dimuon_phi_cs"] = cs_variables(mu1, mu2)


def mass_resolution(is_mc, evaluator, df, year):
    # Returns absolute mass resolution!
    dpt1 = (df.mu1_ptErr * df.dimuon_mass) / (2 * df.mu1_pt)
    dpt2 = (df.mu2_ptErr * df.dimuon_mass) / (2 * df.mu2_pt)

    if is_mc:
        label = f"res_calib_MC_{year}"
    else:
        label = f"res_calib_Data_{year}"
    calibration = np.array(
        evaluator[label](
            df.mu1_pt.values, abs(df.mu1_eta.values), abs(df.mu2_eta.values)
        )
    )

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration


def mass_resolution_pisa(evaluator, df):
    # Returns relative mass resolution!
    mu1_ptErr = evaluator["PtErrParametrization"](
        np.log(df.mu1_pt).values, np.abs(df.mu1_eta).values
    )
    mu2_ptErr = evaluator["PtErrParametrization"](
        np.log(df.mu2_pt).values, np.abs(df.mu2_eta).values
    )
    return np.sqrt(0.5 * (mu1_ptErr * mu1_ptErr + mu2_ptErr * mu2_ptErr))
