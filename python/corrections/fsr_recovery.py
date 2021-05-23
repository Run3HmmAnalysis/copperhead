import numpy as np
import awkward as ak


def fsr_recovery(df):
    mask = (
        (df.Muon.fsrPhotonIdx >= 0) &
        (df.Muon.matched_fsrPhoton.relIso03 < 1.8) &
        (df.Muon.matched_fsrPhoton.dROverEt2 < 0.012) &
        (df.Muon.matched_fsrPhoton.pt / df.Muon.pt < 0.4) &
        (abs(df.Muon.matched_fsrPhoton.eta) < 2.4)
    )
    mask = ak.fill_none(mask, False)

    px = ak.zeros_like(df.Muon.pt)
    py = ak.zeros_like(df.Muon.pt)
    pz = ak.zeros_like(df.Muon.pt)
    e = ak.zeros_like(df.Muon.pt)

    fsr = {
        "pt": df.Muon.matched_fsrPhoton.pt,
        "eta": df.Muon.matched_fsrPhoton.eta,
        "phi": df.Muon.matched_fsrPhoton.phi,
        "mass": 0.
    }

    for obj in [df.Muon, fsr]:
        px_ = obj["pt"] * np.cos(obj["phi"])
        py_ = obj["pt"] * np.sin(obj["phi"])
        pz_ = obj["pt"] * np.sinh(obj["eta"])
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj["mass"]**2)

        px = px + px_
        py = py + py_
        pz = pz + pz_
        e = e + e_

    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    iso = (
        df.Muon.pfRelIso04_all * df.Muon.pt - df.Muon.matched_fsrPhoton.pt
    ) / pt

    df['Muon', 'pt_fsr'] = ak.where(mask, pt, df.Muon.pt)
    df['Muon', 'eta_fsr'] = ak.where(mask, eta, df.Muon.eta)
    df['Muon', 'phi_fsr'] = ak.where(mask, phi, df.Muon.phi)
    df['Muon', 'mass_fsr'] = ak.where(mask, mass, df.Muon.mass)
    df['Muon', 'iso_fsr'] = ak.where(mask, iso, df.Muon.pfRelIso04_all)
    return mask
