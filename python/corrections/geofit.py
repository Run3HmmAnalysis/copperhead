import numpy as np
import awkward as ak


def apply_geofit(df, year, mask):
    d0_BS_charge = np.multiply(df.Muon.dxybs, df.Muon.charge)
    mask = mask & (np.abs(d0_BS_charge) < 999999.)

    pt = df.Muon.pt
    eta = df.Muon.eta

    cuts = {
        'eta_1': (np.abs(eta) < 0.9),
        'eta_2': ((np.abs(eta) < 1.7) & (np.abs(eta) >= 0.9)),
        'eta_3': (np.abs(eta) >= 1.7)
    }

    factors = {
        '2016': {
            'eta_1': 411.34,
            'eta_2': 673.40,
            'eta_3': 1099.0,
        },
        '2017': {
            'eta_1': 582.32,
            'eta_2': 974.05,
            'eta_3': 1263.4,
        },
        '2018': {
            'eta_1': 650.84,
            'eta_2': 988.37,
            'eta_3': 1484.6,
        }
    }
    pt_corr = pt
    for eta_i in ['eta_1', 'eta_2', 'eta_3']:
        value = (
            factors[year][eta_i] *
            d0_BS_charge * pt * pt / 10000.0
        )
        pt_corr = ak.where(cuts[eta_i], value, pt_corr)
    df['Muon', 'pt_gf'] = ak.where(mask, pt - pt_corr, pt)
    return


def geofit_evaluator(muons_pt, muons_eta, muons_dxybs,
                     muons_charge, year, mask):
    pt_cor = np.zeros(len(muons_pt.flatten()), dtype=float)
    d0_BS_charge_full = np.multiply(muons_dxybs.flatten(),
                                    muons_charge.flatten())
    passes_mask = mask & (np.abs(d0_BS_charge_full) < 999999.)
    d0_BS_charge = d0_BS_charge_full[passes_mask]
    pt = muons_pt.flatten()[passes_mask]
    eta = muons_eta.flatten()[passes_mask]

    pt_cor_mask = pt_cor[passes_mask]

    cuts = {
        'eta_1': (np.abs(eta) < 0.9),
        'eta_2': ((np.abs(eta) < 1.7) & (np.abs(eta) >= 0.9)),
        'eta_3': (np.abs(eta) >= 1.7)
    }

    factors = {
        '2016': {
            'eta_1': 411.34,
            'eta_2': 673.40,
            'eta_3': 1099.0,
        },
        '2017': {
            'eta_1': 582.32,
            'eta_2': 974.05,
            'eta_3': 1263.4,
        },
        '2018': {
            'eta_1': 650.84,
            'eta_2': 988.37,
            'eta_3': 1484.6,
        }
    }

    for eta_i in ['eta_1', 'eta_2', 'eta_3']:
        pt_cor_mask[cuts[eta_i]] = factors[year][eta_i] *\
            d0_BS_charge[cuts[eta_i]] * pt[cuts[eta_i]] *\
            pt[cuts[eta_i]] / 10000.0
    pt_cor[passes_mask] = pt_cor_mask
    return (muons_pt.flatten() - pt_cor)
