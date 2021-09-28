import numpy as np

from python.utils import p4_sum, delta_r, cs_variables


def fill_muons(processor, output, mu1, mu2):
    mu1_variable_names = [
        'mu1_pt', 'mu1_pt_over_mass',
        'mu1_eta', 'mu1_phi', 'mu1_iso'
    ]
    mu2_variable_names = [
        'mu2_pt', 'mu2_pt_over_mass',
        'mu2_eta', 'mu2_phi', 'mu2_iso'
    ]
    dimuon_variable_names = [
        'dimuon_mass',
        #'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel',
        'dimuon_pt', 'dimuon_pt_log',
        'dimuon_eta', 'dimuon_phi',
        'dimuon_dEta', 'dimuon_dPhi',
        'dimuon_dR', 'dimuon_rap',
        'dimuon_cos_theta_cs', 'dimuon_phi_cs'
    ]
    v_names = (
        mu1_variable_names +
        mu2_variable_names +
        dimuon_variable_names
    )

    # Initialize columns for muon variables
    for n in (v_names):
        output[n] = 0.0

    vmap = {
        'pt': 'PT',
        'eta': 'Eta',
        'phi': 'Phi',
        'charge': 'Charge'
    }
    # Fill single muon variables
    for v in ['pt', 'eta', 'phi', 'charge']:
        mu1[v] = mu1[vmap[v]]
        mu2[v] = mu2[vmap[v]]
        output[f'mu1_{v}'] = mu1[v]
        output[f'mu2_{v}'] = mu2[v]
    mu1['mass'] = mu2['mass'] = 0.10566

    # Fill dimuon variables
    mm = p4_sum(mu1, mu2)
    for v in ['pt', 'eta', 'phi', 'mass', 'rap']:
        name = f'dimuon_{v}'
        output[name] = mm[v]
        output[name] = output[name].fillna(-999.)

    output['dimuon_pt_log'] = np.log(output.dimuon_pt)

    mm_deta, mm_dphi, mm_dr = delta_r(
        mu1.eta, mu2.eta,
        mu1.phi, mu2.phi
    )

    output['dimuon_dEta'] = mm_deta
    output['dimuon_dPhi'] = mm_dphi
    output['dimuon_dR'] = mm_dr

    output['dimuon_cos_theta_cs'],\
        output['dimuon_phi_cs'] = cs_variables(mu1, mu2)


