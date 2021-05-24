import pandas as pd
import awkward as ak
import uproot

from coffea.lookup_tools import dense_lookup


def musf_lookup(parameters):
    mu_id_vals = 0
    mu_id_err = 0
    mu_iso_vals = 0
    mu_iso_err = 0
    mu_trig_vals_data = 0
    mu_trig_err_data = 0
    mu_trig_vals_mc = 0
    mu_trig_err_mc = 0

    for scaleFactors in parameters['muSFFileList']:
        id_file = uproot.open(scaleFactors['id'][0])
        iso_file = uproot.open(scaleFactors['iso'][0])
        trig_file = uproot.open(scaleFactors['trig'][0])
        mu_id_vals +=\
            id_file[scaleFactors['id'][1]].values() *\
            scaleFactors['scale']
        mu_id_err +=\
            id_file[scaleFactors['id'][1]].variances()**0.5 *\
            scaleFactors['scale']
        mu_id_edges = [
            id_file[scaleFactors['id'][1]].axis(0).edges(),
            id_file[scaleFactors['id'][1]].axis(1).edges()
        ]
        mu_iso_vals +=\
            iso_file[scaleFactors['iso'][1]].values() *\
            scaleFactors['scale']
        mu_iso_err +=\
            iso_file[scaleFactors['iso'][1]].variances()**0.5 *\
            scaleFactors['scale']
        mu_iso_edges = [
            iso_file[scaleFactors['iso'][1]].axis(0).edges(),
            iso_file[scaleFactors['iso'][1]].axis(1).edges()
        ]
        mu_trig_vals_data +=\
            trig_file[scaleFactors['trig'][1]].values() *\
            scaleFactors['scale']
        mu_trig_vals_mc +=\
            trig_file[scaleFactors['trig'][2]].values() *\
            scaleFactors['scale']
        mu_trig_err_data +=\
            trig_file[scaleFactors['trig'][1]].variances()**0.5 *\
            scaleFactors['scale']
        mu_trig_err_mc +=\
            trig_file[scaleFactors['trig'][2]].variances()**0.5 *\
            scaleFactors['scale']
        mu_trig_edges = [
            trig_file[scaleFactors['trig'][1]].axis(0).edges(),
            trig_file[scaleFactors['trig'][1]].axis(1).edges()
        ]

    mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
    mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
    mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
    mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)

    mu_trig_eff_data = dense_lookup.dense_lookup(
        mu_trig_vals_data,
        mu_trig_edges
    )
    mu_trig_eff_mc = dense_lookup.dense_lookup(
        mu_trig_vals_mc,
        mu_trig_edges
    )
    mu_trig_err_data = dense_lookup.dense_lookup(
        mu_trig_err_data,
        mu_trig_edges
    )
    mu_trig_err_mc = dense_lookup.dense_lookup(
        mu_trig_err_mc,
        mu_trig_edges
    )

    return {
        'mu_id_sf': mu_id_sf,
        'mu_id_err': mu_id_err,
        'mu_iso_sf': mu_iso_sf,
        'mu_iso_err': mu_iso_err,
        'mu_trig_eff_data': mu_trig_eff_data,
        'mu_trig_eff_mc': mu_trig_eff_mc,
        'mu_trig_err_data': mu_trig_err_data,
        'mu_trig_err_mc': mu_trig_err_mc
    }


def musf_evaluator(lookups, year, numevents, mu1, mu2):
    sf = pd.DataFrame(
        index=mu1.index,
        columns=[
            'muID_nom', 'muID_up', 'muID_down',
            'muIso_nom', 'muIso_up', 'muIso_down',
            'muTrig_nom', 'muTrig_up', 'muTrig_down',
        ]
    )
    sf = sf.fillna(1.0)

    for how in ['nom', 'up', 'down']:
        sf[f'trig_num_{how}'] = 1.0
        sf[f'trig_denom_{how}'] = 1.0

    for mu in [mu1, mu2]:
        pt = mu.pt_raw.values
        eta = mu.eta_raw.values
        abs_eta = abs(mu.eta_raw.values)

        if '2016' in year:
            muID_ = lookups['mu_id_sf'](eta, pt)
            muIso_ = lookups['mu_iso_sf'](eta, pt)
            muIDerr = lookups['mu_id_err'](eta, pt)
            muIsoerr = lookups['mu_iso_err'](eta, pt)
        else:
            muID_ = lookups['mu_id_sf'](pt, abs_eta)
            muIso_ = lookups['mu_iso_sf'](pt, abs_eta)
            muIDerr = lookups['mu_id_err'](pt, abs_eta)
            muIsoerr = lookups['mu_iso_err'](pt, abs_eta)

        muTrig_data = lookups['mu_trig_eff_data'](abs_eta, pt)
        muTrig_mc = lookups['mu_trig_eff_mc'](abs_eta, pt)
        muTrigerr_data = lookups['mu_trig_err_data'](abs_eta, pt)
        muTrigerr_mc = lookups['mu_trig_err_mc'](abs_eta, pt)

        sf['trig_num_nom'] *= (1. - ak.to_numpy(muTrig_data))
        sf['trig_num_up'] *= (
            1. - ak.to_numpy(muTrig_data - muTrigerr_data)
        )
        sf['trig_num_down'] *= (
            1. - ak.to_numpy(muTrig_data + muTrigerr_data)
        )
        sf['trig_denom_nom'] *= (1. - ak.to_numpy(muTrig_mc))
        sf['trig_denom_up'] *= (
            1. - ak.to_numpy(muTrig_mc - muTrigerr_mc)
        )
        sf['trig_denom_down'] *= (
            1. - ak.to_numpy(muTrig_mc + muTrigerr_mc)
        )

        sf['muID_nom'] *= ak.to_numpy(muID_)
        sf['muID_up'] *= ak.to_numpy(muID_ + muIDerr)
        sf['muID_down'] *= ak.to_numpy(muID_ - muIDerr)
        sf['muIso_nom'] *= ak.to_numpy(muIso_)
        sf['muIso_up'] *= ak.to_numpy(muIso_ + muIsoerr)
        sf['muIso_down'] *= ak.to_numpy(muIso_ - muIsoerr)

    for how in ['nom', 'up', 'down']:
        sf[f'trig_num_{how}'] = (1 - sf[f'trig_num_{how}'])
        sf[f'trig_denom_{how}'] = (1 - sf[f'trig_denom_{how}'])
        cut = (sf[f'trig_denom_{how}'] != 0)
        sf.loc[cut, f'muTrig_{how}'] = (
            sf.loc[cut, f'trig_num_{how}'] /
            sf.loc[cut, f'trig_denom_{how}']
        )
    muID = {
        'nom': sf['muID_nom'],
        'up': sf['muID_up'],
        'down': sf['muID_down']
    }
    muIso = {
        'nom': sf['muIso_nom'],
        'up': sf['muIso_up'],
        'down': sf['muIso_down']
    }
    muTrig = {
        'nom': sf['muTrig_nom'],
        'up': sf['muTrig_up'],
        'down': sf['muTrig_down']
    }

    return muID, muIso, muTrig
