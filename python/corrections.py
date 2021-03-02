import numpy as np
import pandas as pd
import awkward as ak
import awkward
import uproot

from coffea.lookup_tools import dense_lookup


class NNLOPS_Evaluator(object):
    def __init__(self, input_path):
        with uproot.open(input_path) as f:
            self.ratio_0jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_0jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_0jet"]}
            self.ratio_1jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_1jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_1jet"]}
            self.ratio_2jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_2jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_2jet"]}
            self.ratio_3jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_3jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_3jet"]}

    def evaluate(self, hig_pt, njets, mode):
        result = np.ones(len(hig_pt), dtype=float)
        hig_pt = np.array(hig_pt)
        njets = np.array(njets)
        result[njets == 0] = np.interp(
            np.minimum(hig_pt[njets == 0], 125.),
            self.ratio_0jet[mode]._fX,
            self.ratio_0jet[mode]._fY)
        result[njets == 1] = np.interp(
            np.minimum(hig_pt[njets == 1], 625.),
            self.ratio_1jet[mode]._fX,
            self.ratio_1jet[mode]._fY)
        result[njets == 2] = np.interp(
            np.minimum(hig_pt[njets == 2], 800.),
            self.ratio_2jet[mode]._fX,
            self.ratio_2jet[mode]._fY)
        result[njets > 2] = np.interp(
            np.minimum(hig_pt[njets > 2], 925.),
            self.ratio_3jet[mode]._fX,
            self.ratio_3jet[mode]._fY)
        return result


def apply_roccor(df, rochester, is_mc):
    if is_mc:
        hasgen = ~np.isnan(ak.fill_none(df.Muon.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(
            *ak.to_numpy(ak.flatten(df.Muon.pt)).shape
        )
        mc_rand = ak.unflatten(mc_rand, ak.num(df.Muon.pt, axis=1))

        corrections = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))
        # errors = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))

        mc_kspread = rochester.kSpreadMC(
                        df.Muon.charge[hasgen],
                        df.Muon.pt[hasgen],
                        df.Muon.eta[hasgen],
                        df.Muon.phi[hasgen],
                        df.Muon.matched_gen.pt[hasgen])
        mc_ksmear = rochester.kSmearMC(
                        df.Muon.charge[~hasgen],
                        df.Muon.pt[~hasgen],
                        df.Muon.eta[~hasgen],
                        df.Muon.phi[~hasgen],
                        df.Muon.nTrackerLayers[~hasgen],
                        mc_rand[~hasgen])
        # TODO: fix errors
        # errspread = rochester.kSpreadMCerror(
        #                 df.Muon.charge[hasgen],
        #                 df.Muon.pt[hasgen],
        #                 df.Muon.eta[hasgen],
        #                 df.Muon.phi[hasgen],
        #                 df.Muon.matched_gen.pt[hasgen])
        # errsmear = rochester.kSmearMCerror(
        #                 df.Muon.charge[~hasgen],
        #                 df.Muon.pt[~hasgen],
        #                 df.Muon.eta[~hasgen],
        #                 df.Muon.phi[~hasgen],
        #                 df.Muon.nTrackerLayers[~hasgen],
        #                 mc_rand[~hasgen])
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        # errors[hasgen.flatten()] = np.array(ak.flatten(errspread))
        # errors[~hasgen.flatten()] = np.array(ak.flatten(errsmear))

        corrections = ak.unflatten(
            corrections, ak.num(df.Muon.pt, axis=1)
        )

    else:
        corrections = rochester.kScaleDT(
                        df.Muon.charge,
                        df.Muon.pt,
                        df.Muon.eta,
                        df.Muon.phi
        )
        # errors = rochester.kScaleDTerror(
        #                 df.Muon.charge,
        #                 df.Muon.pt,
        #                 df.Muon.eta,
        #                 df.Muon.phi)

    df['Muon', 'pt_roch'] = (
        df.Muon.pt * corrections
    )
    # df['Muon', 'pt_roch_up'] = df.Muon.pt_roch + df.Muon.pt*error
    # df['Muon', 'pt_roch_down'] = df.Muon.pt_roch - df.Muon.pt*error


# awkward0 implementation!
def roccor_evaluator(rochester, is_mc, muons):
    if is_mc:
        mc_rand = np.random.rand(*muons.pt.flatten().shape)
        mc_rand = awkward.JaggedArray.fromoffsets(
                        muons.pt.offsets,
                        mc_rand)

        # hasgen = ~np.isnan(
        #    copy.deepcopy(muons.matched_gen.pt).fillna(np.nan))
        hasgen = ~np.isnan(muons.matched_gen.pt.fillna(np.nan))
        mc_rand = awkward.JaggedArray.fromoffsets(
                    hasgen.offsets,
                    mc_rand)._content

        corrections = np.ones_like(muons.pt.flatten())
        errors = np.ones_like(muons.pt.flatten())

        mc_kspread = rochester.kSpreadMC(
                        muons.charge[hasgen],
                        muons.pt[hasgen],
                        muons.eta[hasgen],
                        muons.phi[hasgen],
                        muons.matched_gen.pt[hasgen])
        mc_ksmear = rochester.kSmearMC(
                        muons.charge[~hasgen],
                        muons.pt[~hasgen],
                        muons.eta[~hasgen],
                        muons.phi[~hasgen],
                        muons.nTrackerLayers[~hasgen],
                        mc_rand[~hasgen])
        errspread = rochester.kSpreadMCerror(
                        muons.charge[hasgen],
                        muons.pt[hasgen],
                        muons.eta[hasgen],
                        muons.phi[hasgen],
                        muons.matched_gen.pt[hasgen])
        errsmear = rochester.kSmearMCerror(
                        muons.charge[~hasgen],
                        muons.pt[~hasgen],
                        muons.eta[~hasgen],
                        muons.phi[~hasgen],
                        muons.nTrackerLayers[~hasgen],
                        mc_rand[~hasgen])

        corrections[hasgen.flatten()] = mc_kspread.flatten()
        corrections[~hasgen.flatten()] = mc_ksmear.flatten()
        errors[hasgen.flatten()] = errspread.flatten()
        errors[~hasgen.flatten()] = errsmear.flatten()

    else:
        corrections = rochester.kScaleDT(
                        muons.charge,
                        muons.pt,
                        muons.eta,
                        muons.phi)
        errors = rochester.kScaleDTerror(
                        muons.charge,
                        muons.pt,
                        muons.eta,
                        muons.phi)
    corrections_jagged = awkward.JaggedArray.fromcounts(
                        muons.counts,
                        corrections.flatten())
    errors_jagged = awkward.JaggedArray.fromcounts(
                        muons.counts,
                        errors.flatten())
    return corrections_jagged, errors_jagged


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
                            mu_trig_edges)
    mu_trig_eff_mc = dense_lookup.dense_lookup(
                            mu_trig_vals_mc,
                            mu_trig_edges)
    mu_trig_err_data = dense_lookup.dense_lookup(
                            mu_trig_err_data,
                            mu_trig_edges)
    mu_trig_err_mc = dense_lookup.dense_lookup(
                            mu_trig_err_mc,
                            mu_trig_edges)

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
    # TODO: fix implementation of trig SF
    sf = pd.DataFrame(
        index=mu1.index,
        columns=[
            'muID', 'muID_up', 'muID_down',
            'muIso', 'muIso_up', 'muIso_down',
            'muTrig', 'muTrig_up', 'muTrig_down',
        ]
    )
    sf = sf.fillna(1.0)

    for mu in [mu1, mu2]:
        pt = mu.pt_raw.values
        eta = mu.eta_raw.values
        abs_eta = abs(mu.eta_raw.values)

        if '2016' in year:
            muID_ = lookups['mu_id_sf'](eta, pt)
            muIso_ = lookups['mu_iso_sf'](eta, pt)
            muIDerr = lookups['mu_id_err'](eta, pt)
            muIsoerr = lookups['mu_iso_err'](eta, pt)
            # muTrig_data = lookups['mu_trig_eff_data'](abs_eta, pt)
            # muTrig_mc = lookups['mu_trig_eff_mc'](abs_eta, pt)
            # muTrigerr_data = lookups['mu_trig_err_data'](abs_eta, pt)
            # muTrigerr_mc = lookups['mu_trig_err_mc'](abs_eta, pt)
        else:
            muID_ = lookups['mu_id_sf'](pt, abs_eta)
            muIso_ = lookups['mu_iso_sf'](pt, abs_eta)
            muIDerr = lookups['mu_id_err'](pt, abs_eta)
            muIsoerr = lookups['mu_iso_err'](pt, abs_eta)
            # muTrig_data = lookups['mu_trig_eff_data'](abs_eta, pt)
            # muTrig_mc = lookups['mu_trig_eff_mc'](abs_eta, pt)
            # muTrigerr_data = lookups['mu_trig_err_data'](abs_eta, pt)
            # muTrigerr_mc = lookups['mu_trig_err_mc'](abs_eta, pt)

        # denom = ((1 - (1. - muTrig_mc).prod()))
        # denom_up = ((1 - (1. - muTrig_mc - muTrigerr_mc).prod()) != 0)
        # denom_dn = ((1 - (1. - muTrig_mc + muTrigerr_mc).prod()) != 0)

        # muTrig[denom != 0] = (
        #     (1 - (1. - muTrig_data).prod())[denom != 0] / denom[denom != 0])
        # muTrig_up[denom_up != 0] = (
        #     (1 - (1. - muTrig_data - muTrigerr_data).prod())[denom_up != 0] /
        #     denom_up[denom_up != 0])
        # muTrig_down[denom_dn != 0] = (
        #     (1 - (1. - muTrig_data + muTrigerr_data).prod())[denom_dn != 0] /
        #     denom_dn[denom_dn != 0])

        sf['muID'] = sf['muID']*ak.to_numpy(muID_)
        sf['muID_up'] = sf['muID_up']*ak.to_numpy(muID_ + muIDerr)
        sf['muID_down'] = sf['muID_down']*ak.to_numpy(muID_ - muIDerr)
        sf['muIso'] = sf['muIso']*ak.to_numpy(muIso_)
        sf['muIso_up'] = sf['muIso_up']*ak.to_numpy(muIso_ + muIsoerr)
        sf['muIso_down'] = sf['muIso_down']*ak.to_numpy(muIso_ - muIsoerr)

    return sf


def pu_lookup(parameters, mode='nom', auto=[]):
    if mode == 'nom':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup'].values()
    elif mode == 'up':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup_plus'].values()
    elif mode == 'down':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup_minus'].values()
    else:
        print("PU lookup: incorrect mode ", mode)
        return

    nbins = len(pu_hist_data)
    edges = [[i for i in range(nbins)]]
    # pu_hist_mc = load("data/pileup/pisa_lookup_2018.coffea")(range(102))
    if len(auto) == 0:
        pu_hist_mc = uproot.open(parameters['pu_file_mc'])['pu_mc'].values()
    else:
        pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]

    lookup = dense_lookup.dense_lookup(
                pu_reweight(pu_hist_data, pu_hist_mc),
                edges)
    lookup._axes = lookup._axes[0]
    return lookup


def pu_reweight(pu_hist_data, pu_hist_mc):
    pu_arr_mc_ = np.zeros(len(pu_hist_mc))
    for ibin, value in enumerate(pu_hist_mc):
        pu_arr_mc_[ibin] = max(value, 0)

    pu_arr_data = np.zeros(len(pu_hist_data))
    for ibin, value in enumerate(pu_hist_data):
        pu_arr_data[ibin] = max(value, 0)

    pu_arr_mc_ref = pu_arr_mc_
    pu_arr_mc = pu_arr_mc_ / pu_arr_mc_.sum()
    pu_arr_data = pu_arr_data / pu_arr_data.sum()

    weights = np.ones(len(pu_hist_mc))
    weights[pu_arr_mc != 0] =\
        pu_arr_data[pu_arr_mc != 0] / pu_arr_mc[pu_arr_mc != 0]
    maxw = min(weights.max(), 5.)
    cropped = []
    while (maxw > 3):
        cropped = []
        for i in range(len(weights)):
            cropped.append(min(maxw, weights[i]))
        shift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        if(abs(shift) > 0.0025):
            break
        maxw *= 0.95

    maxw /= 0.95
    if (len(cropped) > 0):
        for i in range(len(weights)):
            cropped[i] = min(maxw, weights[i])
        normshift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        for i in range(len(weights)):
            weights[i] = cropped[i] * (1 - normshift)
    return weights


def checkIntegral(wgt1, wgt2, ref):
    myint = 0
    refint = 0
    for i in range(len(wgt1)):
        myint += wgt1[i] * ref[i]
        refint += wgt2[i] * ref[i]
    return (myint - refint) / refint


def pu_evaluator(lookup, numevents, ntrueint):
    pu_weight = np.ones(numevents)
    pu_weight = lookup(ntrueint)
    pu_weight = np.array(pu_weight)
    pu_weight[ntrueint > 100] = 1
    pu_weight[ntrueint < 1] = 1
    return pu_weight


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


def btag_weights(processor, lookup, systs, jets,
                 weights, bjet_sel_mask, numevents):

    btag = pd.DataFrame(index=bjet_sel_mask.index)
    jets = jets[abs(jets.eta) < 2.4]
    jets.loc[jets.pt > 1000., 'pt'] = 1000.

    jets['btag_wgt'] = lookup.eval(
        'central',
        jets.hadronFlavour.values,
        abs(jets.eta.values),
        jets.pt.values,
        jets.btagDeepB.values,
        True
    )
    btag['wgt'] = jets['btag_wgt'].prod(level=0)
    btag['wgt'] = btag['wgt'].fillna(1.0)
    btag.loc[btag.wgt < 0.01, 'wgt'] = 1.

    flavors = {
        0: ["jes", "hf", "lfstats1", "lfstats2"],
        1: ["jes", "hf", "lfstats1", "lfstats2"],
        2: ["jes", "hf", "lfstats1", "lfstats2"],
        3: ["jes", "hf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "lf", "hfstats1", "hfstats2"],
        21: ["jes", "hf", "lfstats1", "lfstats2"],
    }

    btag_syst = {}
    for sys in systs:
        jets[f'btag_{sys}_up'] = 1.0
        jets[f'btag_{sys}_down'] = 1.0
        btag[f'{sys}_up'] = 1.0
        btag[f'{sys}_down'] = 1.0

        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = (abs(jets.hadronFlavour) == f)
                jets.loc[btag_mask, f'btag_{sys}_up'] = lookup.eval(
                    f'up_{sys}',
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepB[btag_mask].values,
                    True
                )
                jets.loc[btag_mask, f'btag_{sys}_down'] = lookup.eval(
                    f'down_{sys}',
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepB[btag_mask].values,
                    True
                )

        btag[f'{sys}_up'] = jets[f'btag_{sys}_up'].prod(level=0)
        btag[f'{sys}_down'] = jets[f'btag_{sys}_down'].prod(level=0)
        btag_syst[sys] = [
            btag[f'{sys}_up'],
            btag[f'{sys}_down']
        ]

    sum_before = weights.df['nominal'][bjet_sel_mask].sum()
    sum_after = weights.df['nominal'][bjet_sel_mask].multiply(
        btag.wgt[bjet_sel_mask], axis=0).sum()
    btag.wgt = btag.wgt * sum_before / sum_after

    return btag.wgt, btag_syst


def puid_weights(evaluator, year, jets, pt_name,
                 jet_puid_opt, jet_puid, numevents):
    if "2017corrected" in jet_puid_opt:
        h_eff_name_L = f"h2_eff_mc{year}_L"
        h_sf_name_L = f"h2_eff_sf{year}_L"
        h_eff_name_T = f"h2_eff_mc{year}_T"
        h_sf_name_T = f"h2_eff_sf{year}_T"
        puid_eff_L = evaluator[h_eff_name_L](jets[pt_name], jets.eta)
        puid_sf_L = evaluator[h_sf_name_L](jets[pt_name], jets.eta)
        puid_eff_T = evaluator[h_eff_name_T](jets[pt_name], jets.eta)
        puid_sf_T = evaluator[h_sf_name_T](jets[pt_name], jets.eta)

        jets_passed_L = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & jet_puid &\
            ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        jets_failed_L = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & (~jet_puid) &\
            ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        jets_passed_T = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & jet_puid &\
            ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        jets_failed_T = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & (~jet_puid) &\
            ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))

        pMC_L = puid_eff_L[jets_passed_L].prod() *\
            (1. - puid_eff_L[jets_failed_L]).prod()
        pMC_T = puid_eff_T[jets_passed_T].prod() *\
            (1. - puid_eff_T[jets_failed_T]).prod()

        pData_L = puid_eff_L[jets_passed_L].prod() *\
            puid_sf_L[jets_passed_L].prod() *\
            (1. - puid_eff_L[jets_failed_L] *
             puid_sf_L[jets_failed_L]).prod()
        pData_T = puid_eff_T[jets_passed_T].prod() *\
            puid_sf_T[jets_passed_T].prod() *\
            (1. - puid_eff_T[jets_failed_T] *
             puid_sf_T[jets_failed_T]).prod()

        puid_weight = np.ones(numevents)
        puid_weight[pMC_L * pMC_T != 0] = np.divide(
            (pData_L * pData_T)[pMC_L * pMC_T != 0],
            (pMC_L * pMC_T)[pMC_L * pMC_T != 0])

    else:
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{year}_{wp}"
        h_sf_name = f"h2_eff_sf{year}_{wp}"
        puid_eff = evaluator[h_eff_name](jets[pt_name], jets.eta)
        puid_sf = evaluator[h_sf_name](jets[pt_name], jets.eta)
        jets_passed = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & jet_puid
        jets_failed = (jets[pt_name] > 25) &\
            (jets[pt_name] < 50) & (~jet_puid)

        pMC = puid_eff[jets_passed].prod() *\
            (1. - puid_eff[jets_failed]).prod()
        pData = puid_eff[jets_passed].prod() *\
            puid_sf[jets_passed].prod() *\
            (1. - puid_eff[jets_failed] *
             puid_sf[jets_failed]).prod()
        puid_weight = np.ones(numevents)
        puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
    return puid_weight


def qgl_weights(jet, isHerwig):
    df = pd.DataFrame(index=jet.index, columns=['weights'])

    wgt_mask = (
        (jet.partonFlavour != 0) &
        (abs(jet.eta) < 2) &
        (jet.qgl > 0)
    )
    light = wgt_mask & (abs(jet.partonFlavour) < 4)
    gluon = wgt_mask & (jet.partonFlavour == 21)

    qgl = jet.qgl

    if isHerwig:
        df.weights[light] = (
            1.16636 * qgl[light]**3 -
            2.45101 * qgl[light]**2 +
            1.86096 * qgl[light] + 0.596896
        )
        df.weights[gluon] = (
            -63.2397 * qgl[gluon]**7 +
            111.455 * qgl[gluon]**6 -
            16.7487 * qgl[gluon]**5 -
            72.8429 * qgl[gluon]**4 +
            56.7714 * qgl[gluon]**3 -
            19.2979 * qgl[gluon]**2 +
            3.41825 * qgl[gluon] + 0.919838
        )
    else:
        df.weights[light] = (
            -0.666978 * qgl[light]**3 +
            0.929524 * qgl[light]**2 -
            0.255505 * qgl[light] + 0.981581
        )
        df.weights[gluon] = (
            -55.7067 * qgl[gluon]**7 +
            113.218 * qgl[gluon]**6 -
            21.1421 * qgl[gluon]**5 -
            99.927 * qgl[gluon]**4 +
            92.8668 * qgl[gluon]**3 -
            34.3663 * qgl[gluon]**2 +
            6.27 * qgl[gluon] + 0.612992
        )
    return df.weights


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


def get_jec_unc(name, jet_pt, jet_eta, jecunc):
    idx_func = jecunc.levels.index(name)
    jec_unc_func = jecunc._funcs[idx_func]
    function_signature = jecunc._funcs[idx_func].signature
    counts = jet_pt.counts
    args = {
        "JetPt": np.array(jet_pt.flatten()),
        "JetEta": np.array(jet_eta.flatten())
    }
    func_args = tuple([args[s] for s in function_signature])
    jec_unc_vec = jec_unc_func(*func_args)
    return awkward.JaggedArray.fromcounts(counts, jec_unc_vec)
