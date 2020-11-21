import numpy as np
import awkward
import uproot
import numba

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
            id_file[scaleFactors['id'][1]].values *\
            scaleFactors['scale']
        mu_id_err +=\
            id_file[scaleFactors['id'][1]].variances**0.5 *\
            scaleFactors['scale']
        mu_id_edges = id_file[scaleFactors['id'][1]].edges
        mu_iso_vals +=\
            iso_file[scaleFactors['iso'][1]].values *\
            scaleFactors['scale']
        mu_iso_err +=\
            iso_file[scaleFactors['iso'][1]].variances**0.5 *\
            scaleFactors['scale']
        mu_iso_edges = iso_file[scaleFactors['iso'][1]].edges
        mu_trig_vals_data +=\
            trig_file[scaleFactors['trig'][1]].values *\
            scaleFactors['scale']
        mu_trig_vals_mc +=\
            trig_file[scaleFactors['trig'][2]].values *\
            scaleFactors['scale']
        mu_trig_err_data +=\
            trig_file[scaleFactors['trig'][1]].variances**0.5 *\
            scaleFactors['scale']
        mu_trig_err_mc +=\
            trig_file[scaleFactors['trig'][2]].variances**0.5 *\
            scaleFactors['scale']
        mu_trig_edges = trig_file[scaleFactors['trig'][1]].edges

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

    return (mu_id_sf, mu_id_err, mu_iso_sf,
            mu_iso_err, mu_trig_eff_data,
            mu_trig_eff_mc, mu_trig_err_data, mu_trig_err_mc)


def musf_evaluator(lookups, year, numevents, muons):
    mu_id_sf, mu_id_err, mu_iso_sf,\
        mu_iso_err, mu_trig_eff_data,\
        mu_trig_eff_mc, mu_trig_err_data,\
        mu_trig_err_mc = lookups
    pt = muons.pt_raw.compact()
    eta = muons.eta_raw.compact()
    abs_eta = abs(muons.eta_raw.compact())
    muID = np.ones(len(muons.flatten()), dtype=float)
    muIso = np.ones(len(muons.flatten()), dtype=float)
    muTrig = np.ones(numevents, dtype=float)
    muIDerr = np.zeros(len(muons.flatten()), dtype=float)
    muIsoerr = np.zeros(len(muons.flatten()), dtype=float)
    muTrig_up = np.ones(numevents, dtype=float)
    muTrig_down = np.ones(numevents, dtype=float)

    if '2016' in year:
        muID = mu_id_sf(eta, pt)
        muIso = mu_iso_sf(eta, pt)
        muIDerr = mu_id_err(eta, pt)
        muIsoerr = mu_iso_err(eta, pt)
        muTrig_data = mu_trig_eff_data(abs_eta, pt)
        muTrig_mc = mu_trig_eff_mc(abs_eta, pt)
        muTrigerr_data = mu_trig_err_data(abs_eta, pt)
        muTrigerr_mc = mu_trig_err_mc(abs_eta, pt)
    else:
        muID = mu_id_sf(pt, abs_eta)
        muIso = mu_iso_sf(pt, abs_eta)
        muIDerr = mu_id_err(pt, abs_eta)
        muIsoerr = mu_iso_err(pt, abs_eta)
        muTrig_data = mu_trig_eff_data(abs_eta, pt)
        muTrig_mc = mu_trig_eff_mc(abs_eta, pt)
        muTrigerr_data = mu_trig_err_data(abs_eta, pt)
        muTrigerr_mc = mu_trig_err_mc(abs_eta, pt)

    denom = ((1 - (1. - muTrig_mc).prod()))
    denom_up = ((1 - (1. - muTrig_mc - muTrigerr_mc).prod()) != 0)
    denom_dn = ((1 - (1. - muTrig_mc + muTrigerr_mc).prod()) != 0)

    muTrig[denom !=0] = (
        (1 - (1. - muTrig_data).prod())[denom != 0] / denom[denom != 0])
    muTrig_up[denom_up != 0] = (
        (1 - (1. - muTrig_data - muTrigerr_data).prod())[denom_up != 0] /
        denom_up[denom_up != 0])
    muTrig_down[denom_dn != 0] = (
        (1 - (1. - muTrig_data + muTrigerr_data).prod())[denom_dn != 0] /
        denom_dn[denom_dn != 0])

    # muSF = (muID*muIso).prod()*muTrig
    # muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * muTrig_up).prod()
    # muSF_down = ((muID - muIDerr) *
    # (muIso - muIsoerr) * muTrig_down).prod()
    # return muSF, muSF_up, muSF_down

    muID_up = (muID + muIDerr).prod()
    muID_down = (muID - muIDerr).prod()
    muIso_up = (muIso + muIsoerr).prod()
    muIso_down = (muIso - muIsoerr).prod()
    muID = muID.prod()
    muIso = muIso.prod()

    return muID, muID_up, muID_down, muIso,\
        muIso_up, muIso_down, muTrig,\
        muTrig_up, muTrig_down


def pu_lookup(parameters, mode='nom', auto=[]):
    if mode == 'nom':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup']
    elif mode == 'up':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup_plus']
    elif mode == 'down':
        pu_hist_data = uproot.open(
                        parameters['pu_file_data'])['pileup_minus']
    else:
        print("PU lookup: incorrect mode ", mode)
        return

    nbins = len(pu_hist_data)
    edges = [[i for i in range(nbins)]]
    # pu_hist_mc = load("data/pileup/pisa_lookup_2018.coffea")(range(102))
    if len(auto) == 0:
        pu_hist_mc = uproot.open(parameters['pu_file_mc'])['pu_mc']
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
    pu_weight[ntrueint > 100] = 1
    pu_weight[ntrueint < 1] = 1
    return pu_weight


# https://github.com/jpata/hepaccelerate-cms/blob/
# f5965648f8a7861cb9856d0b5dd34a53ed42c027/tests/
# hmm/hmumu_utils.py#L1396
@numba.njit(parallel=True, cache=True)
def fsr_evaluator(muons_offsets, fsr_offsets,
                  muons_pt, muons_eta, muons_phi,
                  muons_mass, muons_iso, muons_fsrIndex,
                  fsr_pt, fsr_eta, fsr_phi, fsr_iso,
                  fsr_drEt2, has_fsr):
    for iev in numba.prange(len(muons_offsets) - 1):
        # loop over muons in event
        mu_first = muons_offsets[iev]
        mu_last = muons_offsets[iev + 1]
        for imu in range(mu_first, mu_last):
            # relative FSR index in the event
            fsr_idx_relative = muons_fsrIndex[imu]

            if (fsr_idx_relative >= 0):
                # absolute index in the full FSR vector for all events
                ifsr = fsr_offsets[iev] + fsr_idx_relative
                mu_kin = {
                    "pt": muons_pt[imu],
                    "eta": muons_eta[imu],
                    "phi": muons_phi[imu],
                    "mass": muons_mass[imu]}
                fsr_kin = {
                    "pt": fsr_pt[ifsr],
                    "eta": fsr_eta[ifsr],
                    "phi": fsr_phi[ifsr],
                    "mass": 0.}

                if fsr_iso[ifsr] > 1.8:
                    continue
                if fsr_drEt2[ifsr] > 0.012:
                    continue
                if fsr_pt[ifsr] / muons_pt[imu] > 0.4:
                    continue
                if abs(fsr_eta[ifsr]) > 2.4:
                    continue

                has_fsr[imu] = True

                # compute and set corrected momentum
                px_total = 0
                py_total = 0
                pz_total = 0
                e_total = 0
                for obj in [mu_kin, fsr_kin]:
                    px = obj["pt"] * np.cos(obj["phi"])
                    py = obj["pt"] * np.sin(obj["phi"])
                    pz = obj["pt"] * np.sinh(obj["eta"])
                    e = np.sqrt(px**2 + py**2 + pz**2 + obj["mass"]**2)
                    px_total += px
                    py_total += py
                    pz_total += pz
                    e_total += e
                out_pt = np.sqrt(px_total**2 + py_total**2)
                out_eta = np.arcsinh(pz_total / out_pt)
                out_phi = np.arctan2(py_total, px_total)
                out_mass = np.sqrt(e_total**2 -
                                   px_total**2 -
                                   py_total**2 -
                                   pz_total**2)

                # reference:
                # https://gitlab.cern.ch/uhh-cmssw/
                # fsr-photon-recovery/tree/master
                muons_iso[imu] = (muons_iso[imu] *
                                  muons_pt[imu] -
                                  fsr_pt[ifsr]) / out_pt

                muons_pt[imu] = out_pt
                muons_eta[imu] = out_eta
                muons_phi[imu] = out_phi
                muons_mass[imu] = out_mass

    return muons_pt, muons_eta, muons_phi,\
           muons_mass, muons_iso, has_fsr


def btag_weights(processor, lookup, systs, jets,
                 weights, bjet_sel_mask, numevents):
    btag_wgt = np.ones(numevents, dtype=float)
    jets_ = jets[abs(jets.eta) < 2.4]
    jet_pt_ = awkward.JaggedArray.fromcounts(
                jets_[jets_.counts > 0].counts,
                np.minimum(jets_.pt.flatten(),
                           1000.))

    btag_wgt[(jets_.counts > 0)] =\
        lookup('central',
               jets_[jets_.counts > 0].hadronFlavour,
               abs(jets_[jets_.counts > 0].eta),
               jet_pt_,
               jets_[jets_.counts > 0].btagDeepB,
               True).prod()

    btag_wgt[btag_wgt < 0.01] = 1.

    btag_syst = {}
    flavors = {
        0: ["jes", "hf", "lfstats1", "lfstats2"],
        1: ["jes", "hf", "lfstats1", "lfstats2"],
        2: ["jes", "hf", "lfstats1", "lfstats2"],
        3: ["jes", "hf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "lf", "hfstats1", "hfstats2"],
        21: ["jes", "hf", "lfstats1", "lfstats2"],
    }

    for sys in systs:
        njets = len(jets_.flatten())
        btag_syst[sys] = [np.ones(njets, dtype=float),
                          np.ones(njets, dtype=float)]
        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = (abs(jets_.hadronFlavour) == f)
                btag_syst[sys][0][btag_mask.flatten()] =\
                    lookup('up_' + sys,
                           jets_.hadronFlavour[btag_mask],
                           abs(jets_.eta)[btag_mask],
                           jets_.pt[btag_mask],
                           jets_.btagDeepB[btag_mask],
                           True).flatten()
                btag_syst[sys][1][btag_mask.flatten()] =\
                    lookup('down_' + sys,
                           jets_.hadronFlavour[btag_mask],
                           abs(jets_.eta)[btag_mask],
                           jets_.pt[btag_mask],
                           jets_.btagDeepB[btag_mask], True).flatten()
        btag_syst[sys][0] = awkward.JaggedArray.fromcounts(
                                jets_.counts,
                                btag_syst[sys][0]).prod()
        btag_syst[sys][1] = awkward.JaggedArray.fromcounts(
                                jets_.counts,
                                btag_syst[sys][1]).prod()

    sum_before = weights.df['nominal'][bjet_sel_mask].sum()
    sum_after = weights.df['nominal'][bjet_sel_mask].multiply(
        btag_wgt[bjet_sel_mask],
        axis=0).sum()
    btag_wgt = btag_wgt * sum_before / sum_after
    return btag_wgt, btag_syst


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
    weights = np.ones(len(jet.qgl), dtype=float)
    wgt_mask = (jet.partonFlavour != 0) &\
        (abs(jet.eta) < 2) & (jet.qgl > 0)
    light = wgt_mask & (abs(jet.partonFlavour) < 4)
    gluon = wgt_mask & (jet.partonFlavour == 21)

    qgl = jet.qgl

    if isHerwig:
        weights[light] = 1.16636 * qgl[light]**3 -\
            2.45101 * qgl[light]**2 +\
            1.86096 * qgl[light] + 0.596896
        weights[gluon] = -63.2397 * qgl[gluon]**7 +\
            111.455 * qgl[gluon]**6 -\
            16.7487 * qgl[gluon]**5 -\
            72.8429 * qgl[gluon]**4 +\
            56.7714 * qgl[gluon]**3 -\
            19.2979 * qgl[gluon]**2 +\
            3.41825 * qgl[gluon] + 0.919838
    else:
        weights[light] = -0.666978 * qgl[light]**3 +\
            0.929524 * qgl[light]**2 -\
            0.255505 * qgl[light] + 0.981581
        weights[gluon] = -55.7067 * qgl[gluon]**7 +\
            113.218 * qgl[gluon]**6 -\
            21.1421 * qgl[gluon]**5 -\
            99.927 * qgl[gluon]**4 +\
            92.8668 * qgl[gluon]**3 -\
            34.3663 * qgl[gluon]**2 +\
            6.27 * qgl[gluon] + 0.612992
    return weights


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
