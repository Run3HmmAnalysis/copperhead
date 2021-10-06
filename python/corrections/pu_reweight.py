from packaging.version import Version
import numpy as np
import uproot

import coffea
from coffea.lookup_tools import dense_lookup


def pu_lookups(parameters, mode="nom", auto=[]):
    lookups = {}
    branch = {"nom": "pileup", "up": "pileup_plus", "down": "pileup_minus"}
    for mode in ["nom", "up", "down"]:
        pu_hist_data = uproot.open(parameters["pu_file_data"])[branch[mode]].values()

        nbins = len(pu_hist_data)
        edges = [[i for i in range(nbins)]]

        if len(auto) == 0:
            pu_hist_mc = uproot.open(parameters["pu_file_mc"])["pu_mc"].values()
        else:
            pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]

        lookup = dense_lookup.dense_lookup(pu_reweight(pu_hist_data, pu_hist_mc), edges)
        if Version(coffea.__version__) < Version("0.7.6"):
            lookup._axes = lookup._axes[0]
        lookups[mode] = lookup
    return lookups


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
    weights[pu_arr_mc != 0] = pu_arr_data[pu_arr_mc != 0] / pu_arr_mc[pu_arr_mc != 0]
    maxw = min(weights.max(), 5.0)
    cropped = []
    while maxw > 3:
        cropped = []
        for i in range(len(weights)):
            cropped.append(min(maxw, weights[i]))
        shift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        if abs(shift) > 0.0025:
            break
        maxw *= 0.95

    maxw /= 0.95
    if len(cropped) > 0:
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


def pu_evaluator(lookups, parameters, numevents, ntrueint, auto_pu):
    if auto_pu:
        lookups = pu_lookups(parameters, auto=ntrueint)
    pu_weights = {}
    for var, lookup in lookups.items():
        pu_weights[var] = np.ones(numevents)
        pu_weights[var] = lookup(ntrueint)
        pu_weights[var] = np.array(pu_weights[var])
        pu_weights[var][ntrueint > 100] = 1
        pu_weights[var][ntrueint < 1] = 1
    return pu_weights
