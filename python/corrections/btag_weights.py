import pandas as pd


def btag_weights(processor, lookup, systs, jets, weights, bjet_sel_mask):

    btag = pd.DataFrame(index=bjet_sel_mask.index)
    jets = jets[abs(jets.eta) < 2.4]
    jets.loc[jets.pt > 1000.0, "pt"] = 1000.0

    jets["btag_wgt"] = lookup.eval(
        "central",
        jets.hadronFlavour.values,
        abs(jets.eta.values),
        jets.pt.values,
        jets.btagDeepB.values,
        True,
    )
    btag["wgt"] = jets["btag_wgt"].prod(level=0)
    btag["wgt"] = btag["wgt"].fillna(1.0)
    btag.loc[btag.wgt < 0.01, "wgt"] = 1.0

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
        jets[f"btag_{sys}_up"] = 1.0
        jets[f"btag_{sys}_down"] = 1.0
        btag[f"{sys}_up"] = 1.0
        btag[f"{sys}_down"] = 1.0

        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = abs(jets.hadronFlavour) == f
                jets.loc[btag_mask, f"btag_{sys}_up"] = lookup.eval(
                    f"up_{sys}",
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepB[btag_mask].values,
                    True,
                )
                jets.loc[btag_mask, f"btag_{sys}_down"] = lookup.eval(
                    f"down_{sys}",
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepB[btag_mask].values,
                    True,
                )

        btag[f"{sys}_up"] = jets[f"btag_{sys}_up"].prod(level=0)
        btag[f"{sys}_down"] = jets[f"btag_{sys}_down"].prod(level=0)
        btag_syst[sys] = {"up": btag[f"{sys}_up"], "down": btag[f"{sys}_down"]}

    sum_before = weights.df["nominal"][bjet_sel_mask].sum()
    sum_after = (
        weights.df["nominal"][bjet_sel_mask]
        .multiply(btag.wgt[bjet_sel_mask], axis=0)
        .sum()
    )
    btag.wgt = btag.wgt * sum_before / sum_after

    return btag.wgt, btag_syst
