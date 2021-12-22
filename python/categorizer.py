from python.math_tools import max_abs_eta


def split_into_channels(df):
    """
    cuts = {
        "vbf": (df.jj_mass > 400) & (df.jj_dEta > 2.5),
        "ggh_0jets": (df.njets == 0) & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
        "ggh_1jet": (df.njets == 1) & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
        "ggh_2orMoreJets": (df.njets >= 2)
        & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
    }
    """
    cuts = {
        "ggh_0jets": (df.njets == 0),
        "ggh_1jet": (df.njets == 1),
        "ggh_2orMoreJets": (df.njets >= 2),
    }
    for cname, cut in cuts.items():
        df.loc[cut, "channel"] = cname

    # df["channel"] = "inclusive"


def categorize_by_score(df, scores, mode="uniform", **kwargs):
    for channel, score_name in scores.items():
        score = df.loc[df.channel == channel, score_name]
        if mode == "uniform":
            nbins = kwargs.pop("nbins", 4)
            for i in range(nbins):
                cat_name = f"{score_name}_cat{i}"
                cut_lo = score.quantile(i / nbins)
                cut_hi = score.quantile((i + 1) / nbins)
                cut = (df.channel == channel) & (score > cut_lo) & (score < cut_hi)
                df.loc[cut, "category"] = cat_name
        elif mode == "hl-lhc":
            # low-purity
            # high-purity barrel
            # high-purity endcap
            dnn_cut = kwargs.pop("dnn_cut", 0.5)
            eta_cut = kwargs.pop("eta_cut", 1.9)
            categories = {
                "LowPurity": score < dnn_cut,
                "HiPurityBarrel": (
                    (score >= dnn_cut) & (df.apply(max_abs_eta, axis=1) < eta_cut)
                ),
                "HiPurityEndcap": (
                    (score >= dnn_cut) & (df.apply(max_abs_eta, axis=1) >= eta_cut)
                ),
            }
            for cat_name, cut in categories.items():
                df.loc[(df.channel == channel) & cut, "category"] = cat_name
