def split_into_channels(df):
    cuts = {
        "vbf": (df.jj_mass > 400) & (df.jj_dEta > 2.5),
        "ggh_0jets": (df.njets == 0) & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
        "ggh_1jet": (df.njets == 1) & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
        "ggh_2orMoreJets": (df.njets >= 2)
        & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5)),
    }
    for cname, cut in cuts.items():
        df.loc[cut, "channel"] = cname


def categorize_by_score(df, scores, mode="uniform", **kwargs):
    nbins = kwargs.pop("nbins", 4)
    for channel, score_name in scores.items():
        score = df.loc[df.channel == channel, score_name]
        if mode == "uniform":
            for i in range(nbins):
                cat_name = f"{score_name}_cat{i}"
                cut_lo = score.quantile(i / nbins)
                cut_hi = score.quantile((i + 1) / nbins)
                cut = (df.channel == channel) & (score > cut_lo) & (score < cut_hi)
                df.loc[cut, "category"] = cat_name
