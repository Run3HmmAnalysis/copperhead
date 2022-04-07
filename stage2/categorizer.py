def split_into_channels(df, v=""):
    if v is None:
        # this was used for Delphes datasets
        cuts = {
            "vbf": (df.jj_mass > 400) & (df.jj_dEta > 2.5) & (df.jet1_pt > 35),
            "ggh_0jets": (df.njets == 0),
            "ggh_1jet": (df.njets == 1)
            & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5) | (df.jet1_pt < 35)),
            "ggh_2orMoreJets": (df.njets >= 2)
            & ((df.jj_mass <= 400) | (df.jj_dEta <= 2.5) | (df.jet1_pt < 35)),
        }

        for cname, cut in cuts.items():
            df.loc[cut, "channel"] = cname
    else:
        df[f"njets {v}"].fillna(0, inplace=True)
        df.loc[:, f"channel {v}"] = "none"
        df.loc[
            (df[f"nBtagLoose {v}"] >= 2) | (df[f"nBtagMedium {v}"] >= 1), f"channel {v}"
        ] = "ttHorVH"
        df.loc[
            (df[f"channel {v}"] == "none")
            & (df[f"jj_mass {v}"] > 400)
            & (df[f"jj_dEta {v}"] > 2.5)
            & (df[f"jet1_pt {v}"] > 35),
            f"channel {v}",
        ] = "vbf"
        df.loc[
            (df[f"channel {v}"] == "none") & (df[f"njets {v}"] < 1), f"channel {v}"
        ] = "ggh_0jets"
        df.loc[
            (df[f"channel {v}"] == "none") & (df[f"njets {v}"] == 1), f"channel {v}"
        ] = "ggh_1jet"
        df.loc[
            (df[f"channel {v}"] == "none") & (df[f"njets {v}"] > 1), f"channel {v}"
        ] = "ggh_2orMoreJets"


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
