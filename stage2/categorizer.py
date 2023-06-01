def split_into_channels(df, v=""):
    df[f"njets_{v}"].fillna(0, inplace=True)
    df.loc[:, f"channel_{v}"] = "none"
    df.loc[
        (df[f"nBtagLoose_{v}"] >= 2) | (df[f"nBtagMedium_{v}"] >= 1), f"channel_{v}"
    ] = "ttHorVH"

    df.loc[
        (df[f"channel_{v}"] == "none")
        & (df[f"jj_mass_{v}"] > 400)
        & (df[f"jj_dEta_{v}"] > 2.5)
        & (df[f"jet1_pt_{v}"] > 35),
        f"channel_{v}",
    ] = "vbf"
    df.loc[
        (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] < 1), f"channel_{v}"
    ] = "ggh_0jets"
    df.loc[
        (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] == 1), f"channel_{v}"
    ] = "ggh_1jet"
    df.loc[
        (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] > 1), f"channel_{v}"
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


def categorize_dnn_output(df, score_name, channel, region, year):
    # Run 2 (VBF yields)
    target_yields = {
        "2016": [
            0.35455259,
            0.50239086,
            0.51152889,
            0.52135985,
            0.5282209,
            0.54285134,
            0.54923751,
            0.56504687,
            0.57204477,
            0.58273066,
            0.5862248,
            0.59568793,
            0.60871905,
        ],
        "2017": [
            0.44194544,
            0.55885064,
            0.57796123,
            0.58007343,
            0.58141001,
            0.58144682,
            0.57858609,
            0.59887533,
            0.59511901,
            0.59644831,
            0.59825915,
            0.59283309,
            0.57329743,
        ],
        "2018": [
            0.22036263,
            1.31808978,
            1.25396849,
            1.183724,
            1.12620194,
            1.06041376,
            0.99941623,
            0.93224412,
            0.87074753,
            0.80599462,
            0.73469265,
            0.668018,
            0.60991945,
        ],
    }

    # 20 uniform bins (all signal)
    target_yields = {
        "2016": [
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
            0.65410305,
        ],
        "2017": [
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
            0.69204835,
        ],
        "2018": [
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
            1.05256315,
        ],
    }
    bins = [df[score_name].max()]

    slicer = (
        (df.channel_nominal == channel) & (df.region == region) & (df.year == int(year))
    )
    df_sorted = (
        df.loc[slicer, :]
        .sort_values(by=score_name, ascending=False)
        .reset_index(drop=True)
    )
    df_sorted["wgt_cumsum"] = df_sorted.wgt_nominal.cumsum()

    tot_yield = 0
    last_yield = 0
    for yi in reversed(target_yields[year]):
        tot_yield += yi
        for i in range(df_sorted.shape[0] - 1):
            value = df_sorted.loc[i, "wgt_cumsum"]
            value1 = df_sorted.loc[i + 1, "wgt_cumsum"]
            if (value < tot_yield) & (value1 > tot_yield):
                if abs(last_yield - tot_yield) < 1e-06:
                    continue
                if abs(tot_yield - sum(target_yields[year])) < 1e-06:
                    continue
                bins.append(df_sorted.loc[i, score_name])
                last_yield = tot_yield
    bins.append(0.0)
    bins = sorted(bins)
    print(bins)
