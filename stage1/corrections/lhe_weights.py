import pandas as pd
import awkward as ak


def lhe_weights(df, output, dataset, year):
    factor2 = ("dy_m105_160_amc" in dataset) and (("2017" in year) or ("2018" in year))
    if factor2:
        lhefactor = 2.0
    else:
        lhefactor = 1.0
    nLHEScaleWeight = ak.count(df.LHEScaleWeight, axis=1)
    lhe_df = pd.DataFrame(
        data=ak.to_numpy(nLHEScaleWeight),
        index=output.index,
        columns=["nLHEScaleWeight"],
    )
    for i in [1, 3, 4, 5, 6, 7, 15, 24, 34]:
        cut = lhe_df.nLHEScaleWeight > i
        cut_ak = nLHEScaleWeight > i
        lhe_df[f"LHE{i}"] = 1.0
        lhe_df.loc[cut, f"LHE{i}"] = ak.to_numpy(df.LHEScaleWeight[cut_ak][:, i])

    cut8 = lhe_df.nLHEScaleWeight > 8
    cut30 = lhe_df.nLHEScaleWeight > 30
    lhe_ren_up = lhe_df.LHE6 * lhefactor
    lhe_ren_up[cut8] = lhe_df.LHE7 * lhefactor
    lhe_ren_up[cut30] = lhe_df.LHE34 * lhefactor
    lhe_ren_down = lhe_df.LHE1 * lhefactor
    lhe_ren_down[cut8] = lhe_df.LHE1 * lhefactor
    lhe_ren_down[cut30] = lhe_df.LHE5 * lhefactor

    lhe_fac_up = lhe_df.LHE4 * lhefactor
    lhe_fac_up[cut8] = lhe_df.LHE5 * lhefactor
    lhe_fac_up[cut30] = lhe_df.LHE24 * lhefactor
    lhe_fac_down = lhe_df.LHE3 * lhefactor
    lhe_fac_down[cut8] = lhe_df.LHE3 * lhefactor
    lhe_fac_down[cut30] = lhe_df.LHE15 * lhefactor

    lhe_ren = {"up": lhe_ren_up, "down": lhe_ren_down}
    lhe_fac = {"up": lhe_fac_up, "down": lhe_fac_down}
    return lhe_ren, lhe_fac
