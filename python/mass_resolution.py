import numpy as np


def mass_resolution_purdue(is_mc, evaluator, df, year):
    # Returns absolute mass resolution!
    dpt1 = (df.mu1_ptErr*df.dimuon_mass) / (2*df.mu1_pt)
    dpt2 = (df.mu2_ptErr*df.dimuon_mass) / (2*df.mu2_pt)

    if is_mc:
        label = f"res_calib_MC_{year}"
    else:
        label = f"res_calib_Data_{year}"
    calibration = np.array(
        evaluator[label](
            df.mu1_pt.values,
            abs(df.mu1_eta.values),
            abs(df.mu2_eta.values)
        )
    )

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration


def mass_resolution_pisa(extractor, df):
    # Returns relative mass resolution!
    evaluator = extractor.make_evaluator()["PtErrParametrization"]
    mu1_ptErr = evaluator(np.log(df.mu1_pt), np.abs(df.mu1_eta))
    mu2_ptErr = evaluator(np.log(df.mu2_pt), np.abs(df.mu2_eta))
    return np.sqrt(0.5 *
                   (mu1_ptErr * mu1_ptErr +
                    mu2_ptErr * mu2_ptErr))
