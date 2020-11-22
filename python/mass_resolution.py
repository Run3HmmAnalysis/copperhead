import numpy as np


def mass_resolution_purdue(is_mc, evaluator, mu1, mu2, mass, two_muons, year):
    # Returns absolute mass resolution!
    dpt1 = (mu1[two_muons].ptErr*mass[two_muons]) / (2*mu1[two_muons].pt)
    dpt2 = (mu2[two_muons].ptErr*mass[two_muons]) / (2*mu2[two_muons].pt)

    if is_mc:
        label = f"res_calib_MC_{year}"
    else:
        label = f"res_calib_Data_{year}"

    calibration = np.array(
        evaluator[label](
            mu1[two_muons].pt.flatten(),
            abs(mu1[two_muons].eta.flatten()),
            abs(mu2[two_muons].eta.flatten())))

    return (np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration).flatten()


def mass_resolution_pisa(extractor, mu1, mu2, two_muons):
    # Returns relative mass resolution!
    evaluator = extractor.make_evaluator()["PtErrParametrization"]
    mu1_ptErr = evaluator(np.log(mu1.pt), np.abs(mu1.eta))
    mu2_ptErr = evaluator(np.log(mu2.pt), np.abs(mu2.eta))
    return np.sqrt(0.5 *
                   (mu1_ptErr * mu1_ptErr +
                    mu2_ptErr * mu2_ptErr))[two_muons].flatten()
