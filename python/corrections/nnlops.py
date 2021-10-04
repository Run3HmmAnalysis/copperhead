import numpy as np
import uproot


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
            self.ratio_0jet[mode].member("fX"),
            self.ratio_0jet[mode].member("fY"))
        result[njets == 1] = np.interp(
            np.minimum(hig_pt[njets == 1], 625.),
            self.ratio_1jet[mode].member("fX"),
            self.ratio_1jet[mode].member("fY"))
        result[njets == 2] = np.interp(
            np.minimum(hig_pt[njets == 2], 800.),
            self.ratio_2jet[mode].member("fX"),
            self.ratio_2jet[mode].member("fY"))
        result[njets > 2] = np.interp(
            np.minimum(hig_pt[njets > 2], 925.),
            self.ratio_3jet[mode].member("fX"),
            self.ratio_3jet[mode].member("fY"))
        return result


def nnlops_weights(df, numevents, parameters, dataset):
    nnlops = NNLOPS_Evaluator(parameters['nnlops_file'])
    nnlopsw = np.ones(numevents, dtype=float)
    if 'amc' in dataset:
        nnlopsw = nnlops.evaluate(
            df.HTXS.Higgs_pt, df.HTXS.njets30, "mcatnlo"
        )
    elif 'powheg' in dataset:
        nnlopsw = nnlops.evaluate(
            df.HTXS.Higgs_pt, df.HTXS.njets30, "powheg"
        )
    return nnlopsw
