import numpy as np


def puid_weights(evaluator, year, jets, pt_name, jet_puid_opt, jet_puid, numevents):
    if "2017corrected" in jet_puid_opt:
        h_eff_name_L = f"h2_eff_mc{year}_L"
        h_sf_name_L = f"h2_eff_sf{year}_L"
        h_eff_name_T = f"h2_eff_mc{year}_T"
        h_sf_name_T = f"h2_eff_sf{year}_T"
        puid_eff_L = evaluator[h_eff_name_L](jets[pt_name], jets.eta)
        puid_sf_L = evaluator[h_sf_name_L](jets[pt_name], jets.eta)
        puid_eff_T = evaluator[h_eff_name_T](jets[pt_name], jets.eta)
        puid_sf_T = evaluator[h_sf_name_T](jets[pt_name], jets.eta)

        jets_passed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & jet_puid
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        jets_failed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & (~jet_puid)
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        jets_passed_T = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & jet_puid
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )
        jets_failed_T = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & (~jet_puid)
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )

        pMC_L = (
            puid_eff_L[jets_passed_L].prod() * (1.0 - puid_eff_L[jets_failed_L]).prod()
        )
        pMC_T = (
            puid_eff_T[jets_passed_T].prod() * (1.0 - puid_eff_T[jets_failed_T]).prod()
        )

        pData_L = (
            puid_eff_L[jets_passed_L].prod()
            * puid_sf_L[jets_passed_L].prod()
            * (1.0 - puid_eff_L[jets_failed_L] * puid_sf_L[jets_failed_L]).prod()
        )
        pData_T = (
            puid_eff_T[jets_passed_T].prod()
            * puid_sf_T[jets_passed_T].prod()
            * (1.0 - puid_eff_T[jets_failed_T] * puid_sf_T[jets_failed_T]).prod()
        )

        puid_weight = np.ones(numevents)
        puid_weight[pMC_L * pMC_T != 0] = np.divide(
            (pData_L * pData_T)[pMC_L * pMC_T != 0], (pMC_L * pMC_T)[pMC_L * pMC_T != 0]
        )

    else:
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{year}_{wp}"
        h_sf_name = f"h2_eff_sf{year}_{wp}"
        puid_eff = evaluator[h_eff_name](jets[pt_name], jets.eta)
        puid_sf = evaluator[h_sf_name](jets[pt_name], jets.eta)
        jets_passed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & jet_puid
        jets_failed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & (~jet_puid)

        pMC = puid_eff[jets_passed].prod() * (1.0 - puid_eff[jets_failed]).prod()
        pData = (
            puid_eff[jets_passed].prod()
            * puid_sf[jets_passed].prod()
            * (1.0 - puid_eff[jets_failed] * puid_sf[jets_failed]).prod()
        )
        puid_weight = np.ones(numevents)
        puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
    return puid_weight
