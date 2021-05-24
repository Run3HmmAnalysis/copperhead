import numpy as np
import awkward as ak


def apply_roccor(df, rochester, is_mc):
    if is_mc:
        hasgen = ~np.isnan(ak.fill_none(df.Muon.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(
            *ak.to_numpy(ak.flatten(df.Muon.pt)).shape
        )
        mc_rand = ak.unflatten(mc_rand, ak.num(df.Muon.pt, axis=1))

        corrections = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))
        errors = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))
        mc_kspread = rochester.kSpreadMC(
            df.Muon.charge[hasgen],
            df.Muon.pt[hasgen],
            df.Muon.eta[hasgen],
            df.Muon.phi[hasgen],
            df.Muon.matched_gen.pt[hasgen]
        )

        mc_ksmear = rochester.kSmearMC(
            df.Muon.charge[~hasgen],
            df.Muon.pt[~hasgen],
            df.Muon.eta[~hasgen],
            df.Muon.phi[~hasgen],
            df.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen]
        )

        errspread = rochester.kSpreadMCerror(
            df.Muon.charge[hasgen],
            df.Muon.pt[hasgen],
            df.Muon.eta[hasgen],
            df.Muon.phi[hasgen],
            df.Muon.matched_gen.pt[hasgen]
        )
        errsmear = rochester.kSmearMCerror(
            df.Muon.charge[~hasgen],
            df.Muon.pt[~hasgen],
            df.Muon.eta[~hasgen],
            df.Muon.phi[~hasgen],
            df.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen]
        )
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))

        corrections = ak.unflatten(
            corrections, ak.num(df.Muon.pt, axis=1)
        )
        errors = ak.unflatten(
            errors, ak.num(df.Muon.pt, axis=1)
        )

    else:
        corrections = rochester.kScaleDT(
            df.Muon.charge,
            df.Muon.pt,
            df.Muon.eta,
            df.Muon.phi
        )
        errors = rochester.kScaleDTerror(
            df.Muon.charge,
            df.Muon.pt,
            df.Muon.eta,
            df.Muon.phi
        )

    df['Muon', 'pt_roch'] = (
        df.Muon.pt * corrections
    )
    df['Muon', 'pt_roch_up'] = df.Muon.pt_roch + df.Muon.pt * errors
    df['Muon', 'pt_roch_down'] = df.Muon.pt_roch - df.Muon.pt * errors
