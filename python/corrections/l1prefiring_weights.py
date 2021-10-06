import awkward as ak


def l1pf_weights(df):
    l1pfw = ak.to_pandas(df.L1PreFiringWeight)
    ret = {
        "nom": l1pfw.Nom,
        "up": l1pfw.Up,
        "down": l1pfw.Dn,
    }
    return ret
