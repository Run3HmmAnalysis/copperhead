def apply_roccor(rochester, is_mc, muons):
    import awkward
    import numpy as np
    muons = muons.compact()
    corrections = muons.pt.ones_like()  
    
    if is_mc:
        mc_rand = np.random.rand(*muons.pt.flatten().shape)
        mc_rand = awkward.JaggedArray.fromoffsets(muons.pt.offsets, mc_rand)
        hasgen = ~np.isnan(muons.matched_gen.pt.fillna(np.nan))
        mc_rand = awkward.JaggedArray.fromoffsets(hasgen.offsets, mc_rand)._content

        mc_kspread = rochester.kSpreadMC(muons.charge[hasgen], muons.pt[hasgen], muons.eta[hasgen], muons.phi[hasgen],
                                         muons.matched_gen.pt[hasgen])
        mc_ksmear = rochester.kSmearMC(muons.charge[~hasgen], muons.pt[~hasgen],muons.eta[~hasgen],muons.phi[~hasgen],
                                       muons.nTrackerLayers[~hasgen], mc_rand[~hasgen])
        corrections = np.ones_like(muons.pt.flatten())
        corrections[hasgen.flatten()] = mc_kspread.flatten()
        corrections[~hasgen.flatten()] = mc_ksmear.flatten() 
    else:
        corrections = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi)      
    
    return corrections

def p4_sum(obj1, obj2):
    import numpy as np
    assert(obj1.shape==obj2.shape)
    px = np.zeros(obj1.shape[0])
    py = np.zeros(obj1.shape[0])
    pz = np.zeros(obj1.shape[0])
    e = np.zeros(obj1.shape[0])
    
    for obj in [obj1, obj2]:
        px_ = obj.pt*np.cos(obj.phi)
        py_ = obj.pt*np.sin(obj.phi)
        pz_ = obj.pt*np.sinh(obj.eta)
        e_  = np.sqrt(px_**2 + py_**2 + pz_**2 + obj.mass**2)
        px = px + px_
        py = py + py_
        pz = pz + pz_
        e = e + e_
        
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    return pt, eta, phi, mass

def p4_sum_alt(obj1_pt, obj1_eta, obj1_phi, obj1_mass, obj2_pt, obj2_eta, obj2_phi, obj2_mass):
    import numpy as np
    assert(len(obj1_pt)==len(obj2_pt))
    px = np.zeros(len(obj1_pt))
    py = np.zeros(len(obj1_pt))
    pz = np.zeros(len(obj1_pt))
    e = np.zeros(len(obj1_pt))
    obj1 = {
        'pt': obj1_pt,
        'eta': obj1_eta,
        'phi': obj1_phi,
        'mass': obj1_mass,
    }
    obj2 = {
        'pt': obj2_pt,
        'eta': obj2_eta,
        'phi': obj2_phi,
        'mass': obj2_mass,
    }

    for obj in [obj1, obj2]:
        px_ = obj['pt']*np.cos(obj['phi'])
        py_ = obj['pt']*np.sin(obj['phi'])
        pz_ = obj['pt']*np.sinh(obj['eta'])
        e_  = np.sqrt(px_**2 + py_**2 + pz_**2 + obj['mass']**2)
        px = px + px_
        py = py + py_
        pz = pz + pz_
        e = e + e_
        
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    return pt, eta, phi, mass


import uproot
import numpy as np

class NNLOPS_Evaluator(object):
    def __init__(self, input_path):
        with uproot.open(input_path) as f:
            self.ratio_0jet = {"mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_0jet"], "powheg": f["gr_NNLOPSratio_pt_powheg_0jet"]}
            self.ratio_1jet = {"mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_1jet"], "powheg": f["gr_NNLOPSratio_pt_powheg_1jet"]}
            self.ratio_2jet = {"mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_2jet"], "powheg": f["gr_NNLOPSratio_pt_powheg_2jet"]}
            self.ratio_3jet = {"mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_3jet"], "powheg": f["gr_NNLOPSratio_pt_powheg_3jet"]}

    def evaluate(self, hig_pt, njets, mode):
        result = np.ones(len(hig_pt), dtype=float)
        hig_pt = np.array(hig_pt)
        njets = np.array(njets)

        result[njets==0] = np.interp(np.minimum(hig_pt[njets==0],125.), self.ratio_0jet[mode]._fX, self.ratio_0jet[mode]._fY)
        result[njets==1] = np.interp(np.minimum(hig_pt[njets==1],652.), self.ratio_1jet[mode]._fX, self.ratio_1jet[mode]._fY)
        result[njets==2] = np.interp(np.minimum(hig_pt[njets==2],800.), self.ratio_2jet[mode]._fX, self.ratio_2jet[mode]._fY)
        result[njets >2] = np.interp(np.minimum(hig_pt[njets >2],925.), self.ratio_3jet[mode]._fX, self.ratio_3jet[mode]._fY)

        return result