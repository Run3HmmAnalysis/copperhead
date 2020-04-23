import numpy as np


def p4_sum(obj1, obj2):
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

def delta_r(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr