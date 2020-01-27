def read_via_xrootd(path, server):
    import subprocess
    command = f"xrdfs {server} ls -R {path} | grep '.root'"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    result = proc.stdout.readlines()
    result = [server + r.rstrip().decode("utf-8") for r in result]
    return result

def apply_roccor(rochester, isData, muons):
    muons = muons.compact()
    corrections = muons.pt.ones_like()  
    if isData:
        corrections = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi)      
    else:
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
    return corrections

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


def get_regions(mass):
    regions = {
        "z-peak": ((mass>70) & (mass<110)),
        "h-sidebands": ((mass>110) & (mass<115)) | ((mass>135) & (mass<150)),
        "h-peak": ((mass>115) & (mass<135)),
    }
    return regions