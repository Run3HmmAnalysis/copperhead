# Simpliest possible implementation
class DimuonProcessorLite(processor.ProcessorABC):
    def __init__(self, mass_window=[76,106]):
        self.mass_window = mass_window
        self._columns = ['Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass']
        dataset_axis = hist.Cat("dataset", "")
        dimuon_mass_ax =  hist.Bin("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", 100, self.mass_window[0], self.mass_window[1])
        accumulators = {'dimuon_mass': hist.Hist("Counts", dataset_axis, dimuon_mass_ax)}
        self._accumulator = processor.dict_accumulator(accumulators)
 
    
    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):    
        output = self.accumulator.identity()
        dataset = df.metadata['dataset']
            
        if 'data' in dataset:
            event_weight = np.ones(df.shape[0])
        else:
            event_weight = df.genWeight
    
        # just take two highest-pt muons
        muons = df.Muon[df.Muon.pt > 20,0:2]
        event_filter = (muons.counts == 2).flatten()

        muons = muons[event_filter]
        event_weight = event_weight[event_filter]
    
        dimuon_pt, dimuon_eta, dimuon_phi, dimuon_mass = p4_sum(muons[muons.pt.argmax()], muons[muons.pt.argmin()])
        
        dimuon_filter = ((dimuon_mass > self.mass_window[0]) & (dimuon_mass < self.mass_window[1])).flatten()        

        dimuon_mass = dimuon_mass[dimuon_filter]
        event_weight = event_weight[dimuon_filter]

        output['dimuon_mass'].fill(dataset=dataset, dimuon_mass=dimuon_mass.flatten(), weight=event_weight)

        return output
    
    def postprocess(self, accumulator):
        return accumulator