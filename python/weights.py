import pandas as pd
import numpy as np

class Weights(object):
    def __init__(self, data):
        self.df = pd.DataFrame(1., index=data.event, columns=['nominal'])
        self.wgts = pd.DataFrame(index=data.event)
        self.variations = []
    
    def add_weight(self, name, wgt):
        columns = self.df.columns
        self.df[f'{name}_off'] = self.df['nominal']
        self.df[columns]  = self.df[columns].multiply(wgt, axis=0)
        self.variations.append(name)
        self.wgts[name] = wgt
        
    def add_weight_with_variations(self, name, wgt, up, down):
        columns = self.df.columns
        self.wgts[name] = wgt
        self.df[f'{name}_off'] = self.df['nominal']
        self.df[f'{name}_up'] = self.df['nominal']*up
        self.df[f'{name}_down'] = self.df['nominal']*down
        self.df[columns]  = self.df[columns].multiply(wgt, axis=0)
        self.variations.append(name)

    def add_only_variations(self, name, up, down):
        columns = self.df.columns
        self.df[f'{name}_up'] = self.df['nominal']*up
        self.df[f'{name}_down'] = self.df['nominal']*down
        self.variations.append(name)
        
    def get_weight(self, name, mask=np.array([])):
        if len(mask)==0:
            mask = np.ones(self.df.shape[0], dtype=bool)        
        if (name in self.df.columns):
            return self.df[name].to_numpy()[mask]
        else:
            return np.array([])
       
    def effect_on_normalization(self, mask=np.array([])):
        if len(mask)==0:
            mask = np.ones(self.df.shape[0], dtype=int)
        for var in self.variations:
            wgt_off = self.df[f'{var}_off'][mask].to_numpy().sum()
            wgt_on = self.df['nominal'][mask].to_numpy().sum()
            effect = (wgt_on - wgt_off)/wgt_on*100
            if effect < 0:
                ef = round(-effect, 2)
                print(f'Enabling {var} decreases yield by {ef}%')
            else:
                ef = round(effect, 2)
                print(f'Enabling {var} increases yield by {ef}%')
                
