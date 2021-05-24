import pandas as pd
import numpy as np


class Weights(object):
    def __init__(self, data):
        self.df = pd.DataFrame(1., index=data.index, columns=['nominal'])
        self.wgts = pd.DataFrame(index=data.index)
        self.variations = []

    def add_weight(self, name, wgt=None, how='nom'):
        if (wgt is None) and ('dummy' not in how):
            return

        if how == 'nom':
            # add only nominal weight
            self.add_nom_weight(name, wgt)
        elif how == 'all':
            # add weight with up and down variations
            nom = wgt['nom']
            up = wgt['up']
            down = wgt['down']
            self.add_weight_with_variations(name, nom, up, down)
        elif how == 'only_vars':
            # add only variations
            up = wgt['up']
            down = wgt['down']
            self.add_only_variations(name, up, down)

        elif (how == 'dummy_nom') or (how == 'dummy'):
            self.add_dummy_weight(name, nom=True, variations=False)
        elif how == 'dummy_all':
            self.add_dummy_weight(name, nom=True, variations=True)
        elif how == 'dummy_vars':
            self.add_dummy_weight(name, nom=False, variations=True)

    def add_nom_weight(self, name, wgt):
        columns = self.df.columns
        self.df[f'{name}_off'] = self.df['nominal']
        self.df[columns] = self.df[columns]\
            .multiply(np.array(wgt), axis=0).astype(np.float64)
        self.variations.append(name)
        self.wgts[name] = wgt

    def add_weight_with_variations(self, name, wgt, up, down):
        columns = self.df.columns
        self.wgts[name] = wgt
        nom = self.df['nominal']
        self.df[f'{name}_off'] = nom
        self.df[f'{name}_up'] = (nom * up).astype(np.float64)
        self.df[f'{name}_down'] = (nom * down).astype(np.float64)
        self.df[columns] = self.df[columns]\
            .multiply(wgt, axis=0).astype(np.float64)
        self.variations.append(name)

    def add_only_variations(self, name, up, down):
        nom = self.df['nominal']
        self.df[f'{name}_up'] = (nom * up).astype(np.float64)
        self.df[f'{name}_down'] = (nom * down).astype(np.float64)
        self.variations.append(name)

    def add_dummy_weight(self, name, nom=True, variations=False):
        self.variations.append(name)
        if nom:
            self.df[f'{name}_off'] = self.df['nominal']
            self.wgts[name] = 1.
        if variations:
            self.df[f'{name}_up'] = np.nan
            self.df[f'{name}_down'] = np.nan
            self.df[f'{name}_up'] =\
                self.df[f'{name}_up'].astype(np.float64)
            self.df[f'{name}_down'] =\
                self.df[f'{name}_down'].astype(np.float64)

    def get_weight(self, name, mask=np.array([])):
        if len(mask) == 0:
            mask = np.ones(self.df.shape[0], dtype=bool)
        if (name in self.df.columns):
            return self.df[name].to_numpy()[mask]
        else:
            return np.array([])

    def effect_on_normalization(self, mask=np.array([])):
        if len(mask) == 0:
            mask = np.ones(self.df.shape[0], dtype=int)
        for var in self.variations:
            if f'{var}_off' not in self.df.columns:
                continue
            wgt_off = self.df[f'{var}_off'][mask].to_numpy().sum()
            wgt_on = self.df['nominal'][mask].to_numpy().sum()
            effect = (wgt_on - wgt_off) / wgt_on * 100
            if effect < 0:
                ef = round(-effect, 2)
                print(f'Enabling {var} decreases yield by {ef}%')
            else:
                ef = round(effect, 2)
                print(f'Enabling {var} increases yield by {ef}%')
