import multiprocessing as mp
import pandas as pd
import numpy as np
import os, sys

from config.variables import variables as variables_
import pandas as pd
import glob
from coffea import util
import warnings
warnings.filterwarnings('ignore')

year = '2017'
load_path = f'/depot/cms/hmm/coffea/all_{year}_feb23/'
dnn_label = '2017'
label = 'm125'
tmp_path = f'/depot/cms/hmm/coffea/tmp_{label}_{year}_feb23/'
systematics = ['nominal', 'muSF_up', 'muSF_down', 'pu_weight_up', 'pu_weight_down']
if '2018' not in year:
    systematics = systematics + ['l1prefiring_weight_up', 'l1prefiring_weight_down']
regions = ['z-peak','h-sidebands', 'h-peak']

def load_sample(s):
    print(f"Adding {s}")

    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import numpy as np
    
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads=1
    config.inter_op_parallelism_threads=1
    K.set_session(tf.Session(config=config))
    
    global label
    global load_path
    global regions
    
    variables = [v.name for v in variables_]
    for syst in systematics:
        variables.append(f'weight_{syst}')
    
    chunked = True
    prefix = ''
    scalers_path = f'output/trained_models/scalers_{dnn_label}.npy'
    model_path = f'output/trained_models/test_{dnn_label}.h5'
    
    training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR',\
         'jj_mass', 'jj_eta', 'jj_phi', 'jj_pt', 'jj_dEta',\
         'mmjj_mass', 'mmjj_eta', 'mmjj_phi','zeppenfeld',\
         'jet1_pt', 'jet1_eta', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_qgl',\
         'dimuon_cosThetaCS',\
         'dimuon_mass_res_rel', 'deta_mumuj1', 'dphi_mumuj1', 'deta_mumuj2', 'dphi_mumuj2',\
         'htsoft5',
        ]

    dnn_model = load_model(model_path)
    scalers = np.load(scalers_path)

    dfs_out = {"placeholder": pd.DataFrame(columns=['dnn_score', 'dataset', 'region', 'channel', 'weight'])}

#     channels = ['ggh_01j', 'ggh_2j','vbf']
    channels = ['vbf']
    norm  = {r:{} for r in regions}
    if chunked:
        proc_outs = []
        paths = glob.glob(f"{load_path}/{prefix}{s}_?.coffea")
        for p in paths:
            proc_outs.append(util.load(p))
    else:
        proc_outs = [util.load(f"{path}/{prefix}{s}.coffea")]
    for ip, proc_out in enumerate(proc_outs):
        for r in regions:
            for c in channels:
                lbl = f'{c}_channel_{s}_{r}_{ip}'
                df = pd.DataFrame(columns=variables)
                dfs_out[lbl] = pd.DataFrame(columns=['dnn_score', 'dataset', 'region', 'channel', 'weight'])
                len_=0
                for v in variables:
                    if v not in training_features+['event','event_weight']+[f'weight_{s}' for s in systematics]: continue
                    if (v=='dimuon_mass') and ('h-peak' not in r):
                        df[v] = np.full(len(proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value), 125.)
                    elif 'weight' not in v:
                        len_ = len(proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value)
                        df[v] = proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value
                    else:
                        if 'data' in s:
                            df[v] = proc_out[f'event_weight_unbin_{s}_c_{c}_r_{r}'].value
                        else:
#                            print(r,c,v)
                            df[v] = proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value[0:len_]
                if s in norm[r]:
                    norm[r][s] += df['weight_nominal'].sum()
                else:
                    norm[r][s] = df['weight_nominal'].sum()
                if 'data' in s:
                    df_test = df
                else:
                    train_fraction = 0.6
                    df_test = df[df['event']%(1/train_fraction)>=1]
#                 df = df.iloc[0:100]
#                print(r,c,df_test)
                df_test = (df_test[training_features]-scalers[0])/scalers[1]
                if df.shape[0]>0:
                    dfs_out[lbl]['dnn_score'] = dnn_model.predict(df_test.reset_index(drop=True)).flatten()
                    dfs_out[lbl]['dataset'] = s
                    dfs_out[lbl]['region'] = r
                    dfs_out[lbl]['channel'] = c
                    dfs_out[lbl]['weight'] = df['event_weight'].reset_index(drop=True)
                    for syst in systematics:
                        dfs_out[lbl][f'weight_{syst}'] = df[f'weight_{syst}'].reset_index(drop=True)


    result = pd.concat(list(dfs_out.values())).reset_index(drop=True)
    print(f"Done: {s}")
    return s,result, norm

def load_data(samples, out_path, parallelize=True):
    global regions
    norms  = {r:{} for r in regions}
    try:
        os.mkdir(out_path)
    except OSError as error: 
        print(error)
    if parallelize:
        pool = mp.Pool(mp.cpu_count()-4)
        a = [pool.apply_async(load_sample, args=(s,)) for s in samples]
        results = []
        for process in a:
            process.wait()
            s, res, norm = process.get()
            for r in regions:
                norms[r].update(norm[r])
            np.save(f'{out_path}/temp_{s}', res)
        pool.close()
    else:
        for s in samples:
            load_sample(s)
    return norms

training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR',\
         'jj_mass', 'jj_eta', 'jj_phi', 'jj_pt', 'jj_dEta',\
         'mmjj_mass', 'mmjj_eta', 'mmjj_phi','zeppenfeld',\
         'jet1_pt', 'jet1_eta', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_qgl',\
         'dimuon_cosThetaCS',\
         'dimuon_mass_res_rel', 'deta_mumuj1', 'dphi_mumuj1', 'deta_mumuj2', 'dphi_mumuj2',\
         'htsoft5',
        ]


classes = {
    'data': ['data_A', 'data_B','data_C','data_D','data_E','data_F','data_G','data_H'], 
    'DY': ['dy', 'dy_0j', 'dy_1j', 'dy_2j', 'dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'dy_m105_160_mg', 'dy_m105_160_vbf_mg'],
    'EWK': ['ewk_lljj_mll50_mjj120','ewk_lljj_mll105_160', 'ewk_lljj_mll105_160_ptj0'],
    'TTbar + Single Top':['ttjets_dl', 'ttjets_sl', 'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
    'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
}

samples = [
## Data: ##
       'data_A', 'data_B','data_C','data_D','data_E','data_F','data_G','data_H',
##

## MC for DNN training (Purdue): ##
     'dy_m105_160_amc',
     'dy_m105_160_vbf_amc',

     'ggh_amcPS',
     'vbf_amcPS',
     'ggh_powhegPS', 
     'vbf_powhegPS', 

# #     'ggh_amcPS_m120',
# #     'vbf_amcPS_m120',
# #     'ggh_powhegPS_m120',
# #     'vbf_powhegPS_m120', 

# #     'ggh_amcPS_m130',
# #     'vbf_amcPS_m130',
# #     'ggh_powhegPS_m130',
# #     'vbf_powhegPS_m130', 
    

    'ttjets_dl', 
# # ##
    
# # # ## MC for DNN training (Legnaro): ##    
#    "ewk_lljj_mll105_160_ptj0", 
# # # ##    
    
# # # ## Most important of other MC: ##   
    'dy_0j',
    'dy_1j', 
    'dy_2j',
     'ewk_lljj_mll50_mjj120',
# # # ##

# # # ## Less important other MC: ##
    'ttz',
    'ttw',
    'st_tw_top','st_tw_antitop',
    'ww_2l2nu',
    'wz_2l2q',
    'wz_3lnu',
    'wz_1l1nu2q',
    'zz',
    'www','wwz',
    'wzz','zzz',
# # # ##

    
   'ttjets_sl',
   'ewk_lljj_mll105_160',
#     
]

import pandas as pd
import json
norms = load_data(samples, tmp_path, parallelize = True)
print(norms)
with open(f'output/norms_{year}.json', 'w') as fp:
    json.dump(norms, fp)
