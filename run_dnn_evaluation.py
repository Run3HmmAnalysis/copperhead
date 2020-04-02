import multiprocessing as mp
import pandas as pd
import numpy as np
import os, sys

from config.variables import variables as variables_
from config.parameters import parameters
import pandas as pd
import glob
from coffea import util
import warnings
warnings.filterwarnings('ignore')

do_unbin=True
do_bin=False


year = '2016'
path_label = 'mar25'
do_jer = False
do_jecunc = False
#if not do_jer:
#    path_label += "_nojer"
if do_jecunc:
    path_label += "_jecunc"

nbins=12
load_path = f'/depot/cms/hmm/coffea/all_{year}_{path_label}/unbinned/'
load_path_binned = f'/depot/cms/hmm/coffea/all_{year}_{path_label}/binned/'
dnn_label = '2017'
label = 'm125'
tmp_path = f'/depot/cms/hmm/coffea/tmp_{label}_{year}_{path_label}/'
tmp_path_dnn = f'/depot/cms/hmm/coffea/tmp_{label}_{year}_{path_label}_dnn/'
systematics = ['nominal', 'muSF_up', 'muSF_down', 'pu_weight_up', 'pu_weight_down']
if '2018' not in year:
    systematics = systematics + ['l1prefiring_weight_up', 'l1prefiring_weight_down']
regions = ['z-peak','h-sidebands', 'h-peak']

parameters = {k:v[year] for k,v in parameters.items()}

print(load_path)
print(load_path_binned)

def load_sample(s):
    print(f"Adding {s}")

#    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import numpy as np
    from config.parameters import training_features

#    from tensorflow.keras.models import load_model
#    config = tf.compat.v1.ConfigProto(
#                        intra_op_parallelism_threads=1, 
#                        inter_op_parallelism_threads=1, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU': 1})
#    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#    sess = tf.compat.v1.Session(config=config)
#    K.set_session(sess)
    
    global label
    global load_path
    global load_path_binned
    global nbins
    global regions
    global parameters
    
    variables = [v.name for v in variables_]
    for syst in systematics:
        variables.append(f'weight_{syst}')
    
    chunked = True
    prefix = ''
    scalers_path = f'output/trained_models/scalers_{dnn_label}.npy'
    model_path = f'output/trained_models/test_{dnn_label}.h5'

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
        proc_outs_binned = []
        paths_binned = glob.glob(f"{load_path_binned}/{prefix}{s}_?.coffea")
        for p in paths_binned:
            proc_outs_binned.append(util.load(p))
            
    else:
        proc_outs = [util.load(f"{load_path}/{prefix}{s}.coffea")]
        proc_outs_binned = [util.load(f"{load_path_binned}/{prefix}{s}.coffea")]


    from tensorflow.keras.models import load_model
    config = tf.compat.v1.ConfigProto(
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    sess = tf.compat.v1.Session(config=config)
    with sess:    
        dnn_model = load_model(model_path)
        for ip, proc_out in enumerate(proc_outs):
            if not do_unbin: continue
            for r in regions:
                for c in channels:
                    lbl = f'{c}_channel_{s}_{r}_{ip}'
                    df = pd.DataFrame(columns=variables)
                    dfs_out[lbl] = pd.DataFrame(columns=['dnn_score', 'dataset', 'region', 'channel', 'weight'])
                    len_=0
                    for v in variables:
#                        print(v)
                        if v not in training_features+['event']+[f'weight_{s}' for s in systematics]: continue
                        if (v=='dimuon_mass') and ('h-peak' not in r):
                            df[v] = np.full(len(proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value), 125.)
                        elif 'weight' not in v:
                            len_ = len(proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value)
                            df[v] = proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value
                        else:
                            if 'data' in s:
                                df[v] = proc_out[f'weight_nominal_unbin_{s}_c_{c}_r_{r}'].value
                            else:
                                #                            print(r,c,v)
                                df[v] = proc_out[f'{v}_unbin_{s}_c_{c}_r_{r}'].value[0:len_]
                    print(df['weight_nominal'].sum())
                    if s in norm[r]:
                        norm[r][s] += df['weight_nominal'].sum()
                    else:
                        norm[r][s] = df['weight_nominal'].sum()
                    if 'data' in s:
                        df_test = df
                    else:
                        train_fraction = 0.6
                        df_test = df[df['event']%(1/train_fraction)>=1]

                    df_test = (df_test[training_features]-scalers[0])/scalers[1]
                    if df.shape[0]>0:
                        try:
                            dfs_out[lbl]['dnn_score'] = dnn_model.predict(df_test.reset_index(drop=True)).flatten()
                        except:
                            dfs_out[lbl]['dnn_score'] = dnn_model.predict(df_test.reset_index(drop=True))
                        dfs_out[lbl]['dataset'] = s
                        dfs_out[lbl]['region'] = r
                        dfs_out[lbl]['channel'] = c
                        dfs_out[lbl]['weight'] = df['weight_nominal'].reset_index(drop=True)
                        for syst in systematics:
                            dfs_out[lbl][f'weight_{syst}'] = df[f'weight_{syst}'].reset_index(drop=True)

    df_from_unbin = pd.concat(list(dfs_out.values())).reset_index(drop=True)

    dnn_bins = []
    for i in range(nbins):
        dnn_bins.append(f'dnn_{i}')
    #dfs_out_binned = {"placeholder": pd.DataFrame(columns=dnn_bins+['variation', 'dataset', 'region', 'channel', 'weight'])}
    df_dnn = pd.DataFrame(columns=dnn_bins+['variation', 'dataset', 'region', 'channel'])
    for ip, proc_out in enumerate(proc_outs_binned):
        if not do_bin: continue
        for r in regions:
            for c in channels:
                for v in parameters["jec_unc_to_consider"]+["nominal"]:
                    try:
                        if v=="nominal":
                            row = pd.Series(list(proc_out[f'dnn_score_nominal'].values()[(s, r, c, syst)]) + ['nominal', s, r, c], index=df_dnn.columns)
                            df_dnn  = df_dnn.append(row, ignore_index=True)
                        else:
                            for ud in ["_up", "_down"]:
                                syst='nominal'
                                row = pd.Series(list(proc_out[f'dnn_score_{v}{ud}'].values()[(s, r, c, syst)]) + [f'{v}{ud}', s, r, c], index=df_dnn.columns)
                                df_dnn  = df_dnn.append(row, ignore_index=True)
                    except:
                        pass
                        #print(f"Error: {s} {r} {c} {v}")

    print(f"Done: {s}")
    return s,df_from_unbin, df_dnn, norm

def load_data(samples, out_path, out_path_dnn, parallelize=True):
    global regions
    norms  = {r:{} for r in regions}
    if do_unbin:
        try:
            os.mkdir(out_path)
        except OSError as error: 
            print(error)
    if do_bin:
        try:
            os.mkdir(out_path_dnn)
        except OSError as error:
            print(error)        
    if parallelize:
        pool = mp.Pool(mp.cpu_count()-4)
        a = [pool.apply_async(load_sample, args=(s,)) for s in samples]
        results = []
        for process in a:
            process.wait()
            s, res_unbin, res_dnn, norm = process.get()
            for r in regions:
                norms[r].update(norm[r])
            if do_unbin:
                np.save(f'{out_path}/temp_{s}', res_unbin)
            if do_bin:
                np.save(f'{out_path_dnn}/temp_{s}', res_dnn)
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
#       'data_A', 'data_B','data_C','data_D','data_E','data_F','data_G','data_H',
##

## MC for DNN training (Purdue): ##
     'dy_m105_160_amc',
     'dy_m105_160_vbf_amc',

     'ggh_amcPS',
     'vbf_amcPS',
     'ggh_powhegPS', 
     'vbf_powhegPS',

#    'vbf_powheg_herwig',
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
#    'wz_1l1nu2q',
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
norms = load_data(samples, tmp_path, tmp_path_dnn, parallelize = True)
print(norms)
if do_unbin:
    with open(f'output/norms_{year}.json', 'w') as fp:
        json.dump(norms, fp)
