import os, sys
import glob
from dask.distributed import get_client, wait, as_completed
from functools import partial
import lz4.frame
import _pickle as pkl
import copy

import pandas as pd
import numpy as np
import boost_histogram as bh
from boost_histogram import loc

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from config.variables import variables_lookup, Variable

training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_pt_log', 'dimuon_eta', 'dimuon_mass_res', 'dimuon_mass_res_rel',\
                     'dimuon_cos_theta_cs', 'dimuon_phi_cs',
                     'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_qgl',\
                     'jj_mass', 'jj_mass_log', 'jj_dEta', 'rpt', 'll_zstar_log', 'mmj_min_dEta', 'nsoftjets5', 'htsoft2'
                    ] 

grouping = {
    'data_A': 'Data',
    'data_B': 'Data',
    'data_C': 'Data',
    'data_D': 'Data',
    'data_E': 'Data',
    'data_F': 'Data',
    'data_G': 'Data',
    'data_H': 'Data',
    'dy_0j': 'DY',
    'dy_1j': 'DY',
    'dy_2j': 'DY',
#    'dy_m105_160_amc': 'DY_nofilter',
#    'dy_m105_160_vbf_amc': 'DY_filter',
    'dy_m105_160_amc': 'DY',
    'dy_m105_160_vbf_amc': 'DY',
    'ewk_lljj_mll105_160_ptj0': 'EWK',
#    'ewk_lljj_mll105_160_py_dipole': 'EWK_Pythia',
    'ttjets_dl': 'TT+ST',
    'ttjets_sl': 'TT+ST',
    'ttw': 'TT+ST',
    'ttz': 'TT+ST',
    'st_tw_top': 'TT+ST',
    'st_tw_antitop': 'TT+ST',
    'ww_2l2nu': 'VV',
    'wz_2l2q': 'VV',
    'wz_1l1nu2q': 'VV',
    'wz_3lnu': 'VV',
    'zz': 'VV',
    'www': 'VVV',
    'wwz': 'VVV',
    'wzz': 'VVV',
    'zzz': 'VVV',
    'ggh_amcPS': 'ggH',
    'vbf_powheg_dipole': 'VBF',
}

decorrelation_scheme = {
    'LHERen': {'DY':['DY'], 'EWK':['EWK'], 'ggH':['ggH'], 'TT+ST':['TT+ST']},
    'LHEFac': {'DY':['DY'], 'EWK':['EWK'], 'ggH':['ggH'], 'TT+ST':['TT+ST']},
    'pdf_2rms': {'DY':['DY'], 'ggH':['ggH'], 'VBF':['VBF']},
    'pdf_mcreplica': {'DY':['DY'], 'ggH':['ggH'], 'VBF':['VBF']},
#    'LHERen': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'], 'ggH':['ggH'], 'TT+ST':['TT+ST']},
#    'LHEFac': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'], 'ggH':['ggH'], 'TT+ST':['TT+ST']},
#    'pdf_2rms': {'DY':['DY_filter', 'DY_nofilter'], 'ggH':['ggH'], 'VBF':['VBF']},
#    'pdf_mcreplica': {'DY':['DY_filter', 'DY_nofilter'], 'ggH':['ggH'], 'VBF':['VBF']},
}

def workflow(client, argsets, parameters):
    futures = client.map(partial(load_data, parameters=parameters), argsets)

    for model in parameters['dnn_models']:
        futures = client.map(partial(dnn_evaluation, parameters=parameters, model=model),futures)
    for model in parameters['bdt_models']:
        futures = client.map(partial(bdt_evaluation, parameters=parameters, model=model),futures)

    hist_args = []
    for future in futures:
        for var in parameters['hist_vars']:
            hist_args.append({'args':future, 'var':var})
    hist_futures = client.map(partial(get_histogram,parameters=parameters),hist_args)

    df = pd.DataFrame()
    for future in as_completed(futures):
        result = future.result()
        if 'df' in result.keys():
            df = pd.concat([df, future.result()['df']])

    hist = pd.DataFrame()
    for future in as_completed(hist_futures):
        hist = pd.concat([hist, future.result()[0]])
        
    return df, hist
    
def load_data(args, parameters):
    import dask
    with lz4.frame.open(args['input']) as fin:
        input_ = pkl.load(fin)
    suff = f"_{args['c']}_{args['r']}"

    df = pd.DataFrame()

    len_ = len(input_['wgt_nominal'+suff].value)
    if len_==0: return args
    
    columns = sorted([c.replace(suff, '') for c in list(input_.keys()) if suff in c])
    for var in columns:
        if 'wgt_' in var:
            try:
                df[var] = input_[var+suff].value
            except:
                df[var] = input_['wgt_nominal'+suff].value
        else:
            try:
                df[var] = input_[var+suff].value
            except:
                pass
    df['year']=int(args['y'])
    df['c'] = args['c']
    df['r'] = args['r']
    df['s'] = args['s']
    df['v'] = args['v']

    ret = copy.deepcopy(args)
    ret.update(**{'df':df})
    return ret

def prepare_features(df, parameters, add_year=False):
    global training_features
    features = copy.deepcopy(training_features)
    df['dimuon_pt_log'] = np.log(df['dimuon_pt'])
    df['jj_mass_log'] = np.log(df['jj_mass'])
    if add_year and ('year' not in features):
        features+=['year']
    if not add_year and ('year' in features):
        features = [t for t in features if t!='year']
    for trf in features:
        if trf not in df.columns:
            print(f'Variable {trf} not found in training dataframe!')
    return df, features

def dnn_evaluation(args, model, parameters):
    if not args: return args
    if 'df' not in args: return args
    df = args['df']
    if df.shape[0]==0: return args

    import keras.backend as K
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    config = tf.compat.v1.ConfigProto(
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    sess = tf.compat.v1.Session(config=config)
    if parameters['do_massscan']:
        mass_shift = args['mass']-125.0
    allyears = ('allyears' in model)
    df, features = prepare_features(df, parameters, allyears)
    score_name = f'score_{model}'
    df[score_name] = 0
    with sess:
        nfolds = 4
        for i in range(nfolds):
            if allyears:
                label = f"allyears_jul7_{i}"
            else:
                label = f"{args['y']}_jul7_{i}"

            train_folds = [(i+f)%nfolds for f in [0,1]]
            val_folds = [(i+f)%nfolds for f in [2]]
            eval_folds = [(i+f)%nfolds for f in [3]]

            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            scalers_path = f"{parameters['models_path']}/{model}/scalers_{label}.npy"
            scalers = np.load(scalers_path)
            model_path = f"{parameters['models_path']}/{model}/dnn_{label}.h5"
            dnn_model = load_model(model_path)
            df_i = df[eval_filter]
            if args['r']!='h-peak':
                df_i['dimuon_mass'] = 125.
            if parameters['do_massscan']:
                df_i['dimuon_mass'] = df_i['dimuon_mass']-mass_shift

            df_i = (df_i[features]-scalers[0])/scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return args

def bdt_evaluation(args, model, parameters):
    if not args: return args
    if 'df' not in args: return args
    df = args['df']
    if df.shape[0]==0: return args

    import xgboost as xgb
    import pickle
    if parameters['do_massscan']:
        mass_shift = args['mass']-125.0
    allyears = ('allyears' in model)
    df, features = prepare_features(df, parameters, allyears)
    score_name = f'score_{model}'
    df[score_name] = 0

    nfolds = 4
    for i in range(nfolds):
        if allyears:
            label = f"allyears_jul7_{i}"
        else:
            label = f"{args['y']}_jul7_{i}"
                    
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]
            
        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        scalers_path = f"{parameters['models_path']}/{model}/scalers_{label}.npy"
        scalers = np.load(scalers_path)
        model_path = f"{parameters['models_path']}/{model}/BDT_model_earlystop50_{label}.pkl"   

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        if args['r']!='h-peak':
            df_i['dimuon_mass'] = 125.
        if parameters['do_massscan']:
            df_i['dimuon_mass'] = df_i['dimuon_mass']-mass_shift
        df_i = (df_i[features]-scalers[0])/scalers[1]
        #prediction = np.array(bdt_model.predict_proba(df_i)[:, 1]).ravel()
        if len(df_i)>0:
            if 'multiclass' in model:
                prediction = np.array(bdt_model.predict_proba(df_i.values)[:, 5]).ravel()
            else:
                prediction = np.array(bdt_model.predict_proba(df_i.values)[:, 1]).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return args

def get_histogram(hist_args, parameters):
    args = hist_args['args']
    var = hist_args['var']
    if not args: return
    if 'df' not in args: return None, None
    df = args['df']
    if df is None: return None,None
    if len(df)==0: return None,None
    year = args['y']
    mva_bins = parameters['mva_bins']
    if var in variables_lookup.keys():
        var = variables_lookup[var]
    else:
        var = Variable(var, var, 50, 0, 5)
    
    if ('score' in var.name):
        bins = mva_bins[var.name.replace('score_','')][year]
    dataset_axis = bh.axis.StrCategory(df.s.unique())
    region_axis = bh.axis.StrCategory(df.r.unique())
    channel_axis = bh.axis.StrCategory(df.c.unique())
    syst_axis = bh.axis.StrCategory(df.v.unique())
    val_err_axis = bh.axis.StrCategory(['value', 'sumw2'])
    if ('score' in var.name) and len(bins)>0:
        var_axis = bh.axis.Variable(bins)
        nbins = len(bins)-1
    else:
        var_axis = bh.axis.Regular(var.nbins, var.xmin, var.xmax)
        nbins = var.nbins
    df_out = pd.DataFrame()
    edges = []
    regions = df.r.unique()
    channels = df.c.unique()
    for s in df.s.unique():
        if ('data' in s):
            if ('nominal' in parameters['syst_variations']):
                syst_variations = ['nominal']
                wgts = ['wgt_nominal']
            else:
                syst_variations = []
                wgts = []
        else:
            syst_variations = parameters['syst_variations']
            wgts = [c for c in df.columns if ('wgt_' in c)]
        mcreplicas = [c for c in df.columns if ('mcreplica' in c)]
        mcreplicas = []
        if len(mcreplicas)>0:
            wgts = [wgt for wgt in wgts if ('pdf_2rms' not in wgt)]
        if len(mcreplicas)>0 and ('wgt_nominal' in df.columns) and (s in grouping.keys()):
            decor = decorrelation_scheme['pdf_mcreplica']
            for decor_group, proc_groups in decor.items():
                for imcr, mcr in enumerate(mcreplicas):
                    wgts += [f'pdf_mcreplica{imcr}_{decor_group}']
                    if grouping[s] in proc_groups:
                        df.loc[:,f'pdf_mcreplica{imcr}_{decor_group}'] = np.multiply(df.wgt_nominal,df[mcr])
                    else:
                        df.loc[:,f'pdf_mcreplica{imcr}_{decor_group}'] = df.wgt_nominal
        for w in wgts:
            hist = bh.Histogram(dataset_axis, region_axis, channel_axis, syst_axis, val_err_axis, var_axis)
            hist.fill(df.s.to_numpy(), df.r.to_numpy(), df.c.to_numpy(), df.v.to_numpy(), 'value',\
                              df[var.name].to_numpy(), weight=df[w].to_numpy())
            hist.fill(df.s.to_numpy(), df.r.to_numpy(), df.c.to_numpy(), df.v.to_numpy(), 'sumw2',\
                              df[var.name].to_numpy(), weight=(df[w]*df[w]).to_numpy())    
            for v in df.v.unique():
                if v not in syst_variations: continue
                if (v!='nominal')&(w!='wgt_nominal'): continue
                for r in regions:
                    for c in channels:
                        values = hist[loc(s), loc(r), loc(c), loc(v), loc('value'), :].to_numpy()[0]
                        sumw2 = hist[loc(s), loc(r), loc(c), loc(v), loc('sumw2'), :].to_numpy()[0]
                        values[values<0] = 0
                        sumw2[values<0] = 0
                        integral = values.sum()
                        edges = hist[loc(s), loc(r), loc(c), loc(v), loc('value'), :].to_numpy()[1]
                        contents = {}
                        contents.update({f'bin{i}':[values[i]] for i in range(nbins)})
                        contents.update({f'sumw2_0': 0.}) # add a dummy bin b/c sumw2 indices are shifted w.r.t bin indices
                        contents.update({f'sumw2_{i+1}':[sumw2[i]] for i in range(nbins)})
                        contents.update({'s':[s],'r':[r],'c':[c], 'v':[v], 'w':[w],\
                                         'var':[var.name], 'integral':integral})
                        contents['g'] = grouping[s] if s in grouping.keys() else f"{s}"
                        row = pd.DataFrame(contents)
                        df_out = pd.concat([df_out, row], ignore_index=True)
        if df_out.shape[0]==0:
            return df_out, edges
        bin_names = [n for n in df_out.columns if 'bin' in n]
        sumw2_names = [n for n in df_out.columns if 'sumw2' in n]
        pdf_names = [n for n in df_out.w.unique() if ("mcreplica" in n)]
        for decor_group, proc_groups in decorrelation_scheme['pdf_mcreplica'].items():
            if len(pdf_names)==0: continue
            for r in regions:
                for c in channels:
                    rms = df_out.loc[df_out.w.isin(pdf_names)&(df_out.v=='nominal')&(df_out.r==r)&(df_out.c==c), bin_names].std().values
                    nom = df_out[(df_out.w=='wgt_nominal')&(df_out.v=='nominal')&(df_out.r==r)&(df_out.c==c)]
                    nom_bins = nom[bin_names].values[0]
                    nom_sumw2 = nom[sumw2_names].values[0]
                    row_up = {}
                    row_up.update({f'bin{i}':[nom_bins[i]+rms[i]] for i in range(nbins)})
                    row_up.update({f'sumw2_{i}':[nom_sumw2[i]] for i in range(nbins+1)})
                    row_up.update({f'sumw2_{i+1}':[sumw2[i]] for i in range(nbins)})
                    row_up.update({'s':[s],'r':[r],'c':[c], 'v':['nominal'], 'w':[f'pdf_mcreplica_{decor_group}_up'],\
                                     'var':[var.name], 'integral':(nom_bins.sum()+rms.sum())})
                    row_up['g'] = grouping[s] if s in grouping.keys() else f"{s}"
                    row_down = {}
                    row_down.update({f'bin{i}':[nom_bins[i]-rms[i]] for i in range(nbins)})
                    row_down.update({f'sumw2_{i}':[nom_sumw2[i]] for i in range(nbins+1)})
                    row_down.update({f'sumw2_{i+1}':[sumw2[i]] for i in range(nbins)})
                    row_down.update({'s':[s],'r':[r],'c':[c], 'v':['nominal'], 'w':[f'pdf_mcreplica_{decor_group}_down'],\
                                     'var':[var.name], 'integral':(nom_bins.sum()-rms.sum())})
                    row_down['g'] = grouping[s] if s in grouping.keys() else f"{s}"
                    df_out = pd.concat([df_out, pd.DataFrame(row_up), pd.DataFrame(row_down)],ignore_index=True)
                    df_out = df_out[~(df_out.w.isin(mcreplicas)&(df_out.v=='nominal')&(df_out.r==r)&(df_out.c==c))]

    return df_out, edges