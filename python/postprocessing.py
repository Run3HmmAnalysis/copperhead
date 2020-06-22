import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import coffea
from coffea import util
import glob
from config.parameters import training_features
import boost_histogram as bh
from boost_histogram import loc
import numpy as np
import uproot
from uproot_methods.classes.TH1 import from_numpy
import matplotlib.pyplot as plt
import mplhep as hep
import tqdm

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
    'dy_m105_160_amc': 'DY_nofilter',
    'dy_m105_160_vbf_amc': 'DY_filter',
    'ewk_lljj_mll105_160_ptj0': 'EWK',
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
    'www': 'VVV',
    'wwz': 'VVV',
    'wzz': 'VVV',
    'zzz': 'VVV',
    'ggh_amcPS': 'ggH',
    'vbf_powheg_dipole': 'VBF',
}

def worker(args):
    if 'to_pandas' not in args['modules']:
        print("Need to convert to Pandas DF first!")
        return
    df = to_pandas(args)
    if 'dnn_evaluation' in args['modules']:
        df = dnn_evaluation(df, args)
    hists = {}
    edges = {}
    if 'get_hists' in args['modules']:
        for var in args['vars_to_plot']:
            hists[var.name], edges[var.name] = get_hists(df, var, args, bins=args['dnn_bins'])
    return df, hists, edges


def postprocess(args, parallelize=True):
    dataframes = []
    hist_dfs = {}
    edges_dict = {}
    path = args['in_path']
    argsets = []
    all_training_samples = []
    classes_dict = {}
    if 'training_samples' in args:
        for cl, sm in args['training_samples'].items():
            all_training_samples.extend(sm)
            for smp in sm:
                classes_dict[smp] = cl
    args.update({'classes_dict':classes_dict})

    for s in args['samples']:
        if (s in grouping.keys()) and (('dy' in s) or ('ewk' in s) or ('vbf' in s) or ('ggh' in s)):
            variations = args['syst_variations']
        else:
            variations = ['nominal']
        for v in variations:
            proc_outs = glob.glob(f"{path}/nominal/{s}.coffea") if v=='nominal' else glob.glob(f"{path}/{v}/{s}.coffea")
            for proc_path in proc_outs:
                for c in args['channels']:
                    for r in args['regions']:
                        argset = args.copy()
                        argset.update(**{'proc_path':proc_path,'s':s,'c':c,'r':r,'v':v})
                        argsets.append(argset)
    if len(argsets)==0:
        print("Nothing to load! Check the arguments.")
        sys.exit(0)

    if parallelize:
        cpus = mp.cpu_count()-2
        print(f'Parallelizing over {cpus} CPUs')

        pbar = tqdm.tqdm(total=len(argsets))
        def update(*a):
            pbar.update()

        pool = mp.Pool(cpus)
        a = [pool.apply_async(worker, args=(argset,), callback=update) for argset in argsets]
        results = []
        for process in a:
            process.wait()
            df, hists, edges = process.get()
            dataframes.append(df)
            for var, hist in hists.items():
                if (var in edges_dict.keys()):
                    if edges_dict[var] == []:
                        edges_dict[var] = edges[var]
                else:
                    edges_dict[var] = edges[var]
                if var in hist_dfs.keys():
                    hist_dfs[var].append(hist)
                else:
                    hist_dfs[var] = [hist]
        pool.close()
    else:
        for argset in argsets:
            df, hists, edges = worker(argset)
            dataframes.append(df)
            for var, hist in hists.items():
                if (var in edges_dict.keys()):
                    if edges_dict[var] == []:
                        edges_dict[var] = edges[var]
                else:
                    edges_dict[var] = edges[var]
                if var in hist_dfs.keys():
                    hist_dfs[var].append(hist)
                else:
                    hist_dfs[var] = [hist]
    return dataframes, hist_dfs, edges_dict

# for unbinned processor output (column_accumulators)
def to_pandas(args):
    proc_out = util.load(args['proc_path'])
    c = args['c']
    r = args['r']
    s = args['s']
    v = args['v']
    suff = f'_{c}_{r}'

    groups = ['DY_nofilter', 'DY_filter', 'EWK', 'TT+ST', 'VV', 'ggH', 'VBF']
    columns = [c.replace(suff, '') for c in list(proc_out.keys()) if suff in c]
    df = pd.DataFrame()

    len_ = len(proc_out[f'dimuon_mass_{c}_{r}'].value) if f'dimuon_mass_{c}_{r}' in proc_out.keys() else 0

    for var in columns:
        # if DNN training: get only relevant variables
        if args['training'] and (var not in training_features+['event', 'wgt_nominal']): continue
        # possibility to ignore weight variations
        if (not args['wgt_variations']) and ('wgt_' in var) and ('nominal' not in var): continue
        # for JES/JER systematic variations do not consider weight variations
        if (v!='nominal') and ('wgt_' in var) and ('nominal' not in var): continue
        # Theory uncertainties only for VBF samples
        if s in grouping.keys():
            if (grouping[s]!='VBF') and ('THU' in var): continue
        else:
            if ('THU' in var): continue
        
        done = False
        decorrelate = ['LHEFac', 'LHERen']
        for d in decorrelate:
            # don't include LHE variations for VBF and VV
            if s not in grouping.keys(): continue
            if 'data' in s: continue
            if (grouping[s]=='VBF') or (grouping[s]=='VV'): continue
            if d in var:
                if 'off' in var: continue
                suff = ''
                if '_up' in var: suff = '_up'
                elif '_down' in var: suff = '_down'
                else: continue
                vname = var.replace(suff,'')
                for g in groups:
                    try:
                        df[f'{vname}_{g}{suff}']=proc_out[f'{var}_{c}_{r}'].value if grouping[s]==g\
                            else proc_out[f'wgt_nominal_{c}_{r}'].value
                    except:
                        df[f'{vname}_{g}{suff}'] = proc_out[f'wgt_nominal_{c}_{r}'].value
                    done = True
        
        if not done:
            try:
                if len(proc_out[f'{var}_{c}_{r}'].value)>0:
                    df[var] = proc_out[f'{var}_{c}_{r}'].value
                else:
                    df[var] = proc_out[f'wgt_nominal_{c}_{r}'].value if 'wgt_' in var else np.zeros(len_, dtype=float)
            except:
                df[var] = proc_out[f'wgt_nominal_{c}_{r}'].value if 'wgt_' in var else np.zeros(len_, dtype=float)
            
    df['c'] = c
    df['r'] = r
    df['s'] = s
    df['v'] = v
    if args['training']:
        if s in args['classes_dict'].keys():
            df['cls'] = args['classes_dict'][s]
        else:
            df['cls'] = ''

    if ('extra_events' in args.keys()) and ('plot_extra' in args.keys()):
        if ('data' in s) and (args['plot_extra']):
            df = df[df['event'].isin(args['extra_events'])]
    return df

def dnn_training(df, args):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
    from tensorflow.keras import backend as K

    def scale_data(inputs, label):
        x_mean = np.mean(x_train[inputs].values,axis=0)
        x_std = np.std(x_train[inputs].values,axis=0)
        training_data = (x_train[inputs]-x_mean)/x_std
        validation_data = (x_val[inputs]-x_mean)/x_std
        np.save(f"output/trained_models/scalers_{label}", [x_mean, x_std])
        return training_data, validation_data

    nfolds = 4
    classes = df.cls.unique()
    cls_idx_map = {cls:idx for idx,cls in enumerate(classes)}
    df['cls_idx'] = df['cls'].map(cls_idx_map)

    for i in range(nfolds):    
        label = f"{args['year']}_{args['label']}_{i}"
        
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Train classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        
        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        
        other_columns = ['event', 'wgt_nominal']
        
        df_train = df[train_filter]
        df_val = df[val_filter]
        
        x_train = df_train[training_features]
        y_train = df_train['cls_idx']
        x_val = df_val[training_features]
        y_val = df_val['cls_idx']

        for icls, cls in enumerate(classes):
            train_evts = len(y_train[y_train==icls])
            print(f"{train_evts} training events in class {cls}")

        # scale data
        x_train, x_val = scale_data(training_features, label)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]

        # load model
        input_dim = len(training_features)
        inputs = Input(shape=(input_dim,), name = label+'_input')
        x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        model.summary()

        history = model.fit(x_train[training_features], y_train, epochs=100, batch_size=2048, verbose=1,
                                            validation_data=(x_val[training_features], y_val), shuffle=True)

        model.save(f"output/trained_models/test_{label}.h5")        

def dnn_evaluation(df, args):
    if df.shape[0]==0: return df
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
    df['dnn_score'] = 0
    with sess:
        nfolds = 4
        for i in range(nfolds):    
            label = f"{args['year']}_{args['label']}_{i}"
        
            train_folds = [(i+f)%nfolds for f in [0,1]]
            val_folds = [(i+f)%nfolds for f in [2]]
            eval_folds = [(i+f)%nfolds for f in [3]]

            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            scalers_path = f'output/trained_models/scalers_{label}.npy'
            scalers = np.load(scalers_path)
            model_path = f'output/trained_models/test_{label}.h5'
            dnn_model = load_model(model_path)
            df_i = df[eval_filter]
            if args['r']!='h-peak':
                df_i['dimuon_mass'] = 125.
            df_i = (df_i[training_features]-scalers[0])/scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter,'dnn_score'] = np.arctanh((prediction))
    return df

def dnn_rebin(dfs, args):
    df = pd.concat(dfs)
    df = df[(df.s=='vbf_powhegPS')&(df.r=='h-peak')]
    cols = ['c','r','v','s', 'wgt_nominal','dnn_score']
    df = df[cols]
    df = df.sort_values(by=['dnn_score'], ascending=False)
    bnd = {}
    for c in df.c.unique():
        bnd[c] = {}
        for v in df.v.unique():
            bin_sum = 0
            boundaries = []
            counter = 0
            max_dnn = 2
            for idx, row in df[(df.c==c)&(df.v==v)].iterrows():
                if counter==0: max_dnn = row['dnn_score']
                counter+=1
                bin_sum += row['wgt_nominal']
                if bin_sum>=0.6:
                    boundaries.append(round(row['dnn_score'],3))
                    bin_sum = 0
            bnd[c][v] = sorted([0,round(max_dnn,3)]+boundaries)
    return bnd
                        

def get_hists(df, var, args, bins=[]):
    dataset_axis = bh.axis.StrCategory(df.s.unique())
    region_axis = bh.axis.StrCategory(df.r.unique())
    channel_axis = bh.axis.StrCategory(df.c.unique())
    syst_axis = bh.axis.StrCategory(df.v.unique())
    val_err_axis = bh.axis.StrCategory(['value', 'sumw2'])
    if var.name=='dnn_score' and len(bins)>0:
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
        if 'data' in s:
            syst_variations = ['nominal']
            wgts = ['wgt_nominal']
        else:
            syst_variations = args['syst_variations']
            wgts = [c for c in df.columns if ('wgt_' in c)]
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
    return df_out, edges

def save_yields(var, hist, edges, args):
    metadata = hist[var.name][['s','r','c','v','w', 'integral']]
    yields = metadata.groupby(['s','r','c','v','w']).aggregate(np.sum)
    yields.to_pickle(f"yields/yields_{args['year']}_{args['label']}.pkl")

def load_yields(s,r,c,v,w,args):
    yields = pd.read_pickle(f"yields/yields_{args['year']}_{args['label']}.pkl")
    filter0 = yields.index.get_level_values('s') == s
    filter1 = yields.index.get_level_values('r') == r
    filter2 = yields.index.get_level_values('c') == c
    filter3 = yields.index.get_level_values('v') == v
    filter4 = yields.index.get_level_values('w') == w
    ret = yields.iloc[filter0&filter1&filter2&filter3&filter4].values[0][0]
    return ret

def save_shapes(var, hist, edges, args):   
    edges = np.array(edges)
    tdir = f'/depot/cms/hmm/templates/{args["year"]}_{args["label"]}_{var.name}/'
    try:
        os.mkdir(tdir)
    except:
        pass
    
    r_names = {'h-peak':'SR','h-sidebands':'SB'}
    def get_vwname(v,w):
        vwname = ''
        if 'nominal' in v:
            if 'off' in w: return ''
            elif 'nominal' in w:
                vwname = 'nominal'
            elif '_up' in w:
                vwname = w.replace('_up', 'Up').replace('wgt_', '')
            elif '_down' in w:
                vwname = w.replace('_down', 'Down').replace('wgt_', '')
        else:
            if 'nominal' not in w: return ''
            elif '_up' in v:
                vwname = v.replace('_up', 'Up')
            elif '_down' in v:
                vwname = v.replace('_down', 'Down')
            if 'jer' not in vwname: 
                vwname = 'jes'+vwname
        return vwname  
    
    try:
        os.mkdir(f'combine_new/{args["year"]}_{args["label"]}')
    except:
        pass
    
    to_variate = ['vbf_amcPS', 'vbf_powhegPS', 'vbf_powheg_herwig','vbf_powheg_dipole',\
                  'ewk_lljj_mll105_160_ptj0','ewk_lljj_mll105_160','ewk_lljj_mll105_160_py']
    sample_variations = {
        'SignalPartonShower': {'VBF': ['vbf_powhegPS','vbf_powheg_herwig']}, 
        'EWKPartonShower': {'EWK': ['ewk_lljj_mll105_160','ewk_lljj_mll105_160_py']}, 
    }
    smp_var_alpha ={
        'SignalPartonShower': 1., 
        'EWKPartonShower': 0.2,
    }
    decorrelate = ['LHEFac', 'LHERen']
    variated_shapes = {}
    
    hist = hist[var.name]
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    data_names = [n for n in hist.s.unique() if 'data' in n]
    for cgroup,cc in args['channel_groups'].items():
        for r in args['regions']:
            data_obs_hist = np.zeros(len(bin_columns), dtype=float)
            data_obs_sumw2 = np.zeros(len(sumw2_columns), dtype=float)
            for v in hist.v.unique():
                for w in hist.w.unique():
                    vwname = get_vwname(v,w)
                    if vwname == '': continue
                    if vwname == 'nominal':
                        data_obs = hist[hist.s.isin(data_names)&(hist.r==r)&(hist.c.isin(cc))]                        
                        data_obs_hist = data_obs[bin_columns].sum(axis=0).values
                        data_obs_sumw2 = data_obs[sumw2_columns].sum(axis=0).values
                    for c in cc:
                        nom_hist = hist[~hist.s.isin(data_names)&(hist.v=='nominal')&(hist.w=='wgt_nominal')&\
                                                   (hist.r==r)&(hist.c==c)]
                        mc_hist = hist[~hist.s.isin(data_names)&(hist.v==v)&(hist.w==w)&\
                                                   (hist.r==r)&(hist.c==c)]

                        mc_by_sample = mc_hist.groupby('s').aggregate(np.sum).reset_index()
                        for s in mc_by_sample.s.unique():
                            if vwname!='nominal': continue
                            if s in to_variate:
                                variated_shapes[s] = np.array(mc_hist[mc_hist.s==s][bin_columns].values[0], dtype=float)
                        variations_by_group = {}
                        for smp_var_name, smp_var_items in sample_variations.items():
                            for gr, samples in smp_var_items.items():
                                if len(samples)!=2: continue
                                if samples[0] not in variated_shapes.keys(): continue
                                if samples[1] not in variated_shapes.keys(): continue
                                ratio = np.divide(variated_shapes[samples[0]],variated_shapes[samples[1]])
                                variation_up = np.power(ratio, smp_var_alpha[smp_var_name])
                                variation_down = np.power(ratio, -smp_var_alpha[smp_var_name])
                                variations_by_group[gr] = {}
                                variations_by_group[gr][smp_var_name] = [variation_up, variation_down]

                        mc_hist = mc_hist.groupby('g').aggregate(np.sum).reset_index() 
                        for g in mc_hist.g.unique():
                            if g not in grouping.values():continue
                            decor_ok = True
                            for d in decorrelate:
                                if (d in vwname) and (g not in vwname): decor_ok = False
                            if not decor_ok: continue
                            histo = np.array(mc_hist[mc_hist.g==g][bin_columns].values[0], dtype=float)
                            if len(histo)==0: continue
                            sumw2 = np.array(mc_hist[mc_hist.g==g][sumw2_columns].values[0], dtype=float)
                            
                            if ('LHE' in w) or ('qgl' in w):
                                histo_nom = np.array(nom_hist[nom_hist.g==g][bin_columns].values[0], dtype=float)
                                normalization = histo_nom.sum()/histo.sum()
                                histo = histo*normalization
                                sumw2 = sumw2*normalization
                            
                            histo[np.isinf(histo)] = 0
                            sumw2[np.isinf(sumw2)] = 0
                            histo[np.isnan(histo)] = 0
                            sumw2[np.isnan(sumw2)] = 0

                            if vwname=='nominal':
                                name = f'{r_names[r]}_{args["year"]}_{g}_{c}'
                            else:
                                name = f'{r_names[r]}_{args["year"]}_{g}_{c}_{vwname}'

                            try:
                                os.mkdir(f'{tdir}/{g}')
                            except:
                                pass
                            np.save(f'{tdir}/{g}/{name}', [histo,sumw2])
                            for groupname, var_items in variations_by_group.items():
                                if (groupname==g)&(vwname=='nominal'):
                                    for variname,variations in var_items.items():
                                        for iud, ud in enumerate(['Up','Down']):
                                            if len(variations[iud])==0: variations[iud]=np.ones(len(histo))
                                            histo_ud = histo*variations[iud]
                                            sumw2_ud = np.array([0]+list(sumw2[1:]*variations[iud]))
                                            name = f'{r_names[r]}_{args["year"]}_{g}_{c}_{variname}{ud}'
                                            np.save(f'{tdir}/{g}/{name}', [histo_ud,sumw2_ud])

            try:
                os.mkdir(f'{tdir}/Data/')
            except:
                pass  
            name = f'{r_names[r]}_{args["year"]}_data_obs'
            np.save(f'{tdir}/Data/{name}', [data_obs_hist,data_obs_sumw2])

def prepare_root_files(var, edges, args):
    edges = np.array(edges)
    tdir = f'/depot/cms/hmm/templates/{args["year"]}_{args["label"]}_{var.name}/'
    regions = ['SB','SR']
    centers = (edges[:-1] + edges[1:]) / 2.0
    try:
        os.mkdir(f'combine_new/{args["year"]}_{args["label"]}')
    except:
        pass
    for cgroup,cc in args['channel_groups'].items():
        for r in regions:
            out_fn = f'combine_new/{args["year"]}_{args["label"]}/shapes_{cgroup}_{r}.root'
            out_file = uproot.recreate(out_fn)
            data_hist, data_sumw2 = np.load(f'{tdir}/Data/{r}_{args["year"]}_data_obs.npy',allow_pickle=True)
            th1_data = from_numpy([data_hist, edges])
            th1_data._fName = 'data_obs'
            th1_data._fSumw2 = np.array(data_sumw2)
            th1_data._fTsumw2 = np.array(data_sumw2).sum()
            th1_data._fTsumwx2 = np.array(data_sumw2[1:] * centers).sum()
            out_file['data_obs'] = th1_data
            mc_templates = glob.glob(f'{tdir}/*/{r}_*.npy')
            for path in mc_templates:
                if 'Data' in path: continue
                hist, sumw2 = np.load(path, allow_pickle=True)
                name = os.path.basename(path).replace('.npy','')
                th1 = from_numpy([hist, edges])
                th1._fName = name.replace(f'{r}_{args["year"]}_','')
                th1._fSumw2 = np.array(sumw2)
                th1._fTsumw2 = np.array(sumw2).sum()
                th1._fTsumwx2 = np.array(sumw2[1:] * centers).sum()
                out_file[name.replace(f'{r}_{args["year"]}_','')] = th1
            out_file.close()

rate_syst_lookup = {
    '2016':{
        'XsecAndNorm2016DY_vbf_2j':1.12189,
        'XsecAndNorm2016DY_nofilter_vbf_2j':1.12189,
        'XsecAndNorm2016DY_filter_vbf_2j':1.12189,
        'XsecAndNorm2016EWK_vbf':1.06217,
        'XsecAndNorm2016TT+ST_vbf':1.18261,
        'XsecAndNorm2016VV_vbf':1.0609,
        'XsecAndNorm2016ggH_vbf': 1.36133,
        },
    '2017':{
        'XsecAndNorm2017DY_vbf_2j':1.12452,
        'XsecAndNorm2017DY_nofilter_vbf_2j':1.12452,
        'XsecAndNorm2017DY_filter_vbf_2j':1.12452,
        'XsecAndNorm2017EWK_vbf': 1.05513,
        'XsecAndNorm2017TT+ST_vbf': 1.18402,
        'XsecAndNorm2017VV_vbf':1.05734,
        'XsecAndNorm2017ggH_vbf':1.36667,
        },
    '2018':{
        'XsecAndNorm2018DY_vbf_2j': 1.12152,
        'XsecAndNorm2018DY_nofilter_vbf_2j': 1.12152,
        'XsecAndNorm2018DY_filter_vbf_2j': 1.12152,
        'XsecAndNorm2018EWK_vbf':1.05851,
        'XsecAndNorm2018TT+ST_vbf':1.18592,
        'XsecAndNorm2018VV_vbf':1.05734,
        'XsecAndNorm2018ggH_vbf': 1.39295,
        },
}            

def get_numbers(var, cc, r, bin_name, args):
    groups = {'Data': ['vbf'],
              'DY_nofilter':['vbf_01j','vbf_2j'],
              'DY_filter':['vbf_01j','vbf_2j'],
              'EWK':['vbf'],
              'TT+ST':['vbf'],
              'VV':['vbf'],
              'ggH':['vbf'],
              'VBF':['vbf']
             }
    regions = ['SB', 'SR']
    year = args['year']
    floating_norm = {'DY':['vbf_01j']}
    sig_groups = ['ggH', 'VBF']
    sample_variations = {
        'SignalPartonShower': {'VBF': ['vbf_powhegPS','vbf_powheg_herwig']}, 
        'EWKPartonShower': {'EWK': ['ewk_lljj_mll105_160','ewk_lljj_mll105_160_py']}, 
    }

    sig_counter = 0
    bkg_counter = 0
    tdir = f'/depot/cms/hmm/templates/{args["year"]}_{args["label"]}_{var.name}/'
    shape_systs_by_group = {}
    for g, cc in groups.items():
        shape_systs_by_group[g] = [os.path.basename(path).replace('.npy','') for path in glob.glob(f'{tdir}/{g}/{r}_*.npy')]
        for c in cc:
            shape_systs_by_group[g] = [path for path in shape_systs_by_group[g] if (path!=f'{r}_{args["year"]}_{g}_{c}') and ('nominal' not in path)]
            shape_systs_by_group[g] = [path.replace(f'{r}_{args["year"]}_{g}_{c}_','').replace('Up','').replace('Down','') for path in shape_systs_by_group[g]]
        shape_systs_by_group[g] = np.unique(shape_systs_by_group[g])

    shape_systs = []
    for shs in shape_systs_by_group.values():
        shape_systs.extend(shs)
    shape_systs = np.unique(shape_systs)
    systs = []
    systs.extend(shape_systs)

    data_yields = pd.DataFrame()
    data_yields['index'] = ['bin','observation']
    
    mc_yields = pd.DataFrame()
    mc_yields['index'] = ['bin','process','process','rate']
    
    systematics = pd.DataFrame(index=systs)

    for g,cc in groups.items():
        if g not in grouping.values(): continue
        for c in cc:
            gcname = f'{g}_{c}'
            counter = 0
            if g in sig_groups:
                sig_counter += 1
                counter = -sig_counter
            elif 'Data'not in g:
                bkg_counter += 1
                counter = bkg_counter
            if g=='Data':
                data = f'{tdir}/Data/{r}_{args["year"]}_data_obs.npy'
                hist,_ = np.load(data, allow_pickle=True)
                rate = hist.sum()          
                data_yields.loc[0,'value'] = bin_name
                data_yields.loc[1,'value'] = f'{rate}'
            else:
                nominal_shape = f'{tdir}/{g}/{r}_{args["year"]}_{g}_{c}.npy'
                hist,_ = np.load(nominal_shape, allow_pickle=True)
                rate = hist.sum()          
                mc_yields.loc[0,gcname] = bin_name
                mc_yields.loc[1,gcname] = gcname
                mc_yields.loc[2,gcname] = f'{counter}'
                mc_yields.loc[3,gcname] = f'{rate}'
                for syst in shape_systs:
                    systematics.loc[syst,'type'] = 'shape'
                    if sum([gname in syst for gname in groups.keys()]):
                        systematics.loc[syst,gcname] = '1.0' if (g in syst) and (syst in shape_systs_by_group[g]) else '-'   
                    else:
                        systematics.loc[syst,gcname] = '1.0' if syst in shape_systs_by_group[g] else '-'
                for syst in rate_syst_lookup[year].keys():
                    systematics.loc[syst,'type'] = 'lnN'
                    if gcname in syst:
                        val = rate_syst_lookup[year][syst]
                    else: val = '-'
                    systematics.loc[syst,gcname] = f'{val}'
    def to_string(df):
        string = ''
        for row in df.values:
            for i,item in enumerate(row):
                ncols = 2 if item in ['bin','process','rate', 'observation'] else 1
                row[i] = item+' '*(ncols*20-len(item))
            string += ' '.join(row)
            string += '\n'
        return string
    print(mc_yields) 
    print(systematics)
    return to_string(data_yields), to_string(mc_yields), to_string(systematics.reset_index())
            
def make_datacards(var, args):
    year = args["year"]
    for cgroup, cc in args['channel_groups'].items():
        for r in ['SB', 'SR']:
            bin_name = f'{r}_{year}'
            datacard_name = f'combine_new/{year}_{args["label"]}/datacard_{cgroup}_{r}.txt'
            shapes_file = f'shapes_{cgroup}_{r}.root'
            datacard = open(datacard_name, 'w')
            datacard.write(f"imax 1\n")
            datacard.write(f"jmax *\n")
            datacard.write(f"kmax *\n")
            datacard.write("---------------\n")
            datacard.write(f"shapes * {bin_name} {shapes_file} $PROCESS $PROCESS_$SYSTEMATIC\n")
            datacard.write("---------------\n")
            data_yields, mc_yields, systematics = get_numbers(var, cc, r, bin_name, args)
            datacard.write(data_yields)
            datacard.write("---------------\n")
            datacard.write(mc_yields)
            datacard.write("---------------\n")
            datacard.write(systematics)
           # datacard.write(f"XSecAndNormDY_01j  rateParam {bin_name} DY_vbf_01j 1 [0.2,5] \n")
            datacard.write(f"XSecAndNorm{year}DY_01j  rateParam {bin_name} DY_nofilter_vbf_01j 1 [0.2,5] \n")
            datacard.write(f"XSecAndNorm{year}DY_01j  rateParam {bin_name} DY_filter_vbf_01j 1 [0.2,5] \n")
            datacard.write(f"{bin_name} autoMCStats 0 1 1\n")
            datacard.close()
            print(f'Saved datacard to {datacard_name}')
    return

def add_source(hist, group_name):
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    vals = hist[hist['g']==group_name]
    vals = vals.groupby('g').aggregate(np.sum).reset_index()
    sumw2 = vals[sumw2_columns].sum(axis=0).reset_index(drop=True).values 
    try:
        vals = vals[bin_columns].values[0] 
        return vals, sumw2
    except:
#        print(group_name, "missing")
        return np.array([]), np.array([])

def plot(var, hists, edges, args, r='', save=True, blind=True, show=False, plotsize=12, compare_with_pisa=False):    
    hist = hists[var.name]
    edges_data = edges
    blind_bins = 5
    if r=='h-sidebands':
        blind = False

    if r!='':
        hist = hist[hist.r==r]
    year = args['year']
    label = args['label']
#    bkg_groups = ['DY','DY_VBF', 'EWK', 'TT+ST', 'VV']
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    
    def get_shapes_for_option(hist_,v,w):
        bkg_groups = ['DY', 'DY_nofilter','DY_filter', 'EWK', 'TT+ST', 'VV']
        hist_nominal = hist_[(hist_.w=='wgt_nominal')&(hist_.v=='nominal')]
        hist = hist_[(hist_.w==w)&(hist_.v==v)]
        
        vbf, vbf_sumw2 = add_source(hist, 'VBF')
        ggh, ggh_sumw2 = add_source(hist, 'ggH')
        data, data_sumw2 = add_source(hist_nominal, 'Data')    

        bin_columns = [c for c in hist.columns if 'bin' in c]
        sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
        bkg_df = hist[hist['g'].isin(bkg_groups)]
        bkg_df.loc[:,'bkg_integral'] = bkg_df[bin_columns].sum(axis=1)
        bkg_df = bkg_df.groupby('g').aggregate(np.sum).reset_index()
        bkg_df = bkg_df.sort_values(by='bkg_integral').reset_index(drop=True)
        bkg_total = bkg_df[bin_columns].sum(axis=0).reset_index(drop=True)    
        bkg_sumw2 = bkg_df[sumw2_columns].sum(axis=0).reset_index(drop=True)

        return {'data':       data, 
                'data_sumw2': data_sumw2,
                'vbf':        vbf,
                'vbf_sumw2':  vbf_sumw2,
                'ggh':        ggh, 
                'ggh_sumw2':  ggh_sumw2,
                'bkg_df':     bkg_df,
                'bkg_total':  bkg_total, 
                'bkg_sumw2':  bkg_sumw2 }
    
    ret_nominal = get_shapes_for_option(hist,'nominal','wgt_nominal')
    data       = ret_nominal['data']
    data_sumw2 = ret_nominal['data_sumw2'][1:]

    if blind and var.name=='dnn_score':
        data = data[:-blind_bins]
        data_sumw2 = data_sumw2[:-blind_bins]
        edges_data = edges_data[:-blind_bins]

    vbf        = ret_nominal['vbf']
    vbf_sumw2  = ret_nominal['vbf_sumw2'][1:]
    ggh        = ret_nominal['ggh']
    ggh_sumw2  = ret_nominal['ggh_sumw2'][1:]
    bkg_df     = ret_nominal['bkg_df']
    bkg_total  = ret_nominal['bkg_total']
    bkg_sumw2  = ret_nominal['bkg_sumw2'][1:].values
    
    if len(bkg_df.values)>1:
        bkg = np.stack(bkg_df[bin_columns].values)
        stack=True
    else:
        bkg = bkg_df[bin_columns].values
        stack=False
    bkg_labels = bkg_df.g
    
    # Report yields
    if not show and var.name=='dimuon_mass':
        print("="*50)
        if r=='':
            print(f"{var.name}: Inclusive yields:")
        else:
            print(f"{var.name}: Yields in {r}")
        print("="*50)
        print('Data', data.sum())
        for row in bkg_df[['g','integral']].values:
            print(row)
        print('VBF', vbf.sum())
        print('ggH', ggh.sum())
        print("-"*50)
    
    # Make plot
    fig = plt.figure()
    plt.rcParams.update({'font.size': 22})
    ratio_plot_size = 0.25
    data_opts = {'color': 'k', 'marker': '.', 'markersize':15}
    data_opts_pisa = {'color': 'red', 'marker': '.', 'markersize':15}
    stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0)}
    stat_err_opts = {'step': 'post', 'label': 'Stat. unc.', 'hatch': '//////',\
                        'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0}
    ratio_err_opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3), 'linewidth': 0}
    
    fig.clf()    
    fig.set_size_inches(plotsize, plotsize*(1+ratio_plot_size))
    gs = fig.add_gridspec(2, 1, height_ratios=[(1-ratio_plot_size),ratio_plot_size], hspace = .05)
    
    # Top panel: Data/MC
    plt1 = fig.add_subplot(gs[0])

    if bkg_total.sum():
        ax_bkg = hep.histplot(bkg, edges, ax=plt1, label=bkg_labels, stack=stack, histtype='fill', **stack_fill_opts)
        err = coffea.hist.plot.poisson_interval(np.array(bkg_total), bkg_sumw2)
        ax_bkg.fill_between(x=edges, y1=np.r_[err[0, :], err[0, -1]], y2=np.r_[err[1, :], err[1, -1]], **stat_err_opts)
            
    if compare_with_pisa:
        r_opt = 'inclusive' if r=='' else r
        pisa_hist, pisa_data_hist = get_pisa_hist(var, r_opt, edges)
        if pisa_hist.sum():
            ax_pisa = hep.histplot(pisa_hist, edges, label='Pisa', histtype='step', **{'linewidth':3, 'color':'red'})
        if pisa_data_hist.sum():
            ax_pisa_data = hep.histplot(pisa_data_hist, edges, label='Pisa Data', histtype='errorbar', yerr=np.sqrt(pisa_data_hist), **data_opts_pisa)
        
    if ggh.sum():
        ax_ggh = hep.histplot(ggh, edges, label='ggH', histtype='step', **{'linewidth':3, 'color':'lime'})
    if vbf.sum():
        ax_vbf = hep.histplot(vbf, edges, label='VBF', histtype='step', **{'linewidth':3, 'color':'aqua'})
    if data.sum() and (not blind or var.name!='dimuon_mass'):
        ax_data = hep.histplot(data, edges_data, label='Data', histtype='errorbar', yerr=np.sqrt(data), **data_opts)
    
    max_variation_up = bkg_total.sum()
    max_variation_down = bkg_total.sum()
    max_var_up_name = ''
    max_var_down_name = ''
    for v in hist.v.unique():
        for w in hist.w.unique():
            continue
            if ('nominal' in v) and ('nominal' in w): continue
            if ('off' in w): continue
            if ('wgt' not in w): continue
            ret = get_shapes_for_option(hist,v,w)
            if ret['bkg_total'].sum():
                #ax_vbf = hep.histplot(ret['bkg_total'].values, edges, histtype='step', **{'linewidth':3})
                if (ret['bkg_total'].values.sum() > max_variation_up):
                    max_variation_up = ret['bkg_total'].values.sum()
                    max_var_up_name = f'{v},{w}'
                if (ret['bkg_total'].values.sum() < max_variation_down):
                    max_variation_down = ret['bkg_total'].values.sum()
                    max_var_down_name = f'{v},{w}'
                    
            if ret['ggh'].sum():
                ax_vbf = hep.histplot(ret['ggh'], edges, histtype='step', **{'linewidth':3})
            if ret['vbf'].sum():
                ax_vbf = hep.histplot(ret['vbf'], edges, histtype='step', **{'linewidth':3})
    
    #print(f"Max. variation up: {max_var_up_name}")
    #print(f"Max. variation down: {max_var_down_name}")
    
    lbl = hep.cms.cmslabel(ax=plt1, data=True, paper=False, year=year)
    
    plt1.set_yscale('log')
    plt1.set_ylim(0.01, 1e9)
#    plt1.set_xlim(var.xmin,var.xmax)
    plt1.set_xlim(edges[0],edges[-1])
    plt1.set_xlabel('')
    plt1.tick_params(axis='x', labelbottom=False)
    plt1.legend(prop={'size': 'small'})
    
    # Bottom panel: Data/MC ratio plot
    plt2 = fig.add_subplot(gs[1], sharex=plt1)
    
    if (data.sum()*bkg_total.sum()) and (not blind or var.name!='dimuon_mass'):
        ratios = np.zeros(len(data))
        yerr = np.zeros(len(data))
        unity = np.ones_like(bkg_total)
        zero = np.zeros_like(bkg_total)
        bkg_total[bkg_total==0] = 1e-20
        ggh[ggh==0] = 1e-20
        vbf[vbf==0] = 1e-20
        bkg_unc = coffea.hist.plot.poisson_interval(unity, bkg_sumw2 / bkg_total**2)
        denom_unc = bkg_unc

        if blind and var.name=='dnn_score':
            bkg_total = bkg_total[:-blind_bins]

        ratios[bkg_total!=0] = np.array(data[bkg_total!=0] / bkg_total[bkg_total!=0])
        yerr[bkg_total!=0] = np.sqrt(data[bkg_total!=0])/bkg_total[bkg_total!=0]
        edges_ratio = edges_data if blind else edges
        ax_ratio = hep.histplot(ratios, edges_ratio, histtype='errorbar', yerr=yerr,**data_opts)
        ax_ratio.fill_between(edges,np.r_[denom_unc[0],denom_unc[0, -1]],np.r_[denom_unc[1], denom_unc[1, -1]], **ratio_err_opts)

    for v in hist.v.unique():
        for w in hist.w.unique():
            if ('nominal' not in v) and ('nominal' not in w): continue
            if ('nominal' in v) and ('nominal' in w): continue
            if ('off' in w): continue
            continue
#            if ('Flavor' not in v): continue
            ret = get_shapes_for_option(hist,v,w)
            syst_ratio = np.zeros(len(bkg_total))
            syst_ratio[bkg_total!=0] = np.array(ret['bkg_total'].values[bkg_total!=0] / bkg_total[bkg_total!=0])
            ax = hep.histplot(syst_ratio, edges, histtype='step', label=f'{v},{w}', **{'linewidth':3})
            plt2.legend(prop={'size': 'xx-small'})

    plt2.axhline(1, ls='--')
    plt2.set_ylim([0.5,1.5])
    plt2.set_ylabel('Data/MC')
    lbl = plt2.get_xlabel()
    plt2.set_xlabel(f'{var.caption}')

    if compare_with_pisa and pisa_hist.sum():
        ratio = np.zeros(len(bkg_total))
        ratio[bkg_total!=0] = np.array(pisa_hist[bkg_total!=0] / bkg_total[bkg_total!=0])
        ax = hep.histplot(ratio, edges, label='Pisa/Purdue MC', histtype='step', **{'linewidth':3, 'color':'red'})
        plt2.legend(prop={'size': 'small'})
        plt2.set_ylim([0.8,1.2])

    if compare_with_pisa and pisa_data_hist.sum():
        ratio_data = np.zeros(len(bkg_total))
        ratio_data[data!=0] = np.array(pisa_data_hist[data!=0] / data[data!=0])
        ax = hep.histplot(ratio_data, edges, label='Pisa/Purdue Data', histtype='step', **{'linewidth':3, 'color':'blue'})
        plt2.legend(prop={'size': 'small'})
        plt2.set_ylim([0.8,1.2])


    if save:
        # Save plots
        out_path = args['out_path']
        full_out_path = f"{out_path}/plots_{year}_{label}"
        if 'plot_extra' in args.keys():
            if args['plot_extra']:
                full_out_path += '_extra'
        try:
            os.mkdir(out_path)
        except:
            pass
        try:
            os.mkdir(full_out_path)
        except:
            pass
        if r=='':
            out_name = f"{full_out_path}/{var.name}_inclusive.png"
        else:
            out_name = f"{full_out_path}/{var.name}_{r}.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)

    if show:
        plt.show()
        

var_map_pisa = {
'mu1_pt':'Mu0_pt',
'mu1_eta':'Mu0_eta',
'mu2_pt':'Mu1_pt',
'mu2_eta':'Mu1_eta',
'dimuon_pt':'Higgs_pt', 
'dimuon_eta':'Higgs_eta',
'dimuon_mass':'Higgs_m',
'jet1_pt':'QJet0_pt_touse',
'jet1_phi':'QJet0_phi',
'jet1_eta':'QJet0_eta',    
'jet2_pt':'QJet1_pt_touse',
'jet2_phi':'QJet1_phi',
'jet2_eta':'QJet1_eta',    

}

def get_pisa_hist(var, r, edges):
    import uproot
    filenames = {
        'data': '/depot/cms/hmm/coffea/pisa-jun12/data2018Snapshot.root',
        'dy_m105_160_amc':'/depot/cms/hmm/coffea/pisa/DY105_2018AMCPYSnapshot.root',
        'dy_m105_160_vbf_amc': '/depot/cms/hmm/coffea/pisa/DY105VBF_2018AMCPYSnapshot.root'
    }
    xsec = {
        'data':1,
        'dy_m105_160_amc': 47.17, #
        'dy_m105_160_vbf_amc': 2.03 #
    }
    N = {
        'data': 1,
        'dy_m105_160_amc': 6995355211.029654,
        'dy_m105_160_vbf_amc': 3146552884.4507833
    }
    qgl_mean = {
        'data': 1,
        'dy_m105_160_amc':  1.04859375342,
        'dy_m105_160_vbf_amc' :1.00809412662
    }


    lumi = 59970.
#    samples = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc']
    samples = ['data']
    total_hist = np.array([])
    data_hist = np.array([])
    for s in samples:
        with uproot.open(filenames[s]) as f:
            tree = f['Events']
            wgt = np.ones(len(tree['event'].array()), dtype=float)
            weights = ['genWeight', 'puWeight', 'btagEventWeight','muEffWeight', 'EWKreweight',\
                       'PrefiringWeight', 'QGLweight']
            if 'data' not in s:
                for i,w in enumerate(weights):
                    wgt = wgt*tree[w].array()
                wgt = wgt*xsec[s]*lumi / N[s] / qgl_mean[s]
            hist = bh.Histogram(bh.axis.Variable(edges))
            var_arr = tree[var_map_pisa[var.name]].array()
            hist.fill(var_arr, weight=wgt)
            if 'data' in s:
                data_hist = hist.to_numpy()[0]
                continue
            if len(total_hist)>0:
                total_hist += hist.to_numpy()[0]
            else:
                total_hist = hist.to_numpy()[0]

    print(f'Pisa data yield ({var.name}):', data_hist.sum())
    return total_hist, data_hist


