import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import coffea
from coffea import util
import glob
from config.parameters import training_features
from config.datasets import grouping
import boost_histogram as bh
from boost_histogram import loc
import numpy as np
import uproot
from uproot_methods.classes.TH1 import from_numpy
import matplotlib.pyplot as plt
import mplhep as hep

def worker(args):
    modules = args['modules']
    if 'to_pandas' not in modules:
        print("Need to convert to Pandas DF first!")
        return
    df = to_pandas(args)
    if 'dnn_evaluation' in modules:
        df = dnn_evaluation(df, args)
    hists = {}
    edges = {}
    dnn_bins = {
        '2016':[0, 0.079, 0.347, 0.578, 0.785, 0.977, 1.158, 1.329, 1.489, 1.636, 1.774, 1.924, 2.002],
        '2017':[0, 0.079, 0.347, 0.578, 0.785, 0.977, 1.158, 1.329, 1.489, 1.636, 1.774, 1.924, 2.002],
        '2018':[0, 0.079, 0.347, 0.578, 0.785, 0.977, 1.158, 1.329, 1.489, 1.636, 1.774, 1.924, 2.002]
    }
    if 'get_hists' in modules:
        for var in args['vars_to_plot']:
            hists[var.name], edges[var.name] = get_hists(df, var, args, bins=dnn_bins[args['year']])
            
    return df, hists, edges


def postprocess(args, parallelize=True):
    dataframes = []
    hist_dfs = {}
    edges_dict = {}
    path = args['in_path']
    argsets = []
    for s in args['samples']:
        variations = ['nominal'] if 'data' in s else args['syst_variations']
        for v in variations:
            proc_outs = glob.glob(f"{path}/unbinned/{s}_?.coffea") if v=='nominal' else glob.glob(f"{path}/{v}/{s}_?.coffea")
            for proc_path in proc_outs:
                for c in args['channels']:
                    for r in args['regions']:
                        argset = args.copy()
                        argset.update(**{'proc_path':proc_path,'s':s,'c':c,'r':r,'v':v})
                        argsets.append(argset)

    if parallelize:
        cpus = mp.cpu_count()-2
        print(f'Using {cpus} CPUs')
        pool = mp.Pool(cpus)
        a = [pool.apply_async(worker, args=(argset,)) for argset in argsets]
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
    decorrelate = ['LHEFac', 'LHERen']
    groups = ['DY', 'EWK', 'TT+ST', 'VV', 'ggH', 'VBF']
    columns = [c.replace(suff, '') for c in list(proc_out.keys()) if suff in c]
    df = pd.DataFrame()
    len_ = len(proc_out[f'dimuon_mass_{c}_{r}'].value) if f'dimuon_mass_{c}_{r}' in proc_out.keys() else 0
    for var in columns:
        if (not args['wgt_variations']) and ('wgt_' in var) and ('nominal' not in var): continue
#        if ('wgt_' not in var) and (var not in [v.name for v in args['vars_to_plot']]): continue
        if (v!='nominal') and ('wgt_' in var) and ('nominal' not in var): continue
        if (('vbf' not in s) or ('dy' in s)) and ('THU' in var): continue
        if 'btag' in var: continue
        
        done = False
        for d in decorrelate:
            if d in var:
                if 'off' in var: continue
                suff = ''
                if '_up' in var: suff = '_up'
                elif '_down' in var: suff = '_down'
                else: continue
                vname = var.replace(suff,'')
                for g in groups:
                    if s not in grouping.keys():continue
                    df[f'{vname}_{g}{suff}']=proc_out[f'{var}_{c}_{r}'].value if grouping[s]==g\
                                                                                else proc_out[f'wgt_nominal_{c}_{r}'].value
                    done = True
        
        if not done:
            try:
                df[var] = proc_out[f'{var}_{c}_{r}'].value
            except:
                df[var] = proc_out[f'wgt_nominal_{c}_{r}'].value if 'wgt_' in var else np.zeros(len_, dtype=float)
            
    df['c'] = c
    df['r'] = r
    df['s'] = s
    df['v'] = v
#    print(df[[k for k in df.columns if 'LHE' in k]])
    return df

def dnn_evaluation(df, args):
    dnn_label = args['year']
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
    scalers_path = f'output/trained_models/scalers_{dnn_label}.npy'
    scalers = np.load(scalers_path)
    model_path = f'output/trained_models/test_{dnn_label}_hw.h5'
    with sess:
        dnn_model = load_model(model_path)
        df_eval = df[training_features]
        if args['r']!='h-peak':
            df_eval['dimuon_mass'] = 125.
        df_eval = (df_eval[training_features]-scalers[0])/scalers[1]
        prediction = dnn_model.predict(df_eval).ravel()
        df['dnn_score'] = np.arctanh((prediction))
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
    print(bnd)
                        

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
            wgts = [c for c in df.columns if ('wgt_' in c)]# and ('_off' not in c)]
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
                        values[values<=0] = 0.000000000000001
                        sumw2[values<=0] = 0.000000000000001
                        integral = values.sum()
                        edges = hist[loc(s), loc(r), loc(c), loc(v), loc('value'), :].to_numpy()[1]
                        contents = {}
                        contents.update({f'bin{i}':[values[i]] for i in range(nbins)})
                        contents.update({f'sumw2_{i}':[sumw2[i]] for i in range(nbins)})
                        contents.update({'s':[s],'r':[r],'c':[c], 'v':[v], 'w':[w],\
                                         'var':[var.name], 'integral':integral})
                        contents['g'] = grouping[s] if s in grouping.keys() else f"_{s}"
                        row = pd.DataFrame(contents)
                        df_out = pd.concat([df_out, row], ignore_index=True)
    return df_out, edges

def save_shapes(var, hist, edges, args):   
    r_names = {'h-peak':'SR','h-sidebands':'SB'}
    def get_vwname(v,w):
        vwname = ''
        if 'nominal' in v:
            if 'off' in w: return ()
            elif 'nominal' in w:
                vwname = 'nominal'
            elif '_up' in w:
                vwname = w.replace('_up', 'Up').replace('wgt_', '')
            elif '_down' in w:
                vwname = w.replace('_down', 'Down').replace('wgt_', '')
        else:
            if 'nominal' not in w: return ()
            elif '_up' in v:
                vwname = v.replace('_up', 'Up')
            elif '_down' in v:
                vwname = v.replace('_down', 'Down')  
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

    variated_shapes = {}
    
    hist = hist[var.name]
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    data_names = [n for n in hist.s.unique() if 'data' in n]
    for cgroup,cc in args['channel_groups'].items():
        for r in args['regions']:
            out_fn = f'combine_new/{args["year"]}_{args["label"]}/shapes_{cgroup}_{r}.root'
            out_file = uproot.recreate(out_fn)
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
                            histo = np.array(mc_hist[mc_hist.g==g][bin_columns].values[0], dtype=float)
                            if len(histo)==0: continue
                            sumw2 = np.array(mc_hist[mc_hist.g==g][sumw2_columns].values[0], dtype=float)
                            histo[np.isinf(histo)] = 0
                            sumw2[np.isinf(sumw2)] = 0
                            histo[np.isnan(histo)] = 0
                            sumw2[np.isnan(sumw2)] = 0
                            name = f'{r_names[r]}_{args["year"]}_{g}_{c}_{vwname}'
                            th1 = from_numpy([histo, edges])
                            th1._fName = name
                            th1._fSumw2 = np.array(sumw2)
                            th1._fTsumw2 = np.array(sumw2).sum()
                            th1._fTsumwx2 = np.array(sumw2 * centers).sum()
                            if vwname=='nominal':
                                out_file[f'{g}_{c}'] = th1
                            else:
                                out_file[f'{g}_{c}_{vwname}'] = th1
                            for groupname, var_items in variations_by_group.items():
                                if (groupname==g)&(vwname=='nominal'):
                                    for variname,variations in var_items.items():
                                        for iud, ud in enumerate(['Up','Down']):
                                            histo_ud = histo*variations[iud]
                                            sumw2_ud = sumw2*variations[iud]
                                            name = f'{r_names[r]}_{args["year"]}_{g}_{c}_{variname}{ud}'
                                            th1 = from_numpy([histo_ud, edges])
                                            th1._fName = name
                                            th1._fSumw2 = np.array(sumw2_ud)
                                            th1._fTsumw2 = np.array(sumw2_ud).sum()
                                            th1._fTsumwx2 = np.array(sumw2_ud * centers).sum()
                                            if vwname=='nominal':
                                                out_file[f'{g}_{c}_{variname}{ud}'] = th1

            th1_data = from_numpy([data_obs_hist, edges])
            th1_data._fName = 'data_obs'
            th1_data._fSumw2 = np.array(data_obs_sumw2)
            th1_data._fTsumw2 = np.array(data_obs_sumw2).sum()
            th1_data._fTsumwx2 = np.array(data_obs_sumw2 * centers).sum()
            out_file['data_obs'] = th1_data
            out_file.close()

rate_syst_lookup = {
    '2016':{
        'XsecAndNormDY_vbf_01j':1.12189,
        'XsecAndNormEWK_vbf':1.06217,
        'XsecAndNormTT+ST_vbf':1.18261,
        'XsecAndNormVV_vbf':1.0609,
        'XsecAndNormggH_vbf': 1.36133,
        },
    '2017':{
        'XsecAndNormDY_vbf_01j':1.12452,
        'XsecAndNormEWK_vbf': 1.05513,
        'XsecAndNormTT+ST_vbf': 1.18402,
        'XsecAndNormVV_vbf':1.05734,
        'XsecAndNormggH_vbf':1.36667,
        },
    '2018':{
        'XsecAndNormDY_vbf_01j': 1.12152,
        'XsecAndNormEWK_vbf':1.05851,
        'XsecAndNormTT+ST_vbf':1.18592,
        'XsecAndNormVV_vbf':1.05734,
        'XsecAndNormggH_vbf': 1.39295,
        },
}            
            
def get_numbers(hist, bin_name, args):
    groups = {}
    for g in hist.g.unique():
        groups[g] = []
        for c in hist[hist.g==g].c.unique():
            groups[g].append(c)
    year = args['year']
    floating_norm = {'DY':['vbf_01j']}
    sig_groups = ['ggH', 'VBF']
    sample_variations = {
        'SignalPartonShower': {'VBF': ['vbf_powhegPS','vbf_powheg_herwig']}, 
        'EWKPartonShower': {'EWK': ['ewk_lljj_mll105_160','ewk_lljj_mll105_160_py']}, 
    }

    sig_counter = 0
    bkg_counter = 0
    
    systs = []
    shape_systs = [w.replace('_up','').replace('wgt_','') for w in hist.w.unique() if '_up' in w]
    shape_systs.extend(['jes'+v.replace('_up','') for v in hist.v.unique() if '_up' in v])
    shape_systs.extend(list(sample_variations.keys()))
#    lnn_systs = ['XsecAndNorm'+g+year for g in groups if 'Data' not in g]
#    lnn_systs = []
#    for g,cc in groups.items():
#        for c in cc:
#            if 'Data' in g: continue
#            gcname = f'{g}_{c}'
#            lnn_systs.append('XsecAndNorm'+gcname+year)
#    lnn_systs.append('lumi')
    systs.extend(shape_systs)
#    systs.extend(lnn_systs)

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
            hist_g = hist[(hist.g==g)&(hist.c==c)]
            systs_g = [w.replace('_up','').replace('wgt_','') for w in hist_g.w.unique() if '_up' in w]
            systs_g.extend(['jes'+v.replace('_up','') for v in hist_g.v.unique() if '_up' in v])
            for smp_var_n, smp_var in sample_variations.items():
                if g in smp_var.keys():
                    systs_g.append(smp_var_n)
            hist_g = hist_g[(hist_g.v=='nominal')&(hist_g.w=='wgt_nominal')].groupby('g').aggregate(np.sum).reset_index()
            rate = hist_g.integral.values[0]
            if g=='Data':
                data_yields.loc[0,'value'] = bin_name
                data_yields.loc[1,'value'] = f'{rate}'
            else:
                mc_yields.loc[0,gcname] = bin_name
                mc_yields.loc[1,gcname] = gcname
                mc_yields.loc[2,gcname] = f'{counter}'
                mc_yields.loc[3,gcname] = f'{rate}'
                for syst in shape_systs:
                    systematics.loc[syst,'type'] = 'shape'
                    if sum([gname in syst for gname in groups.keys()]):
                        systematics.loc[syst,gcname] = '1.0' if (g in syst) and (syst in systs_g) else '-'   
                    else:
                        systematics.loc[syst,gcname] = '1.0' if syst in systs_g else '-'
                for syst in rate_syst_lookup[year].keys():
                    systematics.loc[syst,'type'] = 'lnN'
                    if gcname in syst:
                        val = rate_syst_lookup[year][syst]
                    else: val = '-'
                    systematics.loc[syst,gcname] = f'{val}'
                        
    
    def to_string(df):
        string = ''
        for row in df.values:
#            print(row)
            for i,item in enumerate(row):
                ncols = 2 if item in ['bin','process','rate', 'observation'] else 1
                row[i] = item+' '*(ncols*20-len(item))
            string += ' '.join(row)
            string += '\n'
        return string
    print(mc_yields) 
    print(systematics)
    return to_string(data_yields), to_string(mc_yields), to_string(systematics.reset_index())
    
        
            
def make_datacards(var, hist, args):
    r_names = {'h-peak':'SR','h-sidebands':'SB'}
    hist = hist[var.name]
    for cgroup, cc in args['channel_groups'].items():
        for r in args['regions']:    
            bin_name = f'{r_names[r]}_{args["year"]}'
            datacard_name = f'combine_new/{args["year"]}_{args["label"]}/datacard_{cgroup}_{r}.txt'
            shapes_file = f'shapes_{cgroup}_{r}.root'
            datacard = open(datacard_name, 'w')
            datacard.write(f"imax 1\n") # will combine the datacards later
            datacard.write(f"jmax *\n")
            datacard.write(f"kmax *\n")
            datacard.write("---------------\n")
            datacard.write(f"shapes * {bin_name} {shapes_file} $PROCESS $PROCESS_$SYSTEMATIC\n")
            datacard.write("---------------\n")
            bin_name = f'{r_names[r]}_{args["year"]}'
            ret = data_yields, mc_yields, systematics = get_numbers(hist[(hist.c.isin(cc)) & (hist.r==r)], bin_name, args)
            datacard.write(data_yields)
            datacard.write("---------------\n")
            datacard.write(mc_yields)
            datacard.write("---------------\n")
            datacard.write(systematics)
            datacard.write(f"XSecAndNormDY_01j  rateParam {bin_name} DY_vbf_01j 1 [0.2,5] \n")
#            datacard.write(f"{bin_name} autoMCStats 0 1 1\n")
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
        
def plot(var, hists, edges, args, r='', save=True, show=False, plotsize=12, compare_with_pisa=False):    
    hist = hists[var.name]
    print(hist)
    if r!='':
        hist = hist[hist.r==r]
    year = args['year']
    label = args['label']
#    bkg_groups = ['DY','DY_VBF', 'EWK', 'TT+ST', 'VV']
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    
    def get_shapes_for_option(hist_,v,w):
        bkg_groups = ['DY','DY_VBF', 'EWK', 'TT+ST', 'VV']
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
    data_sumw2 = ret_nominal['data_sumw2']
    vbf        = ret_nominal['vbf']
    vbf_sumw2  = ret_nominal['vbf_sumw2']
    ggh        = ret_nominal['ggh']
    ggh_sumw2  = ret_nominal['ggh_sumw2']
    bkg_df     = ret_nominal['bkg_df']
    bkg_total  = ret_nominal['bkg_total']
    bkg_sumw2  = ret_nominal['bkg_sumw2']
    
    if len(bkg_df.values)>1:
        bkg = np.stack(bkg_df[bin_columns].values)
        stack=True
    else:
        bkg = bkg_df[bin_columns].values
        stack=False
    bkg_labels = bkg_df.g
    
    # Report yields
    if not show:
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
            pisa_hist = get_pisa_hist(var, r_opt, edges)
            ax_pisa = hep.histplot(pisa_hist, edges, label='Pisa', histtype='step', **{'linewidth':3, 'color':'red'})

        
    if ggh.sum():
        ax_ggh = hep.histplot(ggh, edges, label='ggH', histtype='step', **{'linewidth':3, 'color':'lime'})
    if vbf.sum():
        ax_vbf = hep.histplot(vbf, edges, label='VBF', histtype='step', **{'linewidth':3, 'color':'aqua'})
    if data.sum():
        ax_data = hep.histplot(data, edges, label='Data', histtype='errorbar', yerr=np.sqrt(data), **data_opts)
    
    max_variation_up = bkg_total.sum()
    max_variation_down = bkg_total.sum()
    max_var_up_name = ''
    max_var_down_name = ''
    for v in hist.v.unique():
        for w in hist.w.unique():
            if ('nominal' in v) and ('nominal' in w): continue
            if ('off' in w): continue
            if ('wgt' not in w): continue
            ret = get_shapes_for_option(hist,v,w)
            if ret['bkg_total'].sum():
                ax_vbf = hep.histplot(ret['bkg_total'].values, edges, histtype='step', **{'linewidth':3})
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
    
    print(f"Max. variation up: {max_var_up_name}")
    print(f"Max. variation down: {max_var_down_name}")
    
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
    
    if (data.sum()*bkg_total.sum()):
        ratios = np.zeros(len(data))
        yerr = np.zeros(len(data))
        ratios[bkg_total!=0] = np.array(data[bkg_total!=0] / bkg_total[bkg_total!=0])
        yerr[bkg_total!=0] = np.sqrt(data[bkg_total!=0])/bkg_total[bkg_total!=0]
        ax_ratio = hep.histplot(ratios, edges, histtype='errorbar', yerr=yerr,**data_opts)
        unity = np.ones_like(bkg_total)
        zero = np.zeros_like(bkg_total)
        bkg_total[bkg_total==0] = 1e-20
        ggh[ggh==0] = 1e-20
        vbf[vbf==0] = 1e-20
        bkg_unc = coffea.hist.plot.poisson_interval(unity, bkg_sumw2 / bkg_total**2)
        denom_unc = bkg_unc
        ax_ratio.fill_between(edges,np.r_[denom_unc[0],denom_unc[0, -1]],np.r_[denom_unc[1], denom_unc[1, -1]], **ratio_err_opts)

    for v in hist.v.unique():
        for w in hist.w.unique():
            if ('nominal' not in v) and ('nominal' not in w): continue
            if ('nominal' in v) and ('nominal' in w): continue
            if ('off' in w): continue
            if ('Flavor' not in v): continue
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
            
    if compare_with_pisa:
        r_opt = 'inclusive' if r=='' else r
        pisa_hist = get_pisa_hist(var, r_opt, edges)
        ratio = np.zeros(len(bkg_total))
        ratio[bkg_total!=0] = np.array(pisa_hist[bkg_total!=0] / bkg_total[bkg_total!=0])
        ax = hep.histplot(ratio, edges, label='Pisa/Purdue', histtype='step', **{'linewidth':3, 'color':'red'})
        plt2.legend(prop={'size': 'small'})
        plt2.set_ylim([0.8,1.2])
        

    if save:
        # Save plots
        out_path = args['out_path']
        try:
            os.mkdir(out_path)
        except:
            pass
        try:
            os.mkdir(f"{out_path}/plots_{year}_{label}")
        except:
            pass
        if r=='':
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_inclusive.png"
        else:
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_{r}.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)

    if show:
        plt.show()
        

var_map_pisa = {
#'mu1_pt':'Mu0_pt_GeoFitCorrection',
'mu1_pt':'Mu0_pt',
'mu1_eta':'Mu0_eta',
#'mu2_pt':'Mu1_pt_GeoFitCorrection',
'mu2_pt':'Mu1_pt',
'mu2_eta':'Mu1_eta',
'dimuon_pt':'Higgs_pt', 
'dimuon_eta':'Higgs_eta',
#'dimuon_mass':'Higgs_m',
#'dimuon_mass':'Higgs_m_GF',
'dimuon_mass':'Higgs_m_noGF',
'jet1_pt':'QJet0_pt_touse',
#'jet1_pt':'QJet0_pt_nom',
'jet1_phi':'QJet0_phi',
'jet1_eta':'QJet0_eta',    
'jet1_qgl':'QJet0_qgl',
'jet2_pt':'QJet1_pt_touse',
#'jet2_pt':'QJet1_pt_nom',
'jet2_phi':'QJet1_phi',
'jet2_eta':'QJet1_eta',    
'jet2_qgl':'QJet1_qgl',
'jj_dEta':'qqDeltaEta',
'jj_dPhi':'qqDeltaPhi',
'jj_pt':'qq_pt',
'jj_mass':'Mqq',
# b'mmjj_pt', b'mmjj_pt_log', b'mmjj_pz', b'mmjj_pz_logabs', b'CS_theta', b'CS_phi', b'NSoft5NewNoRapClean', b'SAHT2', b'nGenPart', b'GenPart_pdgId', b'GenPart_eta', b'GenPart_phi', b'GenPart_pt', b'nLHEPart', b'LHEPart_pt', b'LHEPart_eta', b'LHEPart_phi', b'LHEPart_pdgId', b'DeltaRelQQ', b'DeltaEtaQQSum', b'PhiHQ1', b'PhiHQ2', b'EtaHQ1', b'EtaHQ2', b'minEtaHQ', b'Rpt', b'theta2', b'NSoft5', b'NSoft5New', b'SAHT', b'Jet_pT30_central', b'Jet_pT30', b'event', b'DNN18Classifier', b'SBClassifier', b'DNN18Atan', b'year', b'genWeight', b'puWeight', b'btagWeight', b'btagEventWeight', b'muEffWeight', b'EWKreweight', b'PrefiringWeight', b'QGLweight', b'QJet1_partonFlavour', b'QJet0_partonFlavour'
}

def get_pisa_hist(var, r, edges):
    import uproot
    filenames = {
        'dy_m105_160_amc':'/depot/cms/hmm/coffea/pisa/DY105_2018AMCPYSnapshot.root',
        'dy_m105_160_vbf_amc': '/depot/cms/hmm/coffea/pisa/DY105VBF_2018AMCPYSnapshot.root'
    }
    xsec = {
        'dy_m105_160_amc': 47.17, #
        'dy_m105_160_vbf_amc': 2.03 #
    }
    N = {
        'dy_m105_160_amc': 6995355211.029654,
        'dy_m105_160_vbf_amc': 3146552884.4507833
    }
    target_yields = { # as in datacards
        'dy_m105_160_amc':1677.5,
        'dy_m105_160_vbf_amc':1917.1,
    }
    qgl_mean = {
        'dy_m105_160_amc':  1.04859375342,
        'dy_m105_160_vbf_amc' :1.00809412662
    }

    lumi = 59970.
    samples = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc']
    total_hist = np.array([])
    for s in samples:
        with uproot.open(filenames[s]) as f:
            tree = f['Events']
            mjj = tree['Mqq'].array()
            deta_jj = tree['qqDeltaEta'].array()
            mmm = tree['Higgs_m_GF'].array()
            cut = {
                'h-peak':(mmm>115)&(mmm<135)&(mjj>400)&(deta_jj>2.5),
                'h-sidebands':(((mmm>110)&(mmm<115))|((mmm>135)&(mmm<150)))&(mjj>400)&(deta_jj>2.5),
                'inclusive': (mjj>400)&(deta_jj>2.5)
            }
#            cut = {l:np.ones(len(mmm), dtype=bool) for l in ['h-peak','h-sidebans','inclusive']}
            wgt = np.ones(len(mmm), dtype=float)
            weights = ['genWeight', 'puWeight', 'btagEventWeight','muEffWeight', 'EWKreweight', 'PrefiringWeight', 'QGLweight']
            for i,w in enumerate(weights):
                wgt = wgt*tree[w].array()
            wgt = wgt*xsec[s]*lumi / N[s] / qgl_mean[s] #tree['QGLweight'].array().mean()
            hist = bh.Histogram(bh.axis.Variable(edges))
            var_arr = tree[var_map_pisa[var.name]].array()
            hist.fill(var_arr[cut[r]], weight=wgt[cut[r]])
            if len(total_hist)>0:
                total_hist += hist.to_numpy()[0]
            else:
                total_hist = hist.to_numpy()[0]
            print(s, f'Yield: {hist.to_numpy()[0].sum()}, should be: {target_yields[s]}, ratio {hist.to_numpy()[0].sum()/target_yields[s]}')
    print(total_hist.sum())
    return total_hist


