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
    if 'get_hists' in modules:
        for var in args['vars_to_plot']:
            hists[var.name], edges[var.name] = get_hists(df, var, args)
            
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
    columns = [c.replace(suff, '') for c in list(proc_out.keys()) if suff in c]
    df = pd.DataFrame()
    len_ = len(proc_out[f'dimuon_mass_{c}_{r}'].value) if f'dimuon_mass_{c}_{r}' in proc_out.keys() else 0
    for var in columns:
        if (not args['wgt_variations']) and ('wgt_' in var) and ('nominal' not in var): continue
#        if ('wgt_' not in var) and (var not in [v.name for v in args['vars_to_plot']]): continue
        if (v!='nominal') and ('wgt_' in var) and ('nominal' not in var): continue
        if ('ggh' in s) and ('prefiring' in var): continue #just for tests to show '-' in datacards
        try:
            df[var] = proc_out[f'{var}_{c}_{r}'].value
        except:
            df[var] = proc_out[f'wgt_nominal_{c}_{r}'].value if 'wgt_' in var else np.zeros(len_, dtype=float)
            
    df['c'] = c
    df['r'] = r
    df['s'] = s
    df['v'] = v
    return df

def dnn_evaluation(df, args):
    dnn_label = args['year']
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

def get_hists(df, var, args):
    dataset_axis = bh.axis.StrCategory(df.s.unique())
    region_axis = bh.axis.StrCategory(df.r.unique())
    channel_axis = bh.axis.StrCategory(df.c.unique())
    syst_axis = bh.axis.StrCategory(df.v.unique())
    val_err_axis = bh.axis.StrCategory(['value', 'sumw2'])
    var_axis = bh.axis.Regular(var.nbins, var.xmin, var.xmax)

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
                        integral = values.sum()
                        sumw2 = hist[loc(s), loc(r), loc(c), loc(v), loc('sumw2'), :].to_numpy()[0]
                        edges = hist[loc(s), loc(r), loc(c), loc(v), loc('value'), :].to_numpy()[1]
                        contents = {}
                        contents.update({f'bin{i}':[values[i]] for i in range(var.nbins)})
                        contents.update({f'sumw2_{i}':[sumw2[i]] for i in range(var.nbins)})
                        contents.update({'g':grouping[s],'s':[s],'r':[r],'c':[c], 'v':[v], 'w':[w],\
                                         'var':[var.name], 'integral':integral})
                        row = pd.DataFrame(contents)
                        df_out = pd.concat([df_out, row], ignore_index=True)
    return df_out, edges

def save_shapes(var, hist, edges, args):    
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
    
    hist = hist[var.name]
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    data_names = [n for n in hist.s.unique() if 'data' in n]
#    for c in args['channels']: # Unite channels here
    for c,cg in args['channel_groups'].items():
        for r in args['regions']:
            out_fn = f'combine_new/shapes_{c}_{r}_{args["year"]}_{args["label"]}.root'
            out_file = uproot.recreate(out_fn)
            data_obs_hist = np.zeros(len(bin_columns), dtype=float)
            data_obs_sumw2 = np.zeros(len(sumw2_columns), dtype=float)
            for v in hist.v.unique():
                for w in hist.w.unique():
                    vwname = get_vwname(v,w)
                    if vwname == '': continue
                    if vwname == 'nominal':
                        #data_obs = hist[hist.s.isin(data_names)&(hist.r==r)&(hist.c==c)]
                        data_obs = hist[hist.s.isin(data_names)&(hist.r==r)&(hist.c.isin(cg))]                        
                        data_obs_hist = data_obs[bin_columns].sum(axis=0).values
                        data_obs_sumw2 = data_obs[sumw2_columns].sum(axis=0).values
                    #mc_hist = hist[~hist.s.isin(data_names)&(hist.v==v)&(hist.w==w)&(hist.r==r)&(hist.c==c)]
                    mc_hist = hist[~hist.s.isin(data_names)&(hist.v==v)&(hist.w==w)&(hist.r==r)&(hist.c.isin(cg))]
                    mc_hist = mc_hist.groupby('g').aggregate(np.sum).reset_index() 
                    for g in mc_hist.g.unique():
                        histo = mc_hist[mc_hist.g==g][bin_columns].values[0]
                        if len(histo)==0: continue
                        sumw2 = mc_hist[mc_hist.g==g][sumw2_columns].values[0]
                        rname = r.replace('-','_')
                        name = f'{rname}_{g}_{vwname}'
                        th1 = from_numpy([histo, edges])
                        th1._fName = name
                        th1._fSumw2 = np.array(sumw2)
                        th1._fTsumw2 = np.array(sumw2).sum()
                        th1._fTsumwx2 = np.array(sumw2 * centers).sum()
                        out_file[f'{g}_{vwname}'] = th1
            th1_data = from_numpy([data_obs_hist, edges])
            th1_data._fName = 'data_obs'
            th1_data._fSumw2 = np.array(data_obs_sumw2)
            th1_data._fTsumw2 = np.array(data_obs_sumw2).sum()
            th1_data._fTsumwx2 = np.array(data_obs_sumw2 * centers).sum()
            out_file['data_obs'] = th1_data
            out_file.close()

rate_syst_lookup = {
    '2016':{
        'DY':{'XsecAndNorm': 1.12189, 'lumi': 1.025},
        'EWK':{'XsecAndNorm': 1.06217,'lumi': 1.025},
        'TT+ST':{'XsecAndNorm': 1.18261,'lumi': 1.025},
        'VV':{'XsecAndNorm': 1.0609,'lumi': 1.025},
        'ggH':{'XsecAndNorm': 1.36133,'lumi': 1.025},
        'VBF':{'lumi': 1.025},
        },
    '2017':{
        'DY':{'XsecAndNorm': 1.12452, 'lumi': 1.025},
        'EWK':{'XsecAndNorm': 1.05513,'lumi': 1.025},
        'TT+ST':{'XsecAndNorm': 1.18402,'lumi': 1.025},
        'VV':{'XsecAndNorm': 1.05734,'lumi': 1.025},
        'ggH':{'XsecAndNorm': 1.36667,'lumi': 1.025},
        'VBF':{'lumi': 1.025},
        },
    '2018':{
        'DY':{'XsecAndNorm': 1.12152, 'lumi': 1.025},
        'EWK':{'XsecAndNorm': 1.05851,'lumi': 1.025},
        'TT+ST':{'XsecAndNorm': 1.18592,'lumi': 1.025},
        'VV':{'XsecAndNorm': 1.05734,'lumi': 1.025},
        'ggH':{'XsecAndNorm': 1.39295,'lumi': 1.025},
        'VBF':{'lumi': 1.025},
        },
}            
            
def get_numbers(hist, bin_name, args):
    groups = hist.g.unique()
    
    floating_norm = {'DY':['vbf_01j','vbf_2j']}
    sig_groups = ['ggH', 'VBF']
    sig_counter = 0
    bkg_counter = 0
    
    systs = []
    shape_systs = [w.replace('_up','').replace('wgt_','') for w in hist.w.unique() if '_up' in w]
    shape_systs.extend(['jes'+v.replace('_up','') for v in hist.v.unique() if '_up' in v])
    lnn_systs = ['XsecAndNorm'+g for g in groups if 'Data' not in g]
    lnn_systs.extend(['lumi'])
    systs.extend(shape_systs)
    systs.extend(lnn_systs)

    data_yields = pd.DataFrame()
    data_yields['index'] = ['bin','observation']
    
    mc_yields = pd.DataFrame()
    mc_yields['index'] = ['bin','process','process','rate']
    
    systematics = pd.DataFrame(index=systs)

    for g in groups:
        counter = 0
        if g in sig_groups:
            sig_counter += 1
            counter = -sig_counter
        elif 'Data'not in g:
            bkg_counter += 1
            counter = bkg_counter
        hist_g = hist[(hist.g==g)]
        systs_g = [w.replace('_up','').replace('wgt_','') for w in hist_g.w.unique() if '_up' in w]
        systs_g.extend(['jes'+v.replace('_up','') for v in hist_g.v.unique() if '_up' in v])
        hist_g = hist_g[(hist_g.v=='nominal')&(hist_g.w=='wgt_nominal')].groupby('g').aggregate(np.sum).reset_index()
        rate = hist_g.integral.values[0]
        if g=='Data':
            data_yields.loc[0,'value'] = bin_name
            data_yields.loc[1,'value'] = f'{rate}'
        else:
            mc_yields.loc[0,g] = bin_name
            mc_yields.loc[1,g] = g
            mc_yields.loc[2,g] = f'{counter}'
            mc_yields.loc[3,g] = f'{rate}'
            for syst in shape_systs:
                systematics.loc[syst,'type'] = 'shape'
                systematics.loc[syst,g] = '1.0' if syst in systs_g else '-'
            for syst in lnn_systs:
                systematics.loc[syst,'type'] = 'lnN'
                val = rate_syst_lookup[args['year']][g][syst.replace(g,'')] if syst.replace(g,'') in\
                                            rate_syst_lookup[args['year']][g].keys() else '-'
                systematics.loc[syst,g] = f'{val}'
                    
    def to_string(df):
        string = ''
        for row in df.values:
            for i,item in enumerate(row):
                ncols = 2 if item in ['bin','process','rate', 'observation'] else 1
                row[i] = item+' '*(ncols*20-len(item))
            string += ' '.join(row)
            string += '\n'
        return string
    return to_string(data_yields), to_string(mc_yields), to_string(systematics.reset_index())
    
        
            
def make_datacards(var, hist, args):
    r_names = {'h-peak':'SR','h-sidebands':'SB'}
    hist = hist[var.name]
#    for c in args['channels']: # Unite channels here
    for c,cg in args['channel_groups'].items():
        for r in args['regions']:            
            datacard_name = f'combine_new/datacard_{c}_{r}_{args["year"]}_{args["label"]}.txt'
            shapes_file = f'combine_new/shapes_{c}_{r}_{args["year"]}_{args["label"]}.root'
            datacard = open(datacard_name, 'w')
            datacard.write(f"imax 1\n") # will combine the datacards later
            datacard.write(f"jmax *\n")
            datacard.write(f"kmax *\n")
            datacard.write("---------------\n")
            datacard.write(f"shapes * {r} {shapes_file} $PROCESS $PROCESS_$SYSTEMATIC\n")
            datacard.write("---------------\n")
            bin_name = f'{r_names[r]}_{args["year"]}'
#            ret = data_yields, mc_yields, systematics = get_numbers(hist[(hist.c==c) & (hist.r==r)], bin_name, args)
            print(hist)
            ret = data_yields, mc_yields, systematics = get_numbers(hist[(hist.c.isin(cg)) & (hist.r==r)], bin_name, args)
            datacard.write(data_yields)
            datacard.write("---------------\n")
            datacard.write(mc_yields)
            datacard.write("---------------\n")
            datacard.write(systematics)
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
        print(group_name, "missing")
        return np.array([]), np.array([])
        
def plot(var, hist, wgt_option, edges, args, r='', save=True, show=False, plotsize=12):    
    hist = hist[var.name]
    hist = hist[(hist.w==wgt_option)&(hist.v=='nominal')]
    if r!='':
        hist = hist[hist.r==r]
        
#    if 'dnn_score' in var.name:
#        var.xmax = hist.max_score.max()
        
    year = args['year']
    label = args['label']

    bkg = ['DY','DY_VBF', 'EWK', 'TT+ST', 'VV']

    vbf, vbf_sumw2 = add_source(hist, 'VBF')
    ggh, ggh_sumw2 = add_source(hist, 'ggH')
    data, data_sumw2 = add_source(hist, 'Data')    

    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    bkg_df = hist[hist['g'].isin(bkg)]
    bkg_df.loc[:,'bkg_integral'] = bkg_df[bin_columns].sum(axis=1)
    bkg_df = bkg_df.groupby('g').aggregate(np.sum).reset_index()
    bkg_df = bkg_df.sort_values(by='bkg_integral').reset_index(drop=True)
    bkg_total = bkg_df[bin_columns].sum(axis=0).reset_index(drop=True)    
    bkg_sumw2 = bkg_df[sumw2_columns].sum(axis=0).reset_index(drop=True)
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
    if ggh.sum():
        ax_ggh = hep.histplot(ggh, edges, label='ggH', histtype='step', **{'linewidth':3, 'color':'lime'})
    if vbf.sum():
        ax_vbf = hep.histplot(vbf, edges, label='VBF', histtype='step', **{'linewidth':3, 'color':'aqua'})
    if data.sum():
        ax_data = hep.histplot(data, edges, label='Data', histtype='errorbar', yerr=np.sqrt(data), **data_opts)
    
    lbl = hep.cms.cmslabel(ax=plt1, data=True, paper=False, year=year)
    
    plt1.set_yscale('log')
    plt1.set_ylim(0.01, 1e9)
    plt1.set_xlim(var.xmin,var.xmax)
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
    
    plt2.axhline(1, ls='--')
    plt2.set_ylim([0.5,1.5])    
    plt2.set_ylabel('Data/MC')
    lbl = plt2.get_xlabel()
    plt2.set_xlabel(f'{var.caption}')
    
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
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_{wgt_option}_inclusive.png"
        else:
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_{wgt_option}_{r}.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)

    if show:
        plt.show()
        






