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
        if 'data' in s:
            variations = ['nominal']
        else:
            variations = args['syst_variations']
        for v in variations:
            if v=='nominal':
                proc_outs = glob.glob(f"{path}/unbinned/{s}_?.coffea")
            else:
                proc_outs = glob.glob(f"{path}/{v}/{s}_?.coffea")
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
    suff = f'_{c}_{r}'
    columns = [c.replace(suff, '') for c in list(proc_out.keys()) if suff in c]
    df = pd.DataFrame()
    len_ = len(proc_out[f'dimuon_mass_{c}_{r}'].value)
    for var in columns:
        if (not args['wgt_variations']) and ('wgt_' in var) and ('nominal' not in var): continue
        if ('wgt_' not in var) and (var not in [v.name for v in args['vars_to_plot']]): continue
        try:
            df[var] = proc_out[f'{var}_{c}_{r}'].value
        except:
            if 'wgt_' in var:
                df[var] = proc_out[f'wgt_nominal_{c}_{r}'].value
            else:
                df[var] = np.zeros(len_, dtype=float)
    df['c'] = c
    df['r'] = r
    df['s'] = args['s']
    df['v'] = args['v']
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
    model_path = f'output/trained_models/test_{dnn_label}_hw.h5'
    with sess:
        dnn_model = load_model(model_path)
        df_eval = df[training_features]
        if args['r']!='h-peak':
            df_eval['dimuon_mass'] = 125.
        prediction = dnn_model.predict(df_eval).ravel()
        df['dnn_score'] = np.arctanh((prediction))
        print(df['dnn_score'])
    return df

def get_hists(df, var, args):
    dataset_axis = bh.axis.StrCategory(df.s.unique())
    region_axis = bh.axis.StrCategory(df.r.unique())
    channel_axis = bh.axis.StrCategory(df.c.unique())
    val_err_axis = bh.axis.StrCategory(['value', 'sumw2'])
    var_axis = bh.axis.Regular(var.nbins, var.xmin, var.xmax)

    df_out = pd.DataFrame()#columns = ['var','s','r','c','v','w']+[f'bin{i}' for i in range(var.nbins)]+\
                                                        #  [f'sumw2_{i}' for i in range(var.nbins)])
    edges = []
    syst_variations = args['syst_variations']
    wgts = [c for c in df.columns if ('wgt_' in c) and ('_off' not in c)]
    regions = df.r.unique()
    channels = df.c.unique()
    
    for s in df.s.unique():
        if 'data' in s:
            syst_variations = ['nominal']
            wgts = ['wgt_nominal']
        for v in syst_variations:
            for w in wgts:
                if (v!='nominal')&(w!='wgt_nominal'): continue
                hist = bh.Histogram(dataset_axis, region_axis, channel_axis, val_err_axis, var_axis)
                hist.fill(df.s.to_numpy(), df.r.to_numpy(), df.c.to_numpy(), 'value',\
                          df[var.name].to_numpy(), weight=df[w].to_numpy())
                hist.fill(df.s.to_numpy(), df.r.to_numpy(), df.c.to_numpy(), 'sumw2',\
                          df[var.name].to_numpy(), weight=(df[w]*df[w]).to_numpy())
                for r in regions:
                    for c in channels:
                        values = hist[loc(s), loc(r), loc(c), loc('value'), :].to_numpy()[0]
                        sumw2 = hist[loc(s), loc(r), loc(c), loc('sumw2'), :].to_numpy()[0]
                        edges = hist[loc(s), loc(r), loc(c), loc('value'), :].to_numpy()[1]
                        contents = {}
                        contents.update({f'bin{i}':[values[i]] for i in range(var.nbins)})
                        contents.update({f'sumw2_{i}':[sumw2[i]] for i in range(var.nbins)})
                        contents.update({'s':[s],'r':[r],'c':[c], 'v':[v], 'w':[w], 'var':[var.name]})
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
    for c in args['channels']:
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
                        data_obs = hist[hist.s.isin(data_names)&(hist.r==r)&(hist.c==c)]
                        data_obs_hist = data_obs[bin_columns].sum(axis=0).values
                        data_obs_sumw2 = data_obs[sumw2_columns].sum(axis=0).values
                    mc_hist = hist[~hist.s.isin(data_names)&(hist.v==v)&(hist.w==w)&(hist.r==r)&(hist.c==c)]
                    for s in mc_hist.s.unique():
                        if s in grouping.keys():
                            mc_hist.loc[hist.s==s,'group'] = grouping[s]
                    mc_hist = mc_hist.groupby('group').aggregate(np.sum).reset_index() 
                    for g in mc_hist.group.unique():
                        histo = mc_hist[mc_hist.group==g][bin_columns].values[0]
                        if len(histo)==0: continue
                        sumw2 = mc_hist[mc_hist.group==g][sumw2_columns].values[0]
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

def plot(var, hist, wgt_option, edges, args, r='', save=True, show=False, plotsize=12):
    def add_source(hist, group_name):
        bin_columns = [c for c in hist.columns if 'bin' in c]
        sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
        vals = hist[hist['group']==group_name]
        vals = vals.groupby('group').aggregate(np.sum).reset_index()
        sumw2 = vals[sumw2_columns].sum(axis=0).reset_index(drop=True).values 
        try:
            vals = vals[bin_columns].values[0] 
            return vals, sumw2
        except:
            print(group_name, "missing")
            return np.array([]), np.array([])
    
    hist = hist[var.name]
    hist = hist[hist.w==wgt_option]
    if r!='':
        hist = hist[hist.r==r]
        
    year = args['year']
    label = args['label']

    bkg = ['DY','DY_VBF', 'EWK', 'TT+ST', 'VV']

    hist.loc[:,'group'] = None
    for s in hist.s.unique():
        if s in grouping.keys():
            hist.loc[hist.s==s,'group'] = grouping[s]


    vbf, vbf_sumw2 = add_source(hist, 'VBF')
    ggh, ggh_sumw2 = add_source(hist, 'ggH')
    data, data_sumw2 = add_source(hist, 'Data')    

    bin_columns = [c for c in hist.columns if 'bin' in c]
    sumw2_columns = [c for c in hist.columns if 'sumw2' in c]
    bkg_df = hist[hist['group'].isin(bkg)]
#    print(bkg_df[bin_columns])
    bkg_df.loc[:,'integral'] = bkg_df[bin_columns].sum(axis=1)
    bkg_df = bkg_df.groupby('group').aggregate(np.sum).reset_index()
    bkg_df = bkg_df.sort_values(by='integral').reset_index(drop=True)
    bkg_total = bkg_df[bin_columns].sum(axis=0).reset_index(drop=True)    
    bkg_sumw2 = bkg_df[sumw2_columns].sum(axis=0).reset_index(drop=True)
    if len(bkg_df.values)>1:
        bkg = np.stack(bkg_df[bin_columns].values)
        stack=True
    else:
        bkg = bkg_df[bin_columns].values
        stack=False
    bkg_labels = bkg_df['group']

    
    # Report yields
    if not show:
        print("="*50)
        if r=='':
            print(f"{var.name}: Inclusive yields:")
        else:
            print(f"{var.name}: Yields in {r}")
        print("="*50)
        print('Data', data.sum())
        for row in bkg_df[['group','integral']].values:
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
        try:
            os.mkdir(out_path)
        except:
            pass
        try:
            os.mkdir(f"{out_path}/plots_{year}_{label}")
        except:
            pass

        out_path = args['out_path']
        if r=='':
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_{wgt_option}_inclusive.png"
        else:
            out_name = f"{out_path}/plots_{year}_{label}/{var.name}_{wgt_option}_{r}.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)

    if show:
        plt.show()
        






