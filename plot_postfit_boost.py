import uproot
from python.plotting import Plotter
from python.dimuon_processor import DimuonProcessor
import numpy as np
import pandas as pd
import boost_histogram as bh
from boost_histogram import loc
import coffea
import matplotlib.pyplot as plt
import mplhep as hep
import json
year = '2018'
#suffix = ''
suffix = '_CRfit'
input_file = uproot.open(f"combine/fitDiagnostics{year}{suffix}.root")
#print(input_file['shapes_prefit'].keys())
#print(input_file['shapes_fit_b'][b'h_peak;1'][b'ggh_amcPS;1'].values)

regions = ['ch1', 'ch2_h_sidebands', 'ch2_z_peak']
#regions = ['h_peak', 'h_sidebands', 'z_peak']
#regions = ['h_peak', 'h_sidebands']
#regions = ['z_peak']

shapes = {
    'prefit':input_file['shapes_prefit'],
    'postfit':input_file['shapes_fit_b']
}

all_bkg_sources = {
    'DY': ['dy', 'dy_0j', 'dy_1j', 'dy_2j', 'dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'dy_m105_160_mg', 'dy_m105_160_vbf_mg'],
    'EWK': ['ewk_lljj_mll50_mjj120','ewk_lljj_mll105_160', 'ewk_lljj_mll105_160_ptj0'],
    'TTbar + Single Top':['ttjets_dl', 'ttjets_sl', 'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
    'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
}

all_bkg = []
for group, bkgs in all_bkg_sources.items():
    all_bkg += bkgs 
print(all_bkg)

signal = ['ggh_amcPS', 'vbf_powhegPS']
all_datasets = all_bkg+['data']+signal

channels = ['vbf']
systematics = ['nominal']

from coffea import hist

nbins = 12

dataset_axis = bh.axis.StrCategory(all_datasets)
region_axis = bh.axis.StrCategory(regions)
channel_axis = bh.axis.StrCategory(channels)
syst_axis = bh.axis.StrCategory(systematics)
var_axis = bh.axis.Regular(nbins, 0, nbins)
val_err_axis = bh.axis.StrCategory(['value','err_hi','err_lo','sumw2'])
hists = {}


for option in ['prefit', 'postfit']:

    hists[option] = bh.Histogram(dataset_axis, region_axis, channel_axis, syst_axis, var_axis, val_err_axis)

    for region_ in shapes[option].keys():
        for sample_ in shapes[option][region_].keys():
            sample = sample_.decode("utf-8")[:-2]
            region = region_.decode("utf-8")[:-2]
            if sample not in all_datasets: continue
            if 'total' in sample: continue

            if 'data' in sample:
                values = shapes[option][region][sample].yvalues
                err_hi = shapes[option][region][sample].yerrorshigh
                err_lo = shapes[option][region][sample].yerrorslow
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('value')] = values
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('err_hi')] = err_hi
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('err_lo')] = err_lo
            elif sample in signal:
                values = shapes['prefit'][region][sample].values
                sumw2 = shapes['prefit'][region][sample].variances
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('value')] = values
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('sumw2')] = sumw2
            else:
#                print(shapes[option][region][sample].__dict__)
                values = shapes[option][region][sample].values
                sumw2 = shapes[option][region][sample].variances
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('value')] = values
                hists[option][loc(sample), loc(region), loc('vbf'), loc('nominal'), :, loc('sumw2')] = sumw2

def plot_data_mc(hist, datasets, r, c, s, p):
    bkg_df = pd.DataFrame()
    bkg_labels = []
    ggh = []
    vbf = []
    data = []
    edges = []

    def get_hist(hist, d_,r_,c_,s_,v_):
#        return hist[loc(d_), loc(r_), loc(c_), loc(s_), ::bh.rebin(4), loc(v_)]
        return hist[loc(d_), loc(r_), loc(c_), loc(s_), :, loc(v_)]
    
    for d in datasets:
        if 'data' in d:
            data = get_hist(hist,d,r,c,s,'value').to_numpy()[0]
            data_err_hi = get_hist(hist,d,r,c,s,'err_hi').to_numpy()[0]
            data_err_lo = get_hist(hist,d,r,c,s,'err_lo').to_numpy()[0]
            if ('h_peak' in r) or ('ch1' in r):
                data[-6:] = 0
        elif 'ggh_amcPS' in d:
            ggh = get_hist(hist,d,r,c,s,'value').to_numpy()[0]
            ggh_sumw2 = get_hist(hist,d,r,c,s,'sumw2').to_numpy()[0]
        elif 'vbf_powhegPS' in d:
            vbf = get_hist(hist,d,r,c,s,'value').to_numpy()[0]
            vbf_sumw2 = get_hist(hist,d,r,c,s,'sumw2').to_numpy()[0]
        else:
            bin_contents = get_hist(hist,d,r,c,s,'value').to_numpy()[0]
            bin_sumw2 = get_hist(hist,d,r,c,s,'sumw2').to_numpy()[0]
            edges = get_hist(hist,d,r,c,s,'value').to_numpy()[1]
            contents = {f'bin{i}':[bin_content] for i,bin_content in enumerate(bin_contents)}
            sumw2s = {f'sumw2_{i}':[smw2] for i, smw2 in enumerate(bin_sumw2)}
            contents.update(sumw2s)
            contents['name']=[d]
            for group, sources in all_bkg_sources.items():
                if d in sources:
                    contents['group'] = [group]
            bkg_df = bkg_df.append(pd.DataFrame(data=contents))
            bkg_labels += [d]
            

    bin_columns = [c for c in bkg_df.columns if 'bin' in c]
    sumw2_columns = [c for c in bkg_df.columns if 'sumw2' in c]
    bkg_df['integral'] = bkg_df[bin_columns].sum(axis=1)

    bkg_df = bkg_df.groupby('group').aggregate(np.sum).reset_index()
    bkg_df = bkg_df.sort_values(by='integral').reset_index(drop=True)
#    print(bkg_df[bin_columns])
    bkg_total = bkg_df[bin_columns].sum(axis=0).reset_index(drop=True)
    sumw2_total = bkg_df[sumw2_columns].sum(axis=0).reset_index(drop=True)

    bkg = np.stack(bkg_df[bin_columns].values)
    bkg_labels = bkg_df['group']
    fig = plt.figure()
    plt.rcParams.update({'font.size': 22})
    fig.clf()
    plotsize=12
    ratio_plot_size = 0.25
    fig.set_size_inches(plotsize, plotsize*(1+ratio_plot_size))
    gs = fig.add_gridspec(2, 1, height_ratios=[(1-ratio_plot_size),ratio_plot_size], hspace = .05)
    
    plt1 = fig.add_subplot(gs[0])

    data_opts = {'color': 'k', 'marker': '.', 'markersize':15}
    stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0)}
    stack_error_opts = {'label':'Stat. unc.','facecolor':(0,0,0,.4), 'hatch':'', 'linewidth': 0}

    if bkg_total.sum():
        ax_bkg = hep.histplot(bkg, edges, ax=plt1, label=bkg_labels, stack=True, histtype='fill', **stack_fill_opts)
        err = coffea.hist.plot.poisson_interval(np.array(bkg_total), sumw2_total)
        opts = {'step': 'post', 'label': 'Stat. unc.', 'hatch': '//////',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0}
        ax_bkg.fill_between(x=edges, y1=np.r_[err[0, :], err[0, -1]],
                    y2=np.r_[err[1, :], err[1, -1]], **opts)
    if ggh.sum():
        ax_ggh = hep.histplot(ggh, edges, label='ggH', histtype='step', **{'linewidth':3, 'color':'lime'})
    if vbf.sum():
        ax_vbf = hep.histplot(vbf, edges, label='VBF', histtype='step', **{'linewidth':3, 'color':'aqua'})
    if data.sum():
        ax_data = hep.histplot(data, edges, label='Data', histtype='errorbar', yerr=[data_err_lo, data_err_hi], **data_opts)

    
    lbl = hep.cms.cmslabel(ax=plt1, data=True, paper=False, year=year)
    
    plt1.set_yscale('log')
    plt1.set_ylim(0.01, 1e9)
    plt1.set_xlim(0,nbins)
    plt1.set_xlabel('')
    plt1.tick_params(axis='x', labelbottom=False)
    plt1.legend(prop={'size': 'small'})
    
        # Bottom panel: Data/MC ratio plot
    plt2 = fig.add_subplot(gs[1], sharex=plt1)
    
    ratios = np.array(data / bkg_total)
    if (data.sum()*bkg_total.sum()):
        ax_ratio = hep.histplot(ratios, edges,  histtype='errorbar', yerr=[data_err_lo/bkg_total, data_err_hi/bkg_total],  **data_opts)
    
        unity = np.ones_like(bkg_total)
        zero = np.zeros_like(bkg_total)
        bkg_unc = coffea.hist.plot.poisson_interval(unity, sumw2_total / bkg_total**2)
        ggh_unc = coffea.hist.plot.poisson_interval(unity, ggh_sumw2 / ggh**2)
        vbf_unc = coffea.hist.plot.poisson_interval(unity, vbf_sumw2 / vbf**2)
        denom_unc = bkg_unc
        opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3), 'linewidth': 0}
        ax_ratio.fill_between(edges, np.r_[denom_unc[0], denom_unc[0, -1]], np.r_[denom_unc[1], denom_unc[1, -1]], **opts)
    
    plt2.axhline(1, ls='--')
    plt2.set_ylim([0.5,1.5])    
    plt2.set_ylabel('Data/MC')
    lbl = plt2.get_xlabel()
    lbl = lbl if lbl else 'DNN score'
    plt2.set_xlabel(f'{lbl}, {r}, {c} channel')
                
    fig.savefig(f"plots/postfit/test_{r}_{c}_{s}_{p}_{year}{suffix}.png")

def save_to_datacard(hist, datasets, r, c, s, p):
    from uproot_methods.classes.TH1 import from_numpy
    def get_hist(hist, d_,r_,c_,s_,v_):
        return hist[loc(d_), loc(r_), loc(c_), loc(s_), :, loc(v_)]

    global year
    global suffix
    norms = {}
    out_fn = f'plots/postfit/shapes_{r}_{year}{suffix}.root'
    out_file = uproot.recreate(out_fn)
    data_obs_hist = np.zeros(nbins, dtype=float)
    data_obs_sumw2 = np.zeros(nbins, dtype=float)
    for d in datasets:
        histogram = get_hist(hist,d,r,c,s,'value').to_numpy()[0]
        norms[d] = histogram.sum()
        sumw2 = get_hist(hist,d,r,c,s,'sumw2').to_numpy()[0]
        edges = get_hist(hist,d,r,c,s,'value').to_numpy()[1]
        centers = (edges[:-1] + edges[1:]) / 2.0
        name = f'{r}_{d}'
        if 'data' in d:
            data_obs_hist = data_obs_hist + histogram
            data_obs_sumw2 = data_obs_sumw2 + histogram**2
        else:
            th1 = from_numpy([histogram, edges])
            th1._fName = name
            th1._fSumw2 = np.array(sumw2)
            th1._fTsumw2 = np.array(sumw2).sum()
            th1._fTsumwx2 = np.array(sumw2 * centers).sum()
            out_file[d] = th1
    th1_data = from_numpy([data_obs_hist, edges])
    th1_data._fName = 'data_obs'
    th1_data._fSumw2 = np.array(data_obs_sumw2)
    th1_data._fTsumw2 = np.array(data_obs_sumw2).sum()
    th1_data._fTsumwx2 = np.array(data_obs_sumw2 * centers).sum()
    out_file['data_obs'] = th1_data
    out_file.close()
    return norms
            
            
for p in ['prefit','postfit']:
    norms = {}
    for r in regions:
        norms[r] = {}
        for c in channels:
            for s in systematics:
                plot_data_mc(hists[p], all_datasets, r, c, s, p)
                if 'postfit' in p:
                    norms[r] = save_to_datacard(hists[p], all_datasets, r, c, s, p)
                
print(norms)
with open(f'plots/postfit/norms_{year}.json', 'w') as fp:
    json.dump(norms, fp)
