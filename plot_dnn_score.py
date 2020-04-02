from coffea import hist
import pandas as pd
import numpy as np
import glob
import json
import uproot


def dnn_rebin(df, vbf_name, r, nbins=0, bin_yield=0):
    # assuming dnn score is in [0;1]
    if nbins==0 and bin_yield==0:
        raise Exception
    print('Rebinning...')

    global norms
    factor = norms[r][vbf_name]/df['weight'].sum()
    print(factor)
    vbf_yield = df['weight'].sum()*factor
    if nbins>0:
        bin_yield = vbf_yield/nbins
    elif bin_yield>0:
        nbins = int(vbf_yield/bin_yield)+1

    boundaries = []
    nsteps = 1000
    for ibin in range(nbins):
        for i in range(nsteps):
            ii = 1 - (i/nsteps)
            if df[df['dnn_score']>ii].weight.sum()*factor >= bin_yield*(ibin+1):
                boundaries.append(ii)
                break
#    print(boundaries)
    return sorted(boundaries)

year = '2016'
path_label = 'mar17'
label = 'm125'
do_jecunc=False
if do_jecunc:
    path_label += '_jecunc'

tmp_path = f'/depot/cms/hmm/coffea/tmp_{label}_{year}_{path_label}/'
tmp_path_dnn =  f'/depot/cms/hmm/coffea/tmp_{label}_{year}_{path_label}_dnn/'
#print(tmp_path)
#print(tmp_path_dnn)
dfs = []
systematics = ['nominal', 'muSF_up', 'muSF_down', 'pu_weight_up', 'pu_weight_down']
if '2018' not in year:
    systematics = systematics + ['l1prefiring_weight_up', 'l1prefiring_weight_down']
score_rescaling = 1

with open(f"output/norms_{year}.json") as json_file:
    norms = json.load(json_file)

xmax_ = {
    '2016': 1.75,
    '2017': 2.0,
    '2018': 2.35
}

for file in glob.glob(tmp_path+'/*'):
#    if 'vbf_powheg'not in file: continue
#    if ('wz' not in file) and ('ww' not in file): continue
    try:
        df_ = pd.DataFrame(data=np.load(file, allow_pickle=True), columns=['dnn_score', 'dataset', 'region', 'channel', 'weight']+[f'weight_{s}' for s in systematics])
        dfs.append(df_)
    except:
        print(f"{file}: error")
        
df = pd.concat(dfs).reset_index(drop=True)
df = df[~df.isna().any(axis=1)]

vbf_name = {
    'm120': 'vbf_powhegPS_m120',
    'm125': 'vbf_powhegPS',
    'm130': 'vbf_powhegPS_m130',
}

df_vbf = {
    'z-peak': df[(df['dataset']==vbf_name[label]) & (df['region']=='z-peak')],
    'h-sidebands': df[(df['dataset']==vbf_name[label]) & (df['region']=='h-sidebands')],
    'h-peak': df[(df['dataset']==vbf_name[label]) & (df['region']=='h-peak')],
}
df['dnn_bin'] = np.zeros(df.shape[0], dtype=float)
dnn_bins = {}
nbins = 9
if score_rescaling==1:
    xmax = 9
elif score_rescaling==2:
    xmax = xmax_[year]

nbins_r = {}

for r in ['z-peak', 'h-peak', 'h-sidebands']:
    print(r)
    if score_rescaling ==1:
        dnn_bins[r] = dnn_rebin(df_vbf[r], vbf_name[label], r, 0, 0.7)
        print(dnn_bins[r])
        nbins_r[r] = len(dnn_bins[r])-1
        for i in range(len(dnn_bins[r])-1):
            left = dnn_bins[r][i]
            right = dnn_bins[r][i+1]
#            print(df.loc[(df['region']==r) & (df['dnn_score']>=left)&(df['dnn_score']<right)&(df['dataset']==vbf_name[label]), 'weight_nominal'].sum())
            df.loc[(df['region']==r) & (df['dnn_score']>=left)&(df['dnn_score']<right), 'dnn_bin'] = i#(i+1)/nbins
    elif score_rescaling==2:
        dnn = np.array(df[(df['region']==r)].dnn_score.values, dtype=float)
        df.loc[(df['region']==r), 'dnn_bin'] = np.arctanh((dnn))

datasets = df.dataset.unique()
for r in ['z-peak', 'h-peak', 'h-sidebands']:
    for ds in datasets:
        print(ds)
        print('='*20)
        for ibin in range(nbins_r[r]+1):
            factor = norms[r][ds]/df.loc[(df['region']==r)&(df['dataset']==ds)]['weight'].sum()
            sum_ = df.loc[(df['region']==r)&(df['dnn_bin']==ibin)&(df['dataset']==ds)].weight.sum()*factor
            print(f'{sum_}')
        print('='*20)
#df['dnn_bin'] = np.arctanh(np.array(df.dnn_score.values,dtype=float))

dataset_axis = hist.Cat("dataset", "")
region_axis = hist.Cat("region", " ") # Z-peak, Higgs SB, Higgs peak
channel_axis = hist.Cat("channel", " ") # ggh or VBF   
syst_axis = hist.Cat("syst","")
dnn_axis = hist.Bin('dnn_score', 'DNN score', nbins, 0, xmax)

datasets = df.dataset.unique()
regions = df.region.unique()
channels = df.channel.unique()
print(regions)

accumulators = {}
accumulators['dnn_score'] = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, syst_axis, dnn_axis)
for dataset in datasets:
    if 'data' in datasets:
        print(df[(df['dataset']==dataset)&(df['region']==region)&(df['channel']==channel)].dnn_bin.values)
    this_accumulator = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, syst_axis, dnn_axis)
    for region in regions:
#        if ('h-peak' in region) and ('data' in dataset): continue
        for channel in channels:
            # add loop over systematics
            for syst in systematics:
                dnn_scores = df[(df['dataset']==dataset)&(df['region']==region)&(df['channel']==channel)].dnn_bin.values
                weights = df[(df['dataset']==dataset)&(df['region']==region)&(df['channel']==channel)][f'weight_{syst}'].values
#                print(dataset, region, channel, syst, np.array(dnn_scores, dtype=float), np.array(weights, dtype=float))
                this_accumulator.fill(**{'dataset': dataset, 'region': region, 'channel': channel,\
                                             'dnn_score': np.array(dnn_scores, dtype=float), 'syst': syst, 'weight': np.array(weights, dtype=float)})
    accumulators['dnn_score'] = accumulators['dnn_score']+this_accumulator
print(datasets)


from python.plotting import Plotter
from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo

ggh_name = {
    'm120': 'ggh_amcPS_m120',
    'm125': 'ggh_amcPS',
    'm130': 'ggh_amcPS_m130',
}

vbf_name = {
    'm120': 'vbf_powhegPS_m120',
    'm125': 'vbf_powhegPS',
    'm130': 'vbf_powhegPS_m130',
}

ewk_name = {
    '2016':'ewk_lljj_mll105_160_ptj0',
    '2017':'ewk_lljj_mll105_160_ptj0',
#    '2018':'ewk_lljj_mll105_160_ptj0'
#    '2016':'ewk_lljj_mll105_160',
#    '2017':'ewk_lljj_mll105_160',
    '2018':'ewk_lljj_mll105_160'
}

syst_sources = {
    '2016':['muSF', 'pu_weight', 'l1prefiring_weight'],
    '2017':['muSF', 'pu_weight', 'l1prefiring_weight'],
    '2018':['muSF', 'pu_weight',]
}

#with open(f"output/norms_{year}.json") as json_file:
#    norms = json.load(json_file)

pars = {
    'processor': DimuonProcessor(samp_info=SamplesInfo(year)),
    'accumulators': accumulators,
    'chunked': True,
    'vars': ['dnn_score'],
    'year': year,
    'regions' : ["z-peak","h-sidebands", "h-peak"],
#     'regions' : ["h-sidebands", "h-peak"],
    'channels': ["vbf"],
    'ggh_name': ggh_name[label],
    'vbf_name': vbf_name[label],
    'ewk_name': ewk_name[year],
    'syst_sources': syst_sources[year],
    'norms': norms
}

plots = Plotter(**pars)
plots.make_datamc_comparison(do_inclusive=False, do_exclusive=True, normalize=False, logy=True, get_rates=True)
plots.make_datamc_comparison(do_inclusive=False, do_exclusive=True, normalize=False, logy=True, get_rates=True, save_to='plots/dnn_score/')    
from uproot_methods.classes.TH1 import from_numpy

xmin = 0.


binw = (xmax-xmin)/nbins
# edges = [xmin-binw]
edges = []
for i in range(nbins):
    e = xmin+i*binw
    edges.append(e)
edges.append(xmax)
edges = np.array(edges)
centers = (edges[:-1] + edges[1:]) / 2.0

bin_map = {
    'z-peak':'z_peak',
    'h-peak':'h_peak',
    'h-sidebands':'h_sidebands',
}

from config.parameters import parameters
parameters = {k:v[year] for k,v in parameters.items()}

def get_variated_hist(ds, r, c, syst, variation, hist_nom):
    global tmp_path_dnn
    dnn_bins = []
    for i in range(nbins):
        dnn_bins.append(f'dnn_{i}')
    file = tmp_path_dnn+f"temp_{ds}.npy"
    
    df_ = pd.DataFrame(data=np.load(file, allow_pickle=True), columns=dnn_bins+['variation', 'dataset', 'region', 'channel'])
    nominal_integral = df_[((df_['dataset']==ds) & (df_['region']==r) & (df_['channel']==c) & (df_['variation']=='nominal'))][dnn_bins].values.flatten().sum()
    target_integral = hist_nom.sum()
    factor = target_integral / nominal_integral
#    print(factor)
    
    if variation=='nominal':
        return([[],[]])       
    else:
        result = []
        for ud in ['_up', '_down']:
            mask = (df_['dataset']==ds) & (df_['region']==r) & (df_['channel']==c) & (df_['variation']==variation+ud)
            result.append(df_[mask][dnn_bins].values.flatten()*factor)

    return result

for r in ['z-peak', 'h-peak', 'h-sidebands']:
    out_fn = f'combine/datacard_{r}_{label}_{year}.root'
    out_file = uproot.recreate(out_fn)
    data_obs_hist = np.zeros(nbins, dtype=float)
    data_obs_sumw2 = np.zeros(nbins, dtype=float)
    for dataset in plots.datasets[r]:
        if dataset not in plots.hist_dict[r]: continue
        histogram = plots.hist_dict[r][dataset]['hist']
        nominal_histogram = histogram
        sumw2 = plots.hist_dict[r][dataset]['sumw2']
        sumw2 = sumw2[1:-2]
#        print(dataset, sumw2)
        name = f'{bin_map[r]}_{dataset}'
        if 'data' in dataset:
            data_obs_hist = data_obs_hist + histogram
            data_obs_sumw2 = data_obs_sumw2 + sumw2
        else:
            th1 = from_numpy([histogram, edges])
            th1._fName = name
            th1._fSumw2 = np.array(sumw2)
            th1._fTsumw2 = np.array(sumw2).sum()
            th1._fTsumwx2 = np.array(sumw2 * centers).sum()
            out_file[dataset] = th1
        for syst in plots.syst_sources:
            if dataset not in plots.hist_syst[r]: continue
            if 'data' in dataset: continue
            for updown in ['Up','Down']:
                histogram = plots.hist_syst[r][dataset][f'{syst}{updown}']['hist']
                sumw2 = plots.hist_syst[r][dataset][f'{syst}{updown}']['sumw2']
                sumw2 = sumw2[1:-2]
                name = f'{bin_map[r]}_{dataset}_{syst}{updown}'
                th1 = from_numpy([histogram, edges])
                th1._fName = name
                th1._fSumw2 = np.array(sumw2)
                th1._fTsumw2 = np.array(sumw2).sum()
                th1._fTsumwx2 = np.array(sumw2 * centers).sum()
                out_file[f'{dataset}_{syst}{updown}'] = th1
        for variation in parameters["jec_unc_to_consider"]:
            if variation=='nominal': continue
            hist_updown = get_variated_hist(dataset, r, 'vbf', 'nominal', variation, nominal_histogram)
            for iud, ud in enumerate(['Up','Down']):
                hist_ud = hist_updown[iud]
                sumw2 = hist_ud**2 #wrong but maybe doesn't matter
                name = f'{bin_map[r]}_{dataset}_{variation}{ud}'
                th1 = from_numpy([hist_ud, edges])
                th1._fName = name
                th1._fSumw2 = np.array(sumw2)
                th1._fTsumw2 = np.array(sumw2).sum()
                th1._fTsumwx2 = np.array(sumw2 * centers).sum()
                out_file[f'{dataset}_{variation}{ud}'] = th1
    th1_data = from_numpy([data_obs_hist, edges])
    th1_data._fName = 'data_obs'
    th1_data._fSumw2 = np.array(data_obs_sumw2)
    th1_data._fTsumw2 = np.array(data_obs_sumw2).sum()
    th1_data._fTsumwx2 = np.array(data_obs_sumw2 * centers).sum()
    out_file['data_obs'] = th1_data
    out_file.close()
    print(r, data_obs_hist.sum())
