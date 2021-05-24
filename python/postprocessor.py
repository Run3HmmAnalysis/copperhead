import os
import sys
from functools import partial

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from config.variables import variables_lookup, Variable
import boost_histogram as bh

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras  # noqa: E402
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__all__ = ['keras']

training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_pt_log',
                     'dimuon_eta', 'dimuon_mass_res',
                     'dimuon_mass_res_rel', 'dimuon_cos_theta_cs',
                     'dimuon_phi_cs', 'jet1_pt', 'jet1_eta', 'jet1_phi',
                     'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi',
                     'jet2_qgl', 'jj_mass', 'jj_mass_log', 'jj_dEta',
                     'rpt', 'll_zstar_log', 'mmj_min_dEta', 'nsoftjets5',
                     'htsoft2']

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
    # 'dy_m105_160_amc': 'DY_nofilter',
    # 'dy_m105_160_vbf_amc': 'DY_filter',
    'dy_m105_160_amc': 'DY',
    'dy_m105_160_vbf_amc': 'DY',
    'ewk_lljj_mll105_160_ptj0': 'EWK',
    # 'ewk_lljj_mll105_160_py_dipole': 'EWK_Pythia',
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

grouping_alt = {
    'Data': ['data_A', 'data_B', 'data_C', 'data_D', 'data_E',
             'data_F', 'data_G', 'data_H'],
    'DY': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc'],
    'EWK': ['ewk_lljj_mll105_160_ptj0'],
    'TT+ST': ['ttjets_dl', 'ttjets_sl', 'ttw', 'ttz',
              'st_tw_top', 'st_tw_antitop'],
    'VV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'zz'],
    'VVV': ['www', 'wwz', 'wzz', 'zzz'],
    'ggH': ['ggh_amcPS'],
    'VBF': ['vbf_powheg_dipole']
}

decorrelation_scheme = {
    'LHERen': {'DY': ['DY'], 'EWK': ['EWK'], 'ggH': ['ggH'],
               'TT+ST': ['TT+ST']},
    'LHEFac': {'DY': ['DY'], 'EWK': ['EWK'], 'ggH': ['ggH'],
               'TT+ST': ['TT+ST']},
    'pdf_2rms': {'DY': ['DY'], 'ggH': ['ggH'], 'VBF': ['VBF']},
    'pdf_mcreplica': {'DY': ['DY'], 'ggH': ['ggH'], 'VBF': ['VBF']},
    # 'LHERen': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'],
    #            'ggH':['ggH'], 'TT+ST':['TT+ST']},
    # 'LHEFac': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'],
    #            'ggH':['ggH'], 'TT+ST':['TT+ST']},
    # 'pdf_2rms': {'DY':['DY_filter', 'DY_nofilter'],
    #             'ggH':['ggH'], 'VBF':['VBF']},
    # 'pdf_mcreplica': {'DY':['DY_filter', 'DY_nofilter'],
    #                  'ggH':['ggH'], 'VBF':['VBF']},
}


def workflow(client, paths, parameters, timer):
    # Load dataframes
    df_future = client.map(load_data, paths)
    df_future = client.gather(df_future)
    timer.add_checkpoint("Loaded data from Parquet")

    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    npart = df.npartitions
    df = df.compute()
    df.reset_index(inplace=True, drop=True)
    df = dd.from_pandas(df, npartitions=npart)
    df = df.repartition(npartitions=parameters['ncpus'])
    timer.add_checkpoint("Combined into a single Dask DataFrame")

    keep_columns = ['s', 'year', 'r']
    keep_columns += [f'c {v}' for v in parameters['syst_variations']]
    keep_columns += [c for c in df.columns if 'wgt_' in c]
    keep_columns += parameters['hist_vars']

    # Evaluate classifiers
    # TODO: outsource to GPUs
    evaluate_mva = True
    if evaluate_mva:
        for v in parameters['syst_variations']:
            for model in parameters['dnn_models']:
                score_name = f'score_{model} {v}'
                keep_columns += [score_name]
                df[score_name] = df.map_partitions(
                    dnn_evaluation, v, model, parameters,
                    meta=(score_name, float)
                )
                timer.add_checkpoint(f"Evaluated {model} {v}")
            for model in parameters['bdt_models']:
                score_name = f'score_{model} {v}'
                keep_columns += [score_name]
                df[score_name] = df.map_partitions(
                    bdt_evaluation, v, model, parameters,
                    meta=(score_name, float)
                )
                timer.add_checkpoint(f"Evaluated {model} {v}")
    df = df[[c for c in keep_columns if c in df.columns]]

    df = df.compute()
    df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True)
    timer.add_checkpoint("Prepared for histogramming")

    # Make histograms
    hist_futures = client.map(
        partial(histogram, df=df, parameters=parameters),
        parameters['hist_vars']
    )
    hists_ = client.gather(hist_futures)
    hists = {}
    for h in hists_:
        hists.update(h)
    timer.add_checkpoint("Histogramming")

    # Plot histograms
    hists_to_plot = [hist for var, hist in hists.items()
                     if var in parameters['plot_vars']]
    plot_futures = client.map(
        partial(plot, df=df, parameters=parameters),
        hists_to_plot
    )

    client.gather(plot_futures)

    return df, hists


def load_data(path):
    if len(path) > 0:
        df = dd.read_parquet(path)
    else:
        df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    return df


def prepare_features(df, parameters, variation='nominal', add_year=True):
    global training_features
    if add_year:
        features = training_features + ['year']
    else:
        features = training_features
    features_var = []
    for trf in features:
        if f'{trf} {variation}' in df.columns:
            features_var.append(f'{trf} {variation}')
        elif trf in df.columns:
            features_var.append(trf)
        else:
            print(f'Variable {trf} not found in training dataframe!')
    return features_var


def dnn_evaluation(df, variation, model, parameters):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        device_count={'CPU': 1}
    )
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    sess = tf.compat.v1.Session(config=config)
    if parameters['do_massscan']:
        mass_shift = parameters['mass'] - 125.0
    features = prepare_features(df, parameters, variation, add_year=True)
    score_name = f'score_{model} {variation}'
    try:
        df = df.compute()
    except Exception:
        pass
    df[score_name] = 0
    with sess:
        nfolds = 4
        for i in range(nfolds):
            # FIXME
            label = f"allyears_jul7_{i}"

            # train_folds = [(i + f) % nfolds for f in [0, 1]]
            # val_folds = [(i + f) % nfolds for f in [2]]
            eval_folds = [(i + f) % nfolds for f in [3]]

            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            scalers_path =\
                f"{parameters['models_path']}/{model}/scalers_{label}.npy"
            scalers = np.load(scalers_path)
            model_path =\
                f"{parameters['models_path']}/{model}/dnn_{label}.h5"
            dnn_model = load_model(model_path)
            df_i = df.loc[eval_filter, :]
            df_i.loc[df_i.r != 'h-peak', 'dimuon_mass'] = 125.0
            if parameters['do_massscan']:
                df_i['dimuon_mass'] = df_i['dimuon_mass'] - mass_shift
            df_i = (df_i[features] - scalers[0]) / scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df[score_name]


def bdt_evaluation(df, variation, model, parameters):
    import pickle
    if parameters['do_massscan']:
        mass_shift = parameters['mass'] - 125.0
    features = prepare_features(df, parameters, variation, add_year=False)
    score_name = f'score_{model} {variation}'
    try:
        df = df.compute()
    except Exception:
        pass
    df[score_name] = 0
    nfolds = 4
    for i in range(nfolds):
        # FIXME
        label = f"2016_jul7_{i}"

        # train_folds = [(i + f) % nfolds for f in [0, 1]]
        # val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        scalers_path =\
            f"{parameters['models_path']}/{model}/scalers_{label}.npy"
        scalers = np.load(scalers_path)
        model_path =\
            f"{parameters['models_path']}/{model}/"\
            f"BDT_model_earlystop50_{label}.pkl"

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        df_i.loc[df_i.r != 'h-peak', 'dimuon_mass'] = 125.0
        if parameters['do_massscan']:
            df_i['dimuon_mass'] = df_i['dimuon_mass'] - mass_shift
        df_i = (df_i[features] - scalers[0]) / scalers[1]
        if len(df_i) > 0:
            if 'multiclass' in model:
                prediction = np.array(
                    bdt_model.predict_proba(df_i.values)[:, 5]).ravel()
            else:
                prediction = np.array(
                    bdt_model.predict_proba(df_i.values)[:, 1]).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df[score_name]


def histogram(var, df=pd.DataFrame(), parameters={}):
    if var in variables_lookup.keys():
        var = variables_lookup[var]
    else:
        var = Variable(var, var, 50, 0, 5)

    samples = df.s.unique()
    years = df.year.unique()
    regions = parameters['regions']
    categories = parameters['categories']
    syst_variations = parameters['syst_variations']
    wgt_variations = [w for w in df.columns if ('wgt_' in w)]

    regions = [r for r in regions if r in df.r.unique()]
    categories = [c for c in categories if c in df['c nominal'].unique()]

    # sometimes different years have different binnings (MVA score)
    h = {}

    for year in years:
        if ('score' in var.name):
            bins = parameters['mva_bins'][
                var.name.replace('score_', '')][f'{year}']
            h[year] = (
                Hist.new
                .StrCat(samples, name="dataset")
                .StrCat(regions, name="region")
                .StrCat(categories, name="category")
                .StrCat(syst_variations, name="variation")
                .StrCat(['value', 'sumw2'], name='val_err')
                .Var(bins, name=var.name)
                .Double()
            )
            # nbins = len(bins) - 1
        else:
            h[year] = (
                Hist.new
                .StrCat(samples, name="dataset")
                .StrCat(regions, name="region")
                .StrCat(categories, name="category")
                .StrCat(syst_variations, name="variation")
                .StrCat(['value', 'sumw2'], name='val_sumw2')
                .Reg(var.nbins, var.xmin, var.xmax,
                     name=var.name, label=var.caption)
                .Double()
            )
            # nbins = var.nbins

        for s in samples:
            for r in regions:
                for v in syst_variations:
                    varname = f'{var.name} {v}'
                    if varname not in df.columns:
                        if var.name in df.columns:
                            varname = var.name
                        else:
                            continue
                    for c in categories:
                        for w in wgt_variations:
                            slicer = ((df.s == s) &
                                      (df.r == r) &
                                      (df.year == year) &
                                      (df[f'c {v}'] == c))
                            data = df.loc[slicer, varname]
                            weight = df.loc[slicer, w]
                            h[year].fill(s, r, c, v, 'value',
                                         data, weight=weight)
                            h[year].fill(s, r, c, v, 'sumw2',
                                         data, weight=weight * weight)
                            # TODO: add treatment of PDF systematics
                            # (MC replicas)
    return {var.name: h}


def plot(hist, df=pd.DataFrame(), parameters={}):
    if not hist.keys():
        return
    a_year = list(hist.keys())[0]
    var = hist[a_year].axes[-1]

    r = 'h-peak'
    c = 'vbf'
    v = 'nominal'
    # stack_groups = ['DY', 'EWK', 'TT+ST', 'VV', 'VVV']
    step_groups = ['VBF', 'ggH']
    # errorbar_groups = ['Data']

    plotsize = 8
    ratio_plot_size = 0.25
    # data_opts = {'color': 'k', 'marker': '.', 'markersize': 15}
    # stack_fill_opts = {'alpha': 0.8, 'edgecolor': (0, 0, 0)}
    # stat_err_opts = {'step': 'post', 'label': 'Stat. unc.',
    #                  'hatch': '//////', 'facecolor': 'none',
    #                  'edgecolor': (0, 0, 0, .5), 'linewidth': 0}
    # ratio_err_opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3),
    #                   'linewidth': 0}

    fig = plt.figure()
    style = hep.style.CMS
    style['mathtext.fontset'] = 'cm'
    style['mathtext.default'] = 'rm'
    plt.style.use(style)
    for year in hist.keys():
        fig.clf()
        fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
        gs = fig.add_gridspec(2, 1, height_ratios=[
            (1 - ratio_plot_size), ratio_plot_size], hspace=.07)

        # Top panel: Data/MC
        plt1 = fig.add_subplot(gs[0])
        # datasets = hist[year].axes["dataset"]
        # stack = []
        # stack_labels = []
        # for group, entries in grouping_alt.items():
        #    if group in step_groups:
        #        for entry in entries:
        #            if entry not in datasets:
        #                continue
        #            plottable = hist[year][
        #                entry, r, c, v, 'value', :
        #            ].project(var.name)
        #            hep.histplot(plottable, label=group, **{'linewidth': 3})

        #    elif group in stack_groups:
        #        entries_ = [e for e in entries if e in datasets]
        #        if len(entries_) == 0:
        #            continue
        #        plottable = hist[year][
        #            :, r, c, v, 'value', :
        #        ].project('dataset', var.name)
        #        print(plottable.view())

        # stack_entries = [e for e, g in grouping.items() if g in stack_groups]
        step_entries = [e for e, g in grouping.items() if g in step_groups]
        # stack = hist[year][
        #             bh.loc(stack_entries), r, c, v, 'value', :
        #         ].project('dataset', var.name)
        # hep.histplot(
        #     stack, stack=True, histtype='fill',
        #     label=stack_labels, **stack_fill_opts
        # )

        steps = hist[year][
            bh.loc(step_entries), r, c, v, 'value', :
        ].project(var.name)
        hep.histplot(
            steps, stack=False, histtype='step', **{'linewidth': 3}
        )

        plt1.set_yscale('log')
        plt1.set_ylim(0.01, 1e9)
        # plt1.set_xlim(var.xmin,var.xmax)
        # plt1.set_xlim(edges[0], edges[-1])
        plt1.set_xlabel('')
        plt1.tick_params(axis='x', labelbottom=False)
        plt1.legend(prop={'size': 'small'})

        # Bottom panel: Data/MC ratio plot
        plt2 = fig.add_subplot(gs[1], sharex=plt1)

        plt2.axhline(1, ls='--')
        plt2.set_ylim([0.5, 1.5])
        plt2.set_ylabel('Data/MC', loc='center')
        plt2.set_xlabel(var.label, loc='right')

        hep.cms.label(ax=plt1, data=True, paper=False, year=year)

        path = parameters['plots_path']
        out_name = f'{path}/{var.name}_{year}.png'
        fig.savefig(out_name)
        print(f'Saved: {out_name}')
    return
