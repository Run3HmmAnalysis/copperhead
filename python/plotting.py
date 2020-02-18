from coffea import hist, util
import mplhep as hep

def load_sample_chunked(path):
    from coffea import util
    print(f'Loading {path}')
    result = util.load(path)
    print(f'Done: {path}')
    return result

class Plotter(object):
    def __init__(self, **kwargs):
        self.map={}
        self.merge_files=True
        self.ggh_name = 'ggh_amcPS'
        self.vbf_name = 'vbf_powhegPS'
        self.weights_by_ds = {}        
        self.__dict__.update(kwargs)
        if 'accumulators' not in kwargs:
            if self.merge_files:
                self.files = self.samples
                self.accumulators = self.processor.accumulator.identity()
            else:
                self.accumulators = {}
            self.load_data()
        self.rates = {r:{} for r in self.regions}
        self.rates_inclusive = {}
        self.hist_dict = {r:{} for r in self.regions}
        self.datasets = {}

            
    def load_sample_notchunked(self,s):
        pass
    
    def load_data(self):
        import glob
        import multiprocessing as mp
        paths = []
        for f in self.files:
            paths = paths + glob.glob(f"{self.path}/{self.prefix}{f}_?.coffea")
        pool = mp.Pool(mp.cpu_count()-2)
        a = [pool.apply_async(load_sample_chunked, args=(p,)) for p in paths]
        for process in a:
            process.wait()
            if self.merge_files:
                self.accumulators = self.accumulators + process.get()
            else:
                self.accumulators[s] = self.accumulators[s] + process.get()
        pool.close()

        
    def load_data_(self):
        from coffea import util
        import glob
        print(f"Loading output from {self.path}")
        for s in self.files:
            print(f'Loading {s}')
            self.accumulators[s] = self.processor.accumulator.identity()
            if self.chunked:
                out_paths = glob.glob(f"{self.path}/{self.prefix}{s}_?.coffea")
                for path in out_paths:
                    if self.merge_files:
                        self.accumulators = self.accumulators+util.load(path)
                    else:
                        self.accumulators[s] = self.accumulators[s]+util.load(path)
 
            else:
                out_path = f"{self.path}/{self.prefix}{s}.coffea"

                if self.merge_files:
                    self.accumulators = self.accumulators+util.load(out_path)
                else:
                    self.accumulators[s] = util.load(out_path)

        
    def make_datamc_comparison(self, do_inclusive, do_exclusive, normalize=True, logy=True, get_rates=False, mc_factor=1, save_to=''):
        self.make_plots(do_inclusive, do_exclusive, do_data_mc=True, normalize=normalize, logy=logy, get_rates=get_rates, mc_factor=mc_factor, save_to=save_to)
        
    def make_shape_comparison(self, do_inclusive, do_exclusive):
        self.make_plots(do_inclusive, do_exclusive, do_shapes=True)

    def make_plots(self, do_inclusive, do_exclusive, do_data_mc=False, do_shapes=False, normalize=True, logy=True,\
                   get_rates=False, mc_factor=1, save_to=''):
        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.cms.style.ROOT)
        from matplotlib import gridspec
        
        nplots_x, nplots_y = self.make_grid(do_inclusive=do_inclusive, do_exclusive=do_exclusive)
        grid = gridspec.GridSpec(nplots_y, nplots_x, hspace = .3) 

        fig = plt.figure()
        plotsize=12
        if do_data_mc:
            ratio_plot_size=0.25
        else:
            ratio_plot_size=0
        fig.set_size_inches(nplots_x*plotsize,nplots_y*plotsize*(1+ratio_plot_size))

        idx = 0
        for var in self.vars:
            for c in self.channels:
                if do_inclusive:
                    if do_data_mc:
                        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = grid[idx], height_ratios=[(1-ratio_plot_size),ratio_plot_size], hspace = .05)
                        self.plot_data_mc(True, c, "", fig, var, gs, self.year, normalize, logy, get_rates, mc_factor, save_to=save_to)
                    else:
                        gs = grid[idx]
                        self.plot_shapes(self.samples, True, c, "", fig, var, gs, self.year)
                    idx += 1
                if do_exclusive:
                    for r in self.regions:
                        if do_data_mc:
                            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = grid[idx], height_ratios=[(1-ratio_plot_size),ratio_plot_size], hspace = .05)
                            self.plot_data_mc(False, c, r, fig, var, gs, self.year, normalize, logy, get_rates, mc_factor, save_to=save_to)
                        else:
                            gs = grid[idx]
                            self.plot_shapes(self.samples, False, c, r, fig, var, gs, self.year)
                        idx += 1
                        
                        
    def make_grid(self, do_inclusive=True, do_exclusive=False, nplots_per_row=4):
        import math

        nplots_x = nplots_per_row # number of plots in one row

        if do_inclusive and do_exclusive:
            nplots_y = math.ceil(len(self.vars)*(len(self.regions)+1)*len(self.channels) / nplots_x) # number of rows
        elif do_inclusive:
            nplots_y = math.ceil(len(self.vars)*len(self.channels) / nplots_x) # number of rows
        elif do_exclusive:
            nplots_y = math.ceil(len(self.vars)*(len(self.regions))*len(self.channels) / nplots_x) # number of rows
        
        return nplots_x, nplots_y



    def plot_data_mc(self, inclusive, channel, region, fig, var, gs, year='2016', normalize=True, logy=True, get_rates=False, mc_factor=1, save_to=''):
        data_sources = {
            'data': ['data_B','data_C','data_D','data_E','data_F','data_G','data_H']  
        }
        bkg_sources_by_region = {
        "z-peak": {
            'DY': ['dy_0j', 'dy_1j', 'dy_2j'],
            'EWK': ['ewk_lljj_mll50_mjj120'],
            'TTbar + Single Top':['ttjets_dl',\
                                  #'ttjets_sl',\
                                  'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
            'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
        },
        
        "h-sidebands" : {
            'DY': [ 'dy_m105_160_amc', 'dy_m105_160_vbf_amc'],
            'EWK': ['ewk_lljj_mll105_160_ptj0'],
            'TTbar + Single Top':['ttjets_dl',\
                                  #'ttjets_sl',\
                                  'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
            'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
        },
        "h-peak": {
            'DY': [ 'dy_m105_160_amc', 'dy_m105_160_vbf_amc'],
            'EWK': ['ewk_lljj_mll105_160_ptj0'],
            'TTbar + Single Top':['ttjets_dl',\
                                  #'ttjets_sl',\
                                  'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
            'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
        }    
        }
        
        all_bkg_sources = {
            'DY': ['dy', 'dy_0j', 'dy_1j', 'dy_2j', 'dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'dy_m105_160_mg', 'dy_m105_160_vbf_mg'],
            'EWK': ['ewk_lljj_mll50_mjj120','ewk_lljj_mll105_160', 'ewk_lljj_mll105_160_ptj0'],
            'TTbar + Single Top':['ttjets_dl', 'ttjets_sl', 'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
            'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
        }
        
        bkg_sources = {}
        if inclusive:
            bkg_sources = all_bkg_sources
            accumulators_copy = self.accumulators[var].sum('region')[:,channel].copy()
        else:
            bkg_sources = bkg_sources_by_region[region]
            accumulators_copy = self.accumulators[var][:,region, channel].copy()
            
        all_bkg = []
        for group, bkgs in bkg_sources.items():
            for b in bkgs:
                all_bkg.append(b)
                if b not in self.weights_by_ds.keys():
                    self.weights_by_ds[b] = 1
                    
        accumulators_copy.scale(self.weights_by_ds, axis='dataset')
            
        if inclusive:
            data = accumulators_copy.sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), data_sources)
            bkg = accumulators_copy.sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), bkg_sources)
            ggh = accumulators_copy[self.ggh_name].sum('channel')
            vbf = accumulators_copy[self.vbf_name].sum('channel')
        else:
            data = accumulators_copy.sum('region').sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), data_sources)
            bkg = accumulators_copy.sum('region').sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), bkg_sources)
            ggh = accumulators_copy[self.ggh_name].sum('region').sum('channel')
            vbf = accumulators_copy[self.vbf_name].sum('region').sum('channel')
            if get_rates:
#                self.datasets = [i[0] for i in list(accumulators_copy.values().keys())]
                self.datasets[region] = all_bkg+data_sources['data']+[self.ggh_name, self.vbf_name]
                for b in self.datasets[region]:
                    if b in  [i[0] for i in list(accumulators_copy.values().keys())]:
                        histogram = accumulators_copy[b].sum('region').sum('channel').sum('dataset').values()[()]*mc_factor
                        sumw2 = accumulators_copy[b].sum('region').sum('channel').sum('dataset')._sumw2[()]*mc_factor
                        self.hist_dict[region][b] = {'hist':histogram, 'sumw2':sumw2}
                        self.rates[region][b] = histogram.sum()
                        
        data_is_valid = data.sum(var).sum('dataset').values()
        bkg_is_valid = bkg.sum(var).sum('dataset').values()
        ggh_is_valid = ggh.sum(var).sum('dataset').values()
        vbf_is_valid = vbf.sum(var).sum('dataset').values()
        
        bkg.axis('dataset').sorting = 'integral' # sort backgrounds by event yields
        
        if normalize and data_is_valid and bkg_is_valid:
            data_int = data.sum(var).sum('dataset').values()[()]
            bkg_int = bkg.sum(var).sum('dataset').values()[()]    
            bkg_sf = data_int/bkg_int
            bkg.scale(bkg_sf)
            print(bkg_sf)
        
        if bkg_is_valid:
            bkg.scale(mc_factor)
#            print(f'Total bkg: {bkg_is_valid[()]}')
        if ggh_is_valid:
            ggh.scale(mc_factor)
#            print(f'Total ggh: {ggh_is_valid[()]}')
        if vbf_is_valid:
            vbf.scale(mc_factor)
#            print(f'Total vbf: {vbf_is_valid[()]}')
#        if data_is_valid:
#            print(f'Total data: {data_is_valid[()]}')
                  

        data_opts = {'color': 'k', 'marker': '.', 'markersize':15}
        stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0)}
        stack_error_opts = {'label':'Stat. unc.','facecolor':(0,0,0,.4), 'hatch':'', 'linewidth': 0}

        if save_to:
            fig.clf()
            plotsize=12
            ratio_plot_size = 0.25
            fig.set_size_inches(plotsize, plotsize*(1+ratio_plot_size))
            gs = fig.add_gridspec(2, 1, height_ratios=[(1-ratio_plot_size),ratio_plot_size], hspace = .05)

        
        # Top panel: Data vs. MC plot

        plt1 = fig.add_subplot(gs[0])
        if bkg_is_valid:
            ax_bkg = hist.plot1d(bkg, ax=plt1, overlay='dataset', overflow='all', stack=True, fill_opts=stack_fill_opts, error_opts=stack_error_opts)
        # draw signal histograms one by one manually because set_prop_cycle didn't work for changing color map
        if ggh_is_valid:
            ax_ggh = hist.plot1d(ggh, overlay='dataset', overflow='all', line_opts={'linewidth':2, 'color':'r'}, error_opts=None)    
        if vbf_is_valid:
            ax_vbf = hist.plot1d(vbf, overlay='dataset', overflow='all', line_opts={'linewidth':2, 'color':'b'}, error_opts=None)    
        if data_is_valid:
            ax_data = hist.plot1d(data, overlay='dataset', overflow='all', line_opts=None, error_opts=data_opts)
            lbl = hep.cms.cmslabel(ax=plt1, data=True, paper=False, year=year)
        else:
            lbl = hep.cms.cmslabel(ax=plt1, data=False, paper=False, year=year)
        
        if logy:
            plt1.set_yscale('log')
            plt1.set_ylim(0.001, 1e9)
#        else:
#            plt1.set_ylim(0., 1e5)
        plt1.set_xlabel('')
        plt1.tick_params(axis='x', labelbottom=False)
        plt1.legend(prop={'size': 'xx-small'})

        # Bottom panel: Data/MC ratio plot
        plt2 = fig.add_subplot(gs[1], sharex=plt1)
    
        if data_is_valid and bkg_is_valid:
            num = data.sum('dataset')
            denom = bkg.sum('dataset')

            ax = hist.plotratio(num=num, ax=plt2,
                        denom=denom,
                        error_opts=data_opts, denom_fill_opts={}, guide_opts={},
                        unc='num')

        plt2.axhline(1, ls='--')
        plt2.set_ylim([0.6,1.4])    
        plt2.set_ylabel('Data/MC')
        lbl = plt2.get_xlabel()
        lbl = lbl if lbl else var
        if inclusive:
            plt2.set_xlabel(f'{lbl}, inclusive, {channel} channel')
        else:
            plt2.set_xlabel(f'{lbl}, {region}, {channel} channel')
        
        if save_to:
            if inclusive:
                fig.savefig(save_to+f"{var}_{channel}_inclusive.png")
            else:
                fig.savefig(save_to+f"{var}_{channel}_{region}.png")
        
    def plot_shapes(self, samples, inclusive, channel, region, fig, var, gs, year='2016'):

        plots = {}

        if self.merge_files:        
            if inclusive:
                accumulators_copy = self.accumulators[var].sum('region')[:,channel].copy()
                for s in samples:
                    plots[s] = accumulators_copy[s].sum('channel')

            else:
                accumulators_copy = self.accumulators[var][:,region, channel].copy()
                for s in samples:
                    plots[s] = accumulators_copy[s].sum('region').sum('channel')

        else:
            accumulators_copy = []
            for src, acc in self.accumulators.items():
                if inclusive:                
                    acc_copy = acc[var].sum('region')[:,channel].copy()
                    for s in samples:
                        plots[f"{src}_{s}"] = acc_copy[s].sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), {src:s})

                else:
                    acc_copy = acc[var][:,region, channel].copy()
                    for s in samples:
                        plots[f"{src}_{s}"] = acc_copy[s].sum('region').sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), {src:s})

        valid = {}            
        for s in plots.keys():
            valid[s] = plots[s].sum(var).sum('dataset').values()
            if valid[s]:
                integral = plots[s].sum(var).sum('dataset').values()[()]
#                print(f"{s}: {integral}")
                if integral:
                    plots[s].scale(1/integral)
            else:
                plots[s].scale(0)

        data_opts = {'color': 'k', 'marker': '.', 'markersize':15}
        stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0)}
        stack_error_opts = {'label':'Stat. unc.','facecolor':(0,0,0,.4), 'hatch':'', 'linewidth': 0}

        plt1 = fig.add_subplot(gs)
        axes = {}
        colors = ['r', 'g', 'b']
        for i, s in enumerate(plots.keys()):
            if valid[s]:
                axes[s] = hist.plot1d(plots[s], overlay='dataset', overflow='all',\
                                      line_opts={'linewidth':2, 'color':colors[i]}, error_opts=None) 
        plt1.set_yscale('log')
        plt1.set_ylim(0.0001, 1)
        lbl = hep.cms.cmslabel(ax=plt1, data=False, paper=False, year=year)
        plt1.set_xlabel(var)
        plt1.legend(prop={'size': 'small'})

