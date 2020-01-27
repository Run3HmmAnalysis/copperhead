def plot_variable(output, inclusive, channel, region, fig, var, gs):
    
    data_sources = {
        'data': ['data_B','data_C','data_D','data_E','data_F','data_G','data_H']  
    }
    bkg_sources = {
        'DY': ['dy', 'dy_0j', 'dy_1j', 'dy_2j', 'dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'dy_m105_160_mg', 'dy_m105_160_vbf_mg'],
        'EWK': ['ewk_lljj_mll50_mjj120','ewk_lljj_mll105_160'],
        'TTbar + Single Top':['ttjets_dl', 'ttjets_sl', 'ttw', 'ttz', 'st_tw_top', 'st_tw_antitop'],
        'VV + VVV': ['ww_2l2nu', 'wz_2l2q', 'wz_1l1nu2q', 'wz_3lnu', 'www','wwz','wzz','zzz'],
    }

    if inclusive:
        output_copy = output[var].sum('region')[:,channel].copy()
#         output_copy.scale(weights, axis='dataset')
        data = output_copy.sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), data_sources)
        bkg = output_copy.sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), bkg_sources)
        ggh = output_copy['ggh_amcPS'].sum('channel')
        vbf = output_copy['vbf_amcPS'].sum('channel')
    else:
        output_copy = output[var][:,region, channel].copy()
#         output_copy.scale(weights, axis='dataset')
        data = output_copy.sum('region').sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), data_sources)
        bkg = output_copy.sum('region').sum('channel').group('dataset', hist.Cat("dataset", "Dataset"), bkg_sources)
        ggh = output_copy['ggh_amcPS'].sum('region').sum('channel')
        vbf = output_copy['vbf_amcPS'].sum('region').sum('channel')
        
    data_is_valid = data.sum(var).sum('dataset').values()
    bkg_is_valid = bkg.sum(var).sum('dataset').values()
    ggh_is_valid = ggh.sum(var).sum('dataset').values()
    vbf_is_valid = vbf.sum(var).sum('dataset').values()
        
    bkg.axis('dataset').sorting = 'integral' # sort backgrounds by event yields
        
    scale_mc_to_data = True
    if scale_mc_to_data and data_is_valid and bkg_is_valid:
        data_int = data.sum(var).sum('dataset').values()[()]
        bkg_int = bkg.sum(var).sum('dataset').values()[()]    
        bkg_sf = data_int/bkg_int
        bkg.scale(bkg_sf)
        
    data_opts = {'color': 'k', 'marker': '.', 'markersize':15}
    stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0)}
    stack_error_opts = {'label':'Stat. unc.','facecolor':(0,0,0,.4), 'hatch':'', 'linewidth': 0}
    
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
    plt1.set_yscale('log')
    plt1.set_ylim(0.001, 1e9)
    lbl = hep.cms.cmslabel(plt1, data=True, paper=False, year='2016')
    plt1.set_xlabel('')
    plt1.tick_params(axis='x', labelbottom=False)
    plt1.legend(prop={'size': 'xx-small'})
    
    # Bottom panel: Data/MC ratio plot
    plt2 = fig.add_subplot(gs[1], sharex=plt1)
    if data_is_valid and bkg_is_valid:
        num = data.sum('dataset')
        denom = bkg.sum('dataset')
        hist.plotratio(num=num, ax=plt2,
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