runs = {
    '2016': ['B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '2017': ['B', 'C', 'D', 'E', 'F'],
    '2018': ['A', 'B', 'C', 'D']
}

jec_levels_mc = ['L1FastJet', 'L2Relative', 'L3Absolute']
jec_levels_data = ['L1FastJet', 'L2Relative',
                   'L3Absolute', 'L2L3Residual']

jec_tags = {
    '2016': 'Summer16_07Aug2017_V11_MC',
    '2017': 'Fall17_17Nov2017_V32_MC',
    '2018': 'Autumn18_V19_MC'
}

jer_tags = {
    '2016': 'Summer16_25nsV1_MC',
    '2017': 'Fall17_V3_MC',
    '2018': 'Autumn18_V7_MC'
}

jec_data_tags = {
    '2016': {
        'Summer16_07Aug2017BCD_V11_DATA': ['B', 'C', 'D'],
        'Summer16_07Aug2017EF_V11_DATA' : ['E', 'F'],
        'Summer16_07Aug2017GH_V11_DATA' : ['G', 'H'],
    },
    '2017': {
        'Fall17_17Nov2017B_V32_DATA': ['B'],
        'Fall17_17Nov2017C_V32_DATA': ['C'],
        'Fall17_17Nov2017DE_V32_DATA': ['D', 'E'],
        'Fall17_17Nov2017F_V32_DATA': ['F']
    },
    '2018': {
        'Autumn18_RunA_V19_DATA': ['A'],
        'Autumn18_RunB_V19_DATA': ['B'],
        'Autumn18_RunC_V19_DATA': ['C'],
        'Autumn18_RunD_V19_DATA': ['D']
    }
    
}

def jec_names_and_sources(year):
    names = {}
    suffix = {
        'jec_names': [f'_{level}_AK4PFchs' for level in jec_levels_mc],
        'jec_names_data': [f'_{level}_AK4PFchs' for level in jec_levels_data],
        'junc_names': ['_Uncertainty_AK4PFchs'],
        'junc_names_data': ['_Uncertainty_AK4PFchs'],
        'junc_sources': ['_UncertaintySources_AK4PFchs'],
        'junc_sources_data': ['_UncertaintySources_AK4PFchs'],
        'jer_names': ['_PtResolution_AK4PFchs'],
        'jersf_names': ['_SF_AK4PFchs']
    }

    for key, suff in suffix.items():
        if 'data' in key:
            names[key] = {}
            for run in runs[year]:
                for tag, iruns in jec_data_tags[year].items():
                    if run in iruns:
                        names[key].update({
                            run: [f"{tag}{s}"for s in suff]
                        })
        else:
            tag = jer_tags[year] if 'jer' in key else jec_tags[year]
            names[key] = [
                f"{tag}{s}" for s in suff
            ]

    return names


def jec_weight_sets(year):
    weight_sets = {}
    names = jec_names_and_sources(year)

    extensions = {
        'jec_names': 'jec',
        'jer_names': 'jr',
        'jersf_names': 'jersf',
        'junc_names': 'junc',
        'junc_sources': 'junc',
    }
    
    weight_sets['jec_weight_sets'] = []
    weight_sets['jec_weight_sets_data'] = []

    for opt, ext in extensions.items():
        # MC
        weight_sets['jec_weight_sets'].extend(
            [f"* * data/jec/{name}.{ext}.txt"
             for name in names[opt]]
        )
        # Data
        if 'jer' in opt:
            continue
        data = []
        for run, items in names[f'{opt}_data'].items():
            data.extend(items)
        data = list(set(data))
        weight_sets['jec_weight_sets_data'].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in data]
        )

    return weight_sets

