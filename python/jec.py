from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
from config.jec_parameters import jec_weight_sets, jec_names_and_sources, runs


def get_name_map(stack):
    name_map = stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'
    return name_map


def jec_factories(year):

    weight_sets = jec_weight_sets(year)
    names = jec_names_and_sources(year)
    
    jec_factories = {}
    jec_factories_data = {}
    
    # Prepare evaluators for JEC, JER and their systematics
    jetext = extractor()
    jetext.add_weight_sets(weight_sets['jec_weight_sets'])
    jetext.add_weight_sets(weight_sets['jec_weight_sets_data'])
    jetext.finalize()
    jet_evaluator = jetext.make_evaluator()

    stacks_def = {
        'jec_stack': ['jec_names'],
        'jer_stack': ['jer_names', 'jersf_names'],
        'junc_stack': ['junc_names']
    }

    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])

    jec_input_options = {}
    for opt in ['jec', 'junc', 'jer']:
        jec_input_options[opt] = {
            name: jet_evaluator[name]
            for name in stacks[f'{opt}_stack']
        }        

    for src in names['junc_sources']:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options['junc'][key] = jet_evaluator[key]

    # Create separate factories for JEC, JER, JEC variations
    for opt in ['jec', 'junc', 'jer']:
        stack = JECStack(jec_input_options[opt])
        jec_factories[opt] = CorrectedJetsFactory(
            get_name_map(stack), stack
        )

    # Create a separate factory for each data run
    for run in runs[year]:
        jec_inputs_data = {}
        for opt in ['jec', 'junc']:
            jec_inputs_data.update({
                name: jet_evaluator[name] for name
                in names[f'{opt}_names_data'][run]
            })
        for src in names['junc_sources_data'][run]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]

        jec_stack_data = JECStack(jec_inputs_data)
        jec_factories_data[run] = CorrectedJetsFactory(
            get_name_map(jec_stack_data), jec_stack_data
        )
    
    return jec_factories, jec_factories_data
