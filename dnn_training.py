def extract_unbinned_data(path, chunked, prefix, classes, channels, regions, to_plot):

    from config.variables import variables as var
    variables = [v.name for v in var]
    import pandas as pd
    from coffea import util
    import glob
    
    dfs = {}
    cls_idx = 0
    import json
    for cls, smp_list in classes.items():
        for s in smp_list:
            print(f"Adding {s}")
            if chunked:
                proc_outs = []
                paths = glob.glob(f"{path}/unbinned/{prefix}{s}_?.coffea")
                for p in paths:
                    print(p)
                    proc_outs.append(util.load(p))

            else:
                proc_outs = [util.load(f"{path}/{prefix}{s}.coffea")]
            for ip, proc_out in enumerate(proc_outs):
                for r in regions:
                    for c in channels:
                        lbl = f'{c}_channel_{s}_{r}_{ip}'
                        dfs[lbl] = pd.DataFrame(columns=variables+['class','class_idx'])
                        for v in variables:
#                            print(v, proc_out[f'{v}_{c}_{r}'].value)
                            try:
                                dfs[lbl][v] = proc_out[f'{v}_{c}_{r}'].value
                            except:
#                                 print("ignoring: ", s,r,c,v)
                                 continue

                        dfs[lbl]['event_weight'] = proc_out[f'wgt_nominal_{c}_{r}'].value
                        dfs[lbl]['to_plot'] = 1 if (s in to_plot) else 0
                        dfs[lbl]['class'] = cls
                        dfs[lbl]['class_idx'] = cls_idx
                        print(dfs[lbl])
        cls_idx += 1
    return dfs

def scale_data(inputs, label):
    x_mean = np.mean(x_train[inputs].values,axis=0)
    x_std = np.std(x_train[inputs].values,axis=0)
    training_data = (x_train[inputs]-x_mean)/x_std
    testing_data = (x_test[inputs]-x_mean)/x_std
    np.save(f"output/trained_models/scalers_{label}", [x_mean, x_std])
    return training_data, testing_data


inputs_binary_m125 = {
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],#, 'ttjets_dl'],
    'signal': ['ggh_amcPS','vbf_powhegPS', 'vbf_powheg_herwig'],
    #'signal': ['ggh_amcPS', 'ggh_powhegPS', 'vbf_amcPS', 'vbf_powhegPS'],
}

to_plot = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0', 'ttjets_dl',\
           'ggh_amcPS','ggh_amcPS_m120','ggh_amcPS_m130',\
           'vbf_powheg','vbf_powheg_m120','vbf_powheg_m130']


other_columns = ['event', 'event_weight', 'to_plot']

from coffea import util
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config.parameters import training_features

year = '2018'
load_path = f'/depot/cms/hmm/coffea/all_{year}_may11/'
#load_path = '/home/dkondra/all_2016_apr20/'

classes = inputs_binary_m125
label = year

# TODO: parallelize loading

# df_dict = extract_unbinned_data(load_path, True, '', classes, ['vbf'], ['h-sidebands', 'h-peak'], to_plot)
df_dict = extract_unbinned_data(load_path, True, '', classes, ['vbf', 'vbf_01j','vbf_2j'], ['h-peak'], to_plot)
df = pd.DataFrame()
df = pd.concat(df_dict)
df = df.sample(frac=1)

train_fraction = 0.6
df_train = df[df['event']%(1/train_fraction)<1]
df_test = df[df['event']%(1/train_fraction)>=1]
# Check that fraction is correct
print("Training events fraction:", df_train.shape[0]/(df_train.shape[0]+df_test.shape[0]))

x_train = df_train[training_features]
y_train = df_train['class_idx']
x_test = df_test[training_features]
y_test = df_test['class_idx']

for i in range(len(classes)):
    cls_name = list(classes.keys())[i]
    train_evts = len(y_train[y_train==i])
    print(f"{train_evts} training events in class {cls_name}")
    
# scale data
x_train, x_test = scale_data(training_features, label)
x_train[other_columns] = df_train[other_columns]
x_test[other_columns] = df_test[other_columns]


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
from tensorflow.keras import backend as K

# load model
input_dim = len(training_features)
# label = 'test'
inputs = Input(shape=(input_dim,), name = label+'_input')
x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()

history = model.fit(x_train[training_features], y_train, epochs=200, batch_size=2048, verbose=1,
                                    validation_split=0.2, shuffle=True)

model.save(f'output/trained_models/test_{label}_hw.h5')
