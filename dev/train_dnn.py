import numpy as np
from dnn_models import run2_model_purdue


def train_dnn(df, cat_name, df_info, features):
    print()
    print("*" * 60)
    print("In category", cat_name)
    print("Event counts in classes:")
    print(df["class"].value_counts())
    ignore_features = df.loc[:, df.min(axis=0) == -999.0].columns
    features_clean = []
    for c in df.columns:
        if (c in features) and (c not in ignore_features):
            features_clean.append(c)
    print("Training features:")
    print(features_clean)

    nfolds = 4
    for i in range(nfolds):
        train_folds = [(i + f) % nfolds for f in [0, 1]]
        val_folds = [(i + f) % nfolds for f in [2]]
        # eval_folds = [(i + f) % nfolds for f in [3]]

        print(f"Train DNN #{i + 1} out of {nfolds}")
        """
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print("Samples used: ", df.dataset.unique())
        """

        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        # eval_filter = df.event.mod(nfolds).isin(eval_folds)

        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        df_train = df[train_filter]
        df_val = df[val_filter]

        x_train = df_train[features_clean]
        # y_train = df_train["class"]
        x_val = df_val[features_clean]
        # y_val = df_val["class"]

        x_train, x_val = scale_data(x_train, x_val, features_clean)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]

        dnn = run2_model_purdue(len(features_clean), label="test")
        dnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        # dnn.summary()

        """
        history = dnn.fit(
            x_train[features_clean],
            y_train,
            epochs=100,
            batch_size=1024,
            verbose=1,
            validation_data=(x_val[features_clean], y_val),
            shuffle=True,
        )
        """

    return


def scale_data(x_train, x_val, inputs):
    x_mean = np.mean(x_train[inputs].values, axis=0)
    x_std = np.std(x_train[inputs].values, axis=0)
    training_data = (x_train[inputs] - x_mean) / x_std
    validation_data = (x_val[inputs] - x_mean) / x_std
    # np.save(f"output/trained_models/{model}/scalers_{label}", [x_mean, x_std])
    return training_data, validation_data
