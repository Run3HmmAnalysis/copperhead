import pandas as pd
import numpy as np


class Trainer(object):
    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", pd.DataFrame())
        self.cat_name = kwargs.pop("cat_name", "")
        self.ds_dict = kwargs.pop("ds_dict", {})
        self.features = kwargs.pop("features", [])
        self.models = {}
        self.nfolds = 4

        self.prepare_dataset()
        print()
        print("*" * 60)
        print("In category", self.cat_name)
        print("Event counts in classes:")
        print(self.df["class"].value_counts())
        print("Training features:")
        print(self.features)

    def prepare_dataset(self):
        # Convert dictionary of datasets to a more useful dataframe
        df_info = pd.DataFrame()
        for icls, (cls, ds_list) in enumerate(self.ds_dict.items()):
            for ds in ds_list:
                df_info.loc[ds, "dataset"] = ds
                df_info.loc[ds, "class"] = cls
                df_info.loc[ds, "iclass"] = icls
        df_info["iclass"] = df_info["iclass"].astype(int)
        self.df = self.df[self.df.dataset.isin(df_info.dataset.unique())]

        # Assign numerical classes to each event
        cls_map = dict(df_info[["dataset", "iclass"]].values)
        self.df["class"] = self.df.dataset.map(cls_map)

        # Ignore features that have incorrect values
        # (we handle them by categorizing data by njets)
        ignore_features = self.df.loc[:, self.df.min(axis=0) == -999.0].columns
        features_clean = []
        for c in self.df.columns:
            if (c in self.features) and (c not in ignore_features):
                features_clean.append(c)
        self.features = features_clean

    def add_models(self, model_dict):
        self.models = model_dict

    def run_training(self):
        for i in range(self.nfolds):
            self.train_models(i)

    def train_models(self, ifold):
        if ifold > self.nfolds - 1:
            raise Exception(f"Fold number {ifold} out of range [0 - {self.nfolds-1}]")

        train_folds = [(ifold + f) % self.nfolds for f in [0, 1]]
        val_folds = [(ifold + f) % self.nfolds for f in [2]]
        # eval_folds = [(ifold + f) % self.nfolds for f in [3]]

        print(f"Train DNN #{ifold + 1} out of {self.nfolds}")
        """
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print("Samples used: ", df.dataset.unique())
        """

        train_filter = self.df.event.mod(self.nfolds).isin(train_folds)
        val_filter = self.df.event.mod(self.nfolds).isin(val_folds)
        # eval_filter = df.event.mod(self.nfolds).isin(eval_folds)

        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        df_train = self.df[train_filter]
        df_val = self.df[val_filter]

        x_train = df_train[self.features]
        # y_train = df_train["class"]
        x_val = df_val[self.features]
        # y_val = df_val["class"]

        x_train, x_val = self.scale_data(x_train, x_val, self.features)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]

        for model_name, model in self.models.items():
            model = model(len(self.features), label="test")
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

            """
            model.summary()
            history = model.fit(
                x_train[self.features],
                y_train,
                epochs=100,
                batch_size=1024,
                verbose=1,
                validation_data=(x_val[features_clean], y_val),
                shuffle=True,
            )
            """

    def scale_data(self, x_train, x_val, inputs):
        x_mean = np.mean(x_train[inputs].values, axis=0)
        x_std = np.std(x_train[inputs].values, axis=0)
        training_data = (x_train[inputs] - x_mean) / x_std
        validation_data = (x_val[inputs] - x_mean) / x_std
        # np.save(f"output/trained_models/{model}/scalers_{label}", [x_mean, x_std])
        return training_data, validation_data
