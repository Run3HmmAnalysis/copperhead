import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep

import sys

[sys.path.append(i) for i in [".", ".."]]

from python.workflow import parallelize
from python.io import mkdir

style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)


class Trainer(object):
    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", pd.DataFrame())
        self.cat_name = kwargs.pop("cat_name", "")
        self.ds_dict = kwargs.pop("ds_dict", {})
        self.features = kwargs.pop("features", [])
        self.plot_path = kwargs.pop("plot_path", "./")
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

    def run_training(self, client=None):
        fold_filters = []
        for ifold in range(self.nfolds):
            i_fold_filters = {}
            train_folds = [(ifold + f) % self.nfolds for f in [0, 1]]
            val_folds = [(ifold + f) % self.nfolds for f in [2]]
            eval_folds = [(ifold + f) % self.nfolds for f in [3]]
            i_fold_filters["train_filter"] = self.df.event.mod(self.nfolds).isin(
                train_folds
            )
            i_fold_filters["val_filter"] = self.df.event.mod(self.nfolds).isin(
                val_folds
            )
            i_fold_filters["eval_filter"] = self.df.event.mod(self.nfolds).isin(
                eval_folds
            )
            i_fold_filters["ifold"] = ifold
            fold_filters.append(i_fold_filters)
        arg_set = {"model_name": self.models.keys(), "fold_filters": fold_filters}
        if client:
            rets = parallelize(self.train_models, arg_set, client)
            for ret in rets:
                self.df.loc[ret["filter"], ret["score_name"]] = ret["prediction"]
        else:
            for model_name in self.models.keys():
                for ff in fold_filters:
                    ret = self.train_models(
                        {"fold_filters": ff, "model_name": model_name}
                    )
                    self.df.loc[ret["filter"], ret["score_name"]] = ret["prediction"]

    def train_models(self, args, parameters={}):
        fold_filters = args["fold_filters"]
        model_name = args["model_name"]
        df = self.df
        ifold = fold_filters["ifold"]
        print(f"Training model {model_name}, fold #{ifold}...")

        train_filter = fold_filters["train_filter"]
        val_filter = fold_filters["val_filter"]
        eval_filter = fold_filters["eval_filter"]

        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        df_train = df[train_filter]
        df_val = df[val_filter]
        df_eval = df[eval_filter]

        x_train = df_train[self.features]
        y_train = df_train["class"]
        x_val = df_val[self.features]
        y_val = df_val["class"]
        x_eval = df_eval[self.features]
        # y_eval = df_eval["class"]

        x_train, x_val, x_eval = self.scale_data(x_train, x_val, x_eval, self.features)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]
        x_eval[other_columns] = df_eval[other_columns]

        model = self.models[model_name]
        model = model(len(self.features), label="test")
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        score_name = f"{model_name}_score"

        # model.summary()

        # Train
        history = model.fit(
            x_train[self.features],
            y_train,
            epochs=100,
            batch_size=1024,
            verbose=0,
            validation_data=(x_val[self.features], y_val),
            shuffle=True,
        )
        self.plot_history(history, model_name, ifold)
        # Evaluate instantly
        prediction = np.array(model.predict(x_eval[self.features])).ravel()
        ret = {
            "filter": eval_filter,
            "score_name": score_name,
            "prediction": prediction,
        }
        return ret

    def scale_data(self, x_train, x_val, x_eval, inputs):
        x_mean = np.mean(x_train[inputs].values, axis=0)
        x_std = np.std(x_train[inputs].values, axis=0)
        training_data = (x_train[inputs] - x_mean) / x_std
        validation_data = (x_val[inputs] - x_mean) / x_std
        evaluation_data = (x_eval[inputs] - x_mean) / x_std
        # np.save(f"output/trained_models/{model}/scalers_{label}", [x_mean, x_std])
        return training_data, validation_data, evaluation_data

    def plot_history(self, history, model_name, ifold):
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"])
        ax.plot(history.history["val_loss"])
        ax.set_title(f"Loss of {model_name}, fold #{ifold}")
        ax.set_ylabel("Loss")
        ax.set_xlabel("epoch")
        ax.legend(["Training", "Validation"], loc="best")
        out_path = f"{self.plot_path}/losses/"
        mkdir(out_path)
        out_name = f"{out_path}/loss_{model_name}_{ifold}.png"
        fig.savefig(out_name)

    def plot_roc_curves(self):
        roc_curves = {}
        fig = plt.figure()
        fig, ax = plt.subplots()
        for model_name, model in self.models.items():
            score_name = f"{model_name}_score"
            roc_curves[score_name] = roc_curve(
                y_true=self.df["class"],
                y_score=self.df[score_name],
                sample_weight=self.df["lumi_wgt"] * self.df["mc_wgt"],
            )
            ax.plot(
                roc_curves[score_name][0], roc_curves[score_name][1], label=score_name
            )
        ax.legend(prop={"size": "x-small"})
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        out_name = f"{self.plot_path}/rocs.png"
        fig.savefig(out_name)
