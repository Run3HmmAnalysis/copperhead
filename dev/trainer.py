import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
[sys.path.append(i) for i in [".", ".."]]

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep
from tensorflow.keras.models import load_model

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
        self.out_path = kwargs.pop("out_path", "./")
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

        self.fold_filters_list = []
        for ifold in range(self.nfolds):
            i_fold_filters = {}
            train_folds = [(ifold + f) % self.nfolds for f in [0, 1]]
            val_folds = [(ifold + f) % self.nfolds for f in [2]]
            eval_folds = [(ifold + f) % self.nfolds for f in [3]]
            i_fold_filters["ifold"] = ifold
            i_fold_filters["train_filter"] = self.df.event.mod(self.nfolds).isin(
                train_folds
            )
            i_fold_filters["val_filter"] = self.df.event.mod(self.nfolds).isin(
                val_folds
            )
            i_fold_filters["eval_filter"] = self.df.event.mod(self.nfolds).isin(
                eval_folds
            )
            self.fold_filters_list.append(i_fold_filters)

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
        self.trained_models = {n: {} for n in self.models.keys()}
        self.scalers = {n: {} for n in self.models.keys()}

    def run_training(self, client=None):
        if client:
            arg_set = {
                "model_name": self.models.keys(),
                "fold_filters": self.fold_filters_list,
            }
            rets = parallelize(self.train_model_ifold, arg_set, client)

        else:
            rets = []
            for model_name in self.models.keys():
                for ff in self.fold_filters_list:
                    ret = self.train_model_ifold(
                        {"fold_filters": ff, "model_name": model_name}
                    )
                    rets.append(ret)
        for ret in rets:
            model_name = ret["model_name"]
            ifold = ret["ifold"]
            self.trained_models[model_name][ifold] = ret["model_save_path"]
            self.scalers[model_name][ifold] = ret["scalers_save_path"]

    def run_evaluation(self, client=None):
        if client:
            arg_set = {
                "model_name": self.models.keys(),
                "fold_filters": self.fold_filters_list,
            }
            rets = parallelize(self.evaluate_model_ifold, arg_set, client)
        else:
            rets = []
            for model_name in self.models.keys():
                for ff in self.fold_filters_list:
                    ret = self.evaluate_model_ifold(
                        {"fold_filters": ff, "model_name": model_name}
                    )
                    rets.append(ret)
        for ret in rets:
            ifold = ret["ifold"]
            model_name = ret["model_name"]
            eval_filter = self.fold_filters_list[ifold]["eval_filter"]
            score_name = f"{model_name}_score"
            self.df.loc[eval_filter, score_name] = ret["prediction"]

    def train_model_ifold(self, args, parameters={}):
        model_name = args["model_name"]
        fold_filters = args["fold_filters"]
        ifold = fold_filters["ifold"]
        df = self.df
        print(f"Training model {model_name}, fold #{ifold}...")

        train_filter = fold_filters["train_filter"]
        val_filter = fold_filters["val_filter"]

        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        x_train = df.loc[train_filter, self.features]
        y_train = df.loc[train_filter, "class"]
        x_val = df.loc[val_filter, self.features]
        y_val = df.loc[val_filter, "class"]

        normalized, scalers_save_path = self.normalize_data(
            reference=x_train,
            features=self.features,
            to_normalize_dict={"x_train": x_train, "x_val": x_val},
            model_name=model_name,
            ifold=ifold,
        )
        x_train = normalized["x_train"]
        x_val = normalized["x_val"]
        x_train[other_columns] = df.loc[train_filter, other_columns]
        x_val[other_columns] = df.loc[val_filter, other_columns]

        model = self.models[model_name]
        model = model(len(self.features), label="test")
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

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

        out_path = f"{self.out_path}/models/"
        mkdir(out_path)
        model_save_path = f"{out_path}/model_{model_name}_{ifold}.h5"
        model.save(model_save_path)

        self.plot_history(history, model_name, ifold)

        ret = {
            "model_name": model_name,
            "ifold": ifold,
            "model_save_path": model_save_path,
            "scalers_save_path": scalers_save_path,
        }
        return ret

    def evaluate_model_ifold(self, args, parameters={}):
        model_name = args["model_name"]
        fold_filters = args["fold_filters"]
        ifold = fold_filters["ifold"]
        df = self.df
        print(f"Evaluating model {model_name}, fold #{ifold}...")

        eval_filter = fold_filters["eval_filter"]
        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        scalers = np.load(self.scalers[model_name][ifold] + ".npy")
        x_eval = (df.loc[eval_filter, self.features] - scalers[0]) / scalers[1]
        x_eval[other_columns] = df.loc[eval_filter, other_columns]
        model = load_model(self.trained_models[model_name][ifold])

        prediction = np.array(model.predict(x_eval[self.features])).ravel()
        ret = {"model_name": model_name, "ifold": ifold, "prediction": prediction}
        return ret

    def normalize_data(self, reference, features, to_normalize_dict, model_name, ifold):
        mean = np.mean(reference[features].values, axis=0)
        std = np.std(reference[features].values, axis=0)
        out_path = f"{self.out_path}/scalers/"
        mkdir(out_path)
        save_path = f"{out_path}/scalers_{model_name}_{ifold}"
        np.save(save_path, [mean, std])

        normalized = {}
        for key, item in to_normalize_dict.items():
            item_normalized = (item[features] - mean) / std
            normalized[key] = item_normalized
        return normalized, save_path

    def plot_history(self, history, model_name, ifold):
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"])
        ax.plot(history.history["val_loss"])
        ax.set_title(f"Loss of {model_name}, fold #{ifold}")
        ax.set_ylabel("Loss")
        ax.set_xlabel("epoch")
        ax.legend(["Training", "Validation"], loc="best")
        out_path = f"{self.out_path}/losses/"
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
        out_name = f"{self.out_path}/rocs.png"
        fig.savefig(out_name)
