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
import tensorflow.keras.backend as K

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

        self.fix_variables()
        self.prepare_dataset()
        print()
        print("*" * 60)
        print("In category", self.cat_name)
        print("Event counts in classes:")
        print(self.df["class"].value_counts())
        print("Training features:")
        print(self.features)

        # The part below is needed for cross-validation.
        # Training will be done in 4 steps and at each step
        # the dataset will be split as shown below:
        #
        # T = training (50% of data)
        # V = validation (25% of data)
        # E = evaluation (25% of data)
        # ----------------
        # step 0: T T V E
        # step 1: E T T V
        # step 2: V E T T
        # step 3: T V E T
        # ----------------
        # (splitting is based on event number mod 4)
        #
        # This ensures that all data is used for training
        # and for a given model evaluation is never done
        # on the same data as training.

        self.nfolds = 4
        folds_def = {"train": [0, 1], "val": [2], "eval": [3]}
        self.fold_filters_list = []
        for step in range(self.nfolds):
            fold_filters = {}
            fold_filters["step"] = step
            for fname, folds in folds_def.items():
                folds_shifted = [(step + f) % self.nfolds for f in folds]
                fold_filters[f"{fname}_filter"] = self.df.event.mod(self.nfolds).isin(
                    folds_shifted
                )
            self.fold_filters_list.append(fold_filters)

    def fix_variables(self):
        self.df.loc[:, "mu1_pt_over_mass"] = self.df.mu1_pt / self.df.dimuon_mass
        self.df.loc[:, "mu2_pt_over_mass"] = self.df.mu2_pt / self.df.dimuon_mass

    def prepare_dataset(self):
        # Convert dictionary of datasets to a more useful dataframe
        df_info = pd.DataFrame()
        self.train_samples = []
        for icls, (cls, ds_list) in enumerate(self.ds_dict.items()):
            for ds in ds_list:
                df_info.loc[ds, "dataset"] = ds
                if cls != "ignore":
                    self.train_samples.append(ds)
                    df_info.loc[ds, "class"] = cls
                    df_info.loc[ds, "iclass"] = icls
        df_info["iclass"] = df_info["iclass"].fillna(-1).astype(int)

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
            rets = parallelize(self.train_model, arg_set, client)

        else:
            rets = []
            for model_name in self.models.keys():
                for ff in self.fold_filters_list:
                    ret = self.train_model(
                        {"fold_filters": ff, "model_name": model_name}
                    )
                    rets.append(ret)
        for ret in rets:
            model_name = ret["model_name"]
            step = ret["step"]
            self.trained_models[model_name][step] = ret["model_save_path"]
            self.scalers[model_name][step] = ret["scalers_save_path"]

    def run_evaluation(self, client=None):
        if client:
            arg_set = {
                "model_name": self.models.keys(),
                "fold_filters": self.fold_filters_list,
            }
            rets = parallelize(self.evaluate_model, arg_set, client)
        else:
            rets = []
            for model_name in self.models.keys():
                for ff in self.fold_filters_list:
                    ret = self.evaluate_model(
                        {"fold_filters": ff, "model_name": model_name}
                    )
                    rets.append(ret)
        for ret in rets:
            step = ret["step"]
            model_name = ret["model_name"]
            eval_filter = self.fold_filters_list[step]["eval_filter"]
            score_name = f"{model_name}_score"
            self.df.loc[eval_filter, score_name] = ret["prediction"]

    def train_model(self, args, parameters={}):
        model_name = args["model_name"]
        fold_filters = args["fold_filters"]
        step = fold_filters["step"]
        df = self.df[self.df.dataset.isin(self.train_samples)]

        print(f"Training model {model_name}, step #{step+1} out of {self.nfolds}...")
        K.clear_session()

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
            step=step,
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
        model_save_path = f"{out_path}/model_{model_name}_{step}.h5"
        model.save(model_save_path)

        K.clear_session()
        print(f"Done training: model {model_name}, step #{step+1} out of {self.nfolds}")
        self.plot_history(history, model_name, step)
        ret = {
            "model_name": model_name,
            "step": step,
            "model_save_path": model_save_path,
            "scalers_save_path": scalers_save_path,
        }
        return ret

    def evaluate_model(self, args, parameters={}):
        model_name = args["model_name"]
        fold_filters = args["fold_filters"]
        step = fold_filters["step"]
        df = self.df

        print(f"Evaluating model {model_name}, step #{step+1} out of {self.nfolds}...")
        K.clear_session()

        eval_filter = fold_filters["eval_filter"]
        other_columns = ["event", "lumi_wgt", "mc_wgt"]

        scalers = np.load(self.scalers[model_name][step] + ".npy")
        x_eval = (df.loc[eval_filter, self.features] - scalers[0]) / scalers[1]
        x_eval[other_columns] = df.loc[eval_filter, other_columns]
        model = load_model(self.trained_models[model_name][step])
        prediction = np.array(model.predict(x_eval[self.features])).ravel()

        K.clear_session()
        print(
            f"Done evaluating: model {model_name}, step #{step+1} out of {self.nfolds}"
        )

        ret = {"model_name": model_name, "step": step, "prediction": prediction}
        return ret

    def normalize_data(self, reference, features, to_normalize_dict, model_name, step):
        mean = np.mean(reference[features].values, axis=0)
        std = np.std(reference[features].values, axis=0)
        out_path = f"{self.out_path}/scalers/"
        mkdir(out_path)
        save_path = f"{out_path}/scalers_{model_name}_{step}"
        np.save(save_path, [mean, std])

        normalized = {}
        for key, item in to_normalize_dict.items():
            item_normalized = (item[features] - mean) / std
            normalized[key] = item_normalized
        return normalized, save_path

    def plot_history(self, history, model_name, step):
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"])
        ax.plot(history.history["val_loss"])
        ax.set_title(f"Loss of {model_name}, fold #{step}")
        ax.set_ylabel("Loss")
        ax.set_xlabel("epoch")
        ax.legend(["Training", "Validation"], loc="best")
        out_path = f"{self.out_path}/losses/"
        mkdir(out_path)
        out_name = f"{out_path}/loss_{model_name}_{step}.png"
        fig.savefig(out_name)

    def plot_roc_curves(self):
        roc_curves = {}
        fig = plt.figure()
        fig, ax = plt.subplots()
        df = self.df[self.df.dataset.isin(self.train_samples)]
        for model_name, model in self.models.items():
            score_name = f"{model_name}_score"
            roc_curves[score_name] = roc_curve(
                y_true=df["class"],
                y_score=df[score_name],
                sample_weight=df["lumi_wgt"] * df["mc_wgt"],
            )
            ax.plot(
                roc_curves[score_name][0], roc_curves[score_name][1], label=score_name
            )
        ax.legend(prop={"size": "x-small"})
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        out_name = f"{self.out_path}/rocs.png"
        fig.savefig(out_name)
