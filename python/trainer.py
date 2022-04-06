import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from python.workflow import parallelize
from python.io import mkdir
from python.convert import to_histograms
from python.plotter import plotter
from python.variable import Variable


style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)


def run_mva(client, parameters, df):
    mva_path = parameters.pop("mva_path", "./")
    mkdir(mva_path)
    mva_models = parameters.pop("mva_models", {})
    saved_models = parameters.pop("saved_models", {})
    training_datasets = parameters.pop("training_datasets", {})
    features = parameters.pop("training_features", [])
    do_training = parameters.pop("mva_do_training", False)
    do_evaluation = parameters.pop("mva_do_evaluation", False)
    do_plotting = parameters.pop("mva_do_plotting", False)
    channels_to_use = parameters.get("mva_channels", ["ggh_0jets"])

    for channel in channels_to_use:
        out_dir = f"{mva_path}/{channel}"
        mkdir(out_dir)
        parameters["plots_path"] = out_dir

        trainer = Trainer(
            df=df[df.channel == channel],
            channel=channel,
            ds_dict=training_datasets,
            features=features,
            out_path=out_dir,
            training_cut="(dimuon_mass > 110) & (dimuon_mass < 150)",
        )
        # trainer.shape_in_eta_bins(shape_of="dimuon_mass", nbins=10)

        if do_training:
            trainer.add_models(mva_models.copy())
            trainer.run_training(client)

        if channel in saved_models.keys():
            if len(saved_models[channel].keys()) > 0:
                trainer.add_saved_models(saved_models[channel])

        if do_evaluation:
            trainer.run_evaluation(client)
            trainer.shape_in_bins(shape_of="dimuon_mass", nbins=6)
            # trainer.shape_in_bins(shape_of="max_abs_eta", nbins=10)

            for model_name in trainer.models.keys():
                score_name = f"{model_name}_score"
                df.loc[df.channel == channel, score_name] = trainer.df.loc[
                    :, score_name
                ]

        if do_plotting:
            trainer.plot_roc_curves()
            parameters_tmp = parameters.copy()
            parameters_tmp["hist_vars"] = []
            parameters_tmp["plot_vars"] = []
            parameters_tmp["regions"] = ["h-peak", "h-sidebands"]
            parameters_tmp["channels"] = [channel]
            all_models = []
            if channel in mva_models.keys():
                all_models += list(mva_models[channel].keys())
            if channel in saved_models.keys():
                all_models += list(saved_models[channel].keys())
            all_models = list(set(all_models))
            for model_name in all_models:
                score_name = f"{model_name}_score"
                parameters_tmp["hist_vars"].append(score_name)
                parameters_tmp["plot_vars"].append(score_name)
                parameters_tmp["variables_lookup"][score_name] = Variable(
                    score_name, score_name, 50, 0, 1
                )

            hist_df = to_histograms(client, parameters_tmp, trainer.df)
            plotter(client, parameters_tmp, hist_df)


class Trainer(object):
    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", pd.DataFrame())
        self.channel = kwargs.pop("channel", "")
        self.ds_dict = kwargs.pop("ds_dict", {})
        self.features = kwargs.pop("features", [])
        self.out_path = kwargs.pop("out_path", "./")
        self.training_cut = kwargs.pop("training_cut", None)
        self.models = {}
        self.trained_models = {}
        self.scalers = {}

        self.fix_variables()
        self.prepare_dataset()
        print()
        print("*" * 60)
        print("In channel", self.channel)
        print("Event counts in classes:")
        print(self.df["class_name"].value_counts())
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
        # Ignore features that have incorrect values
        # (we handle them by categorizing data by njets)
        ignore_features = self.df.loc[:, self.df.min(axis=0) == -999.0].columns
        features_clean = []
        for c in self.df.columns:
            if (c in self.features) and (c not in ignore_features):
                features_clean.append(c)
        self.features = features_clean

        # Convert dictionary of datasets to a more useful dataframe
        df_info = pd.DataFrame()
        self.train_samples = []
        for icls, (cls, ds_list) in enumerate(self.ds_dict.items()):
            for ds in ds_list:
                df_info.loc[ds, "dataset"] = ds
                df_info.loc[ds, "iclass"] = -1
                if cls != "ignore":
                    self.train_samples.append(ds)
                    df_info.loc[ds, "class_name"] = cls
                    df_info.loc[ds, "iclass"] = icls
        df_info["iclass"] = df_info["iclass"].fillna(-1).astype(int)
        self.df = self.df[self.df.dataset.isin(df_info.dataset.unique())]

        # Assign numerical classes to each event
        cls_map = dict(df_info[["dataset", "iclass"]].values)
        cls_name_map = dict(df_info[["dataset", "class_name"]].values)
        self.df["class"] = self.df.dataset.map(cls_map)
        self.df["class_name"] = self.df.dataset.map(cls_name_map)

    def add_models(self, model_dict):
        if self.channel in model_dict.keys():
            self.models = model_dict[self.channel]
        self.trained_models = {n: {} for n in self.models.keys()}
        self.scalers = {n: {} for n in self.models.keys()}

    def add_saved_models(self, model_dict):
        for model_name, model_props in model_dict.items():
            model_path = model_props["path"]
            self.models[model_name] = {"type": model_props["type"]}
            print(f"Loading model {model_name} from {model_path}")
            self.trained_models[model_name] = {}
            self.scalers[model_name] = {}
            for step in range(self.nfolds):
                self.trained_models[model_name][
                    step
                ] = f"{model_path}/models/model_{model_name}_{step}.h5"
                self.scalers[model_name][
                    step
                ] = f"{model_path}/scalers/scalers_{model_name}_{step}"

    def run_training(self, client=None):
        if len(self.models) == 0:
            return
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
        if len(self.models) == 0:
            return
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
        if self.training_cut is not None:
            df = df.query(self.training_cut)

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

        model = self.models[model_name]["model"]
        model_type = self.models[model_name]["type"]

        out_path = f"{self.out_path}/models/"
        mkdir(out_path)

        if model_type == "dnn":
            model = model(len(self.features), label="test")
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            history = model.fit(
                x_train[self.features],
                y_train,
                epochs=100,
                batch_size=1024,
                verbose=0,
                validation_data=(x_val[self.features], y_val),
                shuffle=True,
            )
            model_save_path = f"{out_path}/model_{model_name}_{step}.h5"
            model.save(model_save_path)
            K.clear_session()
            losses = {
                "train_loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
            }

        elif model_type == "dnn_adv":
            model = model(len(self.features), label="test_adv")
            losses = {"classifier": "binary_crossentropy", "adversary": "mse"}
            loss_weights = {"classifier": 1, "adversary": -1}
            model.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer="adam",
                metrics=["accuracy"],
            )

            history = model.fit(
                x_train[self.features],
                {
                    "classifier": df.loc[train_filter, "class"],
                    "adversary": df.loc[train_filter, "dimuon_mass"],
                },
                epochs=100,
                batch_size=1024,
                verbose=0,
                validation_data=(
                    x_val[self.features],
                    {
                        "classifier": df.loc[val_filter, "class"],
                        "adversary": df.loc[val_filter, "dimuon_mass"],
                    },
                ),
                shuffle=True,
            )
            model_save_path = f"{out_path}/model_{model_name}_{step}.h5"
            model.save(model_save_path)
            K.clear_session()

            losses = {
                "train_loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
            }

        elif model_type == "bdt":
            model.fit(
                x_train[self.features],
                y_train,
                early_stopping_rounds=50,
                eval_metric="logloss",
                eval_set=[
                    (x_train[self.features], y_train),
                    (x_val[self.features], y_val),
                ],
                verbose=False,
            )
            model_save_path = f"{out_path}/model_{model_name}_{step}.pkl"
            pickle.dump(model, open(model_save_path, "wb"))
            results = model.evals_result()
            losses = {
                "train_loss": results["validation_0"]["logloss"],
                "val_loss": results["validation_1"]["logloss"],
            }

        self.plot_history(losses, model_name, step)

        print(f"Done training: model {model_name}, step #{step+1} out of {self.nfolds}")

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
        if x_eval.shape[0] == 0:
            return {"model_name": model_name, "step": step, "prediction": []}

        x_eval[other_columns] = df.loc[eval_filter, other_columns]

        model_type = self.models[model_name]["type"]
        if model_type == "dnn":
            model = load_model(self.trained_models[model_name][step])
            prediction = np.array(model.predict(x_eval[self.features])).ravel()
            K.clear_session()
        if model_type == "dnn_adv":
            model = load_model(self.trained_models[model_name][step])
            prediction = np.array(model.predict(x_eval[self.features])[0]).ravel()
            K.clear_session()
        elif model_type == "bdt":
            model = pickle.load(open(self.trained_models[model_name][step], "rb"))
            prediction = np.array(
                model.predict_proba(x_eval[self.features])[:, 1]
            ).ravel()

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

    def plot_history(self, losses, model_name, step):
        fig = plt.figure()
        fig, ax = plt.subplots()
        for name, loss in losses.items():
            ax.plot(loss)
        # ax.plot(history.history["loss"])
        # ax.plot(history.history["val_loss"])
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
                sample_weight=df["lumi_wgt"],  # * df["mc_wgt"],
            )
            ax.plot(
                roc_curves[score_name][0], roc_curves[score_name][1], label=score_name
            )
        ax.legend(prop={"size": "x-small"})
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        out_name = f"{self.out_path}/rocs.png"
        fig.savefig(out_name)

    def shape_in_bins(self, shape_of="dimuon_mass", nbins=4):
        for model_name in self.models.keys():
            score_name = f"{model_name}_score"
            for cls in self.df["class_name"].dropna().unique():
                df = self.df[self.df["class_name"] == cls]
                score = df[score_name]
                fig = plt.figure()
                fig, ax = plt.subplots()

                for i in range(nbins):
                    cut_lo = score.quantile(i / nbins)
                    cut_hi = score.quantile((i + 1) / nbins)
                    cut = (score > cut_lo) & (score < cut_hi)
                    if shape_of == "max_abs_eta":
                        data = df.loc[cut].apply(max_abs_eta, axis=1)
                    else:
                        data = df.loc[cut, shape_of]

                    if shape_of == "dimuon_mass":
                        data_range = (115, 135)
                        data_bins = 80
                    else:
                        data_range = (data.min(), data.max())
                        data_bins = 25
                    weights = (df.lumi_wgt).loc[cut].values
                    # weights = (df.lumi_wgt * df.mc_wgt).loc[cut].values
                    hist = np.histogram(
                        data.values,
                        bins=data_bins,
                        range=data_range,
                        weights=weights,
                        density=True,
                    )
                    hep.histplot(
                        hist[0],
                        hist[1],
                        histtype="step",
                        label=f"{score_name} bin #{i}",
                    )
                ax.legend(prop={"size": "x-small"})
                ax.set_xlabel(shape_of)
                ax.set_yscale("log")
                ax.set_ylim(0.0001, 1)
                out_name = f"{self.out_path}/shapes_{score_name}_{cls}_{shape_of}.png"
                fig.savefig(out_name)

    def shape_in_eta_bins(self, shape_of="dimuon_mass", nbins=4):

        for cls in self.df["class_name"].dropna().unique():
            df = self.df[self.df["class_name"] == cls]
            score = df.apply(max_abs_eta, axis=1)
            fig = plt.figure()
            fig, ax = plt.subplots()

            for i in range(nbins):
                cut_lo = score.quantile(i / nbins)
                cut_hi = score.quantile((i + 1) / nbins)
                cut = (score > cut_lo) & (score < cut_hi)
                if shape_of == "max_abs_eta":
                    data = df.loc[cut].apply(max_abs_eta, axis=1)
                else:
                    data = df.loc[cut, shape_of]

                if shape_of == "dimuon_mass":
                    data_range = (115, 135)
                    data_bins = 80
                else:
                    data_range = (data.min(), data.max())
                    data_bins = 25
                # weights = (df.lumi_wgt * df.mc_wgt).loc[cut].values
                weights = (df.lumi_wgt).loc[cut].values
                hist = np.histogram(
                    data.values,
                    bins=data_bins,
                    range=data_range,
                    weights=weights,
                    density=True,
                )
                cut_lo = round(cut_lo, 2)
                cut_hi = round(cut_hi, 2)
                hep.histplot(
                    hist[0],
                    hist[1],
                    histtype="step",
                    label=f"max |eta|: [{cut_lo}, {cut_hi}]",
                )
            ax.legend(prop={"size": "x-small"})
            ax.set_xlabel(shape_of)
            out_name = f"{self.out_path}/shapes_eta_{cls}_{shape_of}.png"
            fig.savefig(out_name)


def max_abs_eta(row):
    return max(abs(row["mu1_eta"]), abs(row["mu2_eta"]))
