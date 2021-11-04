from dask.distributed import Client
import ROOT as rt
import argparse
import dask.dataframe as dd
import pandas as pd
import glob
from fitter import Fitter

rt.gROOT.SetBatch(True)
rt.gStyle.SetOptStat(0)

GEN_XSEC = 100
NCATS = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--process", type=str, default="ggH", help="Which process you want to run?"
)
parser.add_argument(
    "--category", type=str, default="0", help="Which category you want to run?"
)
parser.add_argument(
    "--isBlinded", default=True, help="Do you want to run blinded bkg fit?"
)
parser.add_argument(
    "--doBackgroundFit", default=False, help="Do you want to run bkg fit?"
)
parser.add_argument(
    "--doCorePdfFit", default=False, help="Do you want to run corePdf bkg fit?"
)
parser.add_argument(
    "--doSignalFit", default=False, help="Do you want to run signal fit?"
)
parser.add_argument(
    "--getBkgModelFromMC", default=False, help="Do you want to get bkg model from MC?"
)
parser.add_argument("--intLumi", default=3000, help="Integrated Luminosity in 1/fb")
parser.add_argument(
    "--ext", type=str, default="_new", help="The extension of output File names"
)
args = parser.parse_args()


def load_data(path):
    if len(path) > 0:
        df = dd.read_parquet(path)
    else:
        df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    return df


def workflow(client, paths, parameters):
    # Load dataframes
    df_future = client.map(load_data, paths)
    df_future = client.gather(df_future)

    # Select only one column and concatenate
    df_future = [d[["dimuon_mass"]] for d in df_future]
    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    df = df.compute()
    df.reset_index(inplace=True, drop=True)
    # modelNames = ["bwZRedux","bwGamma"]
    # modelNames = ["bwg_model"]
    # fittedmodels = {}
    isBlinded = args.isBlinded
    process = args.process
    category = args.category
    tag = f"_{process}_{category}"
    lumi = args.intLumi

    my_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        process=process,
        category=category,
    )

    if args.doBackgroundFit:
        model_names = ["bwz_redux_model", "bwg_model"]
        ds_name = "background_ds"

        my_fitter.add_data(df, name=ds_name, blinded=isBlinded)
        my_fitter.add_models(model_names)
        my_fitter.workspace.Print()
        my_fitter.fit(
            ds_name,
            model_names,
            blinded=isBlinded,
            fix_parameters=False,
            name=f"data_BackgroundFit{args.ext}",
            title="Background",
            save=True,
        )
        my_fitter.save_workspace(f"workspace_BackgroundFit{tag}{args.ext}")

    if args.doSignalFit:
        isBlinded = False
        model_names = ["dcb_model"]
        ds_name = "signal_ds"

        my_fitter.add_data(df, name=ds_name, blinded=isBlinded)
        my_fitter.add_models(model_names)
        my_fitter.workspace.Print()

        my_fitter.fit(
            ds_name,
            model_names,
            blinded=isBlinded,
            fix_parameters=True,
            name=f"ggH_SignalFit{args.ext}",
            title="Signal",
            save=True,
        )
        my_fitter.save_workspace(f"workspace_sigFit{tag}{args.ext}")

    if args.getBkgModelFromMC:
        isBlinded = False
        model_names = ["bwz_redux_model", "bwg_model"]
        ds_name = "fakedata_ds"

        fake_data = my_fitter.generate_data("bwz_redux_model", tag, GEN_XSEC, lumi)
        my_fitter.add_data(fake_data, name=ds_name, blinded=False)
        my_fitter.add_models(model_names)
        my_fitter.workspace.Print()

        my_fitter.fit(
            ds_name,
            model_names,
            blinded=isBlinded,
            fix_parameters=False,
            name=f"fake_data_BackgroundFit{args.ext}",
            title="Background",
            save=True,
        )
        my_fitter.save_workspace(f"workspace_BackgroundFit{tag}{args.ext}")

    if args.doCorePdfFit:
        isBlinded = False
        core_model_names = ["bwz_redux_model"]

        my_fitter.add_models(
            [{"name": m, "tag": f"_{process}_corepdf"} for m in core_model_names]
        )

        # corePDF_results = {}
        # fake_ds = rt.RooDataSet("full_dataset", "full_dataset")
        # outputfile_fakedata = rt.TFile("data_histograms_EachCat_and_Full.root","recreate")

        hists_all = {}
        data_stack = rt.THStack("full_data", "full_data")
        for icat in range(NCATS):
            hist_name = f"hist_{process}_cat{icat}"
            fake_data = my_fitter.generate_data(
                "bwz_redux_model", f"_{process}_corepdf", GEN_XSEC, lumi
            )
            # fake_ds.append(fake_data)
            hist = rt.RooAbsData.createHistogram(
                fake_data,
                hist_name,
                my_fitter.workspace.var("mass"),
                rt.RooFit.Binning(80),
            )
            # hist.Write()
            hists_all[hist_name] = hist
            data_stack.Add(hist)
            my_fitter.add_data(hist, name=f"{hist_name}_fake", blinded=False)

        data_stack_full = data_stack.GetStack().Last()
        hists_all[data_stack_full.GetName()] = data_stack_full
        # data_stack_full.Write()
        # outputfile_fakedata.Close()

        full_dataset = rt.RooDataHist(
            "core_Data",
            "core_Data",
            rt.RooArgList(my_fitter.workspace.var("mass")),
            data_stack_full,
        )
        ds_core_fake_name = "ds_core_fake"
        my_fitter.add_data(full_dataset, name=ds_core_fake_name, blinded=False)

        my_fitter.workspace.Print()
        my_fitter.fit(
            ds_core_fake_name,
            core_model_names,
            blinded=isBlinded,
            fix_parameters=False,
            tag=f"_{process}_corepdf",
            name=f"fake_data_Background_corPdfFit{args.ext}",
            title="Background",
            save=True,
        )
        """
        norm_Core = rt.RooRealVar(
            "bkg_norm_Core",
            "bkg_norm_Core",
            full_dataset.sumEntries(),
            -float("inf"),
            float("inf"),
        )
        """

        for icat in range(NCATS):
            hist_name = f"hist_{process}_cat{icat}"
            suffix = f"cat{icat}"

            transfer_hist = hists_all[hist_name].Clone()
            transfer_hist.Divide(data_stack_full)
            transfer_hist.Scale(1 / transfer_hist.Integral())
            transfer_dataset = rt.RooDataHist(
                f"transfer_{suffix}",
                f"transfer_{suffix}",
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transfer_hist,
            )
            transfer_data_name = transfer_dataset.GetName()

            my_fitter.add_data(
                transfer_dataset, name=transfer_data_name, blinded=isBlinded
            )

            cheby_order = 3 if icat < 1 else 2
            my_fitter.add_model(
                "chebychev_model", tag=f"_{process}_{suffix}", order=cheby_order
            )
            transfer_func_names = [f"chebychev{cheby_order}"]

            my_fitter.fit(
                transfer_data_name,
                transfer_func_names,
                blinded=isBlinded,
                fix_parameters=False,
                tag=f"_{process}_{suffix}",
                name=f"fake_data_Background_corPdfFit{args.ext}",
                title="Background",
                save=True,
            )
            """
            ws_corepdf = rt.RooWorkspace("ws_corepdf", False)
            ws_corepdf.Import(
                my_fitter.workspace.pdf(
                    f"bwz_redux_model_{process}_corepdf"
                )
            )
            ws_corepdf.Import(transfer_dataset)
            coreBWZRedux = rt.RooProdPdf(
                "bkg_bwzredux_" + "_" + process + "_" + suffix,
                "bkg_bwzredux_" + "_" + process + "_" + suffix,
                my_fitter.workspace.pdf(
                    "bwz_redux_model" + "_" + process + "_corepdf"
                ),
                my_fitter.workspace.pdf(
                    transfer_func_name + "_" + process + "_" + suffix
                ),
            )
            ws_corepdf.Import(coreBWZRedux, rt.RooFit.RecycleConflictNodes())
            cat_dataSet = rt.RooDataHist(
                "data_" + suffix,
                "data_" + suffix,
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transfer_dataset,
            )
            ndata_cat = cat_dataSet.sumEntries()
            norm_cat = rt.RooRealVar(
                "bkg_" + suffix + "_pdf_norm",
                "bkg_" + suffix + "_pdf_norm",
                ndata_cat,
                -float("inf"),
                float("inf"),
            )
            ws_corepdf.Import(cat_dataSet)
            ws_corepdf.Import(norm_cat)
            ws_corepdf.Import(norm_Core)
            saveWorkspace(ws_corepdf, "workspace_BackgroundFit" + suffix + args.ext)
            """


def GOF(ws, pdfName, dsName, tag, ndata):
    normalization = rt.RooRealVar(
        "normaliazation", "normalization", ndata, 0.5 * ndata, 2 * ndata
    )
    model = rt.RooExtendPdf("ext", "ext", ws.pdf(pdfName), normalization)
    xframe = ws.var("mass" + tag).frame()
    ds = ws.data(dsName)
    ds.plotOn(xframe, rt.RooFit.Name("ds"))
    model.plotOn(xframe, rt.RooFit.Name("model"))
    nparam = model.getParameters(ds).getSize()
    chi2 = xframe.chiSquare("model", "ds", nparam)
    nBins = ds.numEntries()
    if float(ndata) / nBins < 5:
        # ntoys = 500
        print(" can't use asymptotic approximation!! need to run toys")
        prob = getPValue(chi2 * (nBins - nparam), nBins - nparam)

    else:
        prob = getPValue(chi2 * (nBins - nparam), nBins - nparam)

    print("chi2/ndof = " + str(chi2))
    print("chi2" + str(chi2 * (nBins - nparam)))
    print("p-value = " + str(prob))

    return chi2, prob


def getPValue(chi2, ndof):
    prob = rt.TMath.Prob(chi2, ndof)
    return prob


if __name__ == "__main__":
    parameters = {"ncpus": 1}
    # paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_*/*.parquet')
    if args.doSignalFit:
        paths = glob.glob("/depot/cms/hmm/coffea/2016_sep26/ggh_amcPS/*.parquet")
    else:
        paths = glob.glob("/depot/cms/hmm/coffea/2016_sep26/data_D/*.parquet")

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
