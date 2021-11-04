from dask.distributed import Client
import ROOT as rt
import argparse
import dask.dataframe as dd
import pandas as pd
import glob
from fitter import Fitter
from fitmodels import chebyshev, doubleCB, bwGamma, bwZredux

rt.gROOT.SetBatch(True)
rt.gStyle.SetOptStat(0)

GEN_XSEC = 100
NCATS = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--channel", type=str, default="ggH", help="Which channel you want to run?"
)
parser.add_argument(
    "--category", type=str, default="cat0", help="Which category you want to run?"
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
    "--ext", type=str, default="", help="The extension of output File names"
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
    df_future = [d[["dimuon_mass", "s"]] for d in df_future]
    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    df = df.compute()
    df.reset_index(inplace=True, drop=True)

    sig_cut = df.s.str.contains("ggh")
    df_signal = df.loc[sig_cut, "dimuon_mass"]
    df_background = df.loc[~sig_cut, "dimuon_mass"]

    category = args.category

    my_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        fitmodels={
            "bwz_redux": bwZredux,
            "bwgamma": bwGamma,
            "dcb": doubleCB,
            "chebyshev": chebyshev,
        },
        requires_order=["chebyshev"],
        channel=args.channel,
        filename_ext=args.ext,
    )

    if args.doBackgroundFit:
        my_fitter.simple_fit(
            dataset=df_background,
            label="background",
            category=args.category,
            blinded=args.isBlinded,
            model_names=["bwz_redux", "bwgamma"],
            fix_parameters=False,
            title="Background",
            save=True,
        )

    if args.doSignalFit:
        my_fitter.simple_fit(
            dataset=df_signal,
            label="ggh_signal",
            category=args.category,
            blinded=False,
            model_names=["dcb"],
            fix_parameters=True,
            title="Signal",
            save=True,
        )

    if args.getBkgModelFromMC:
        fake_data = my_fitter.generate_data(
            "bwz_redux", args.category, GEN_XSEC, args.intLumi
        )
        my_fitter.simple_fit(
            dataset=fake_data,
            label="fakedata",
            category=args.category,
            blinded=False,
            model_names=["bwz_redux", "bwgamma"],
            fix_parameters=False,
            title="Background",
            save=True,
        )

    if args.doCorePdfFit:
        core_model_names = ["bwz_redux"]

        for model_name in core_model_names:
            my_fitter.add_model(model_name, category="corepdf")

        # corePDF_results = {}
        # fake_ds = rt.RooDataSet("full_dataset", "full_dataset")
        # outputfile_fakedata = rt.TFile("data_histograms_EachCat_and_Full.root","recreate")

        hists_all = {}
        data_stack = rt.THStack("full_data", "full_data")
        for icat in range(NCATS):
            hist_name = f"hist_{my_fitter.channel}_cat{icat}"
            fake_data = my_fitter.generate_data(
                "bwz_redux", f"_{my_fitter.channel}_corepdf", GEN_XSEC, args.intLumi
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
            my_fitter.add_data(hist, ds_name=f"{hist_name}_fake", blinded=False)

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
        my_fitter.add_data(full_dataset, ds_name=ds_core_fake_name, blinded=False)

        my_fitter.workspace.Print()
        my_fitter.fit(
            ds_core_fake_name,
            core_model_names,
            blinded=False,
            fix_parameters=False,
            category="corepdf",
            label="fake_data_Background_corPdfFit",
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
            category = f"cat{icat}"
            hist_name = f"hist_{my_fitter.channel}_{category}"

            transfer_hist = hists_all[hist_name].Clone()
            transfer_hist.Divide(data_stack_full)
            transfer_hist.Scale(1 / transfer_hist.Integral())
            transfer_dataset = rt.RooDataHist(
                f"transfer_{category}",
                f"transfer_{category}",
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transfer_hist,
            )
            transfer_data_name = transfer_dataset.GetName()

            my_fitter.add_data(
                transfer_dataset, ds_name=transfer_data_name, blinded=False
            )

            cheby_order = 3 if icat < 1 else 2
            my_fitter.add_model("chebyshev", category=category, order=cheby_order)
            transfer_func_names = [f"chebyshev{cheby_order}"]

            my_fitter.fit(
                transfer_data_name,
                transfer_func_names,
                blinded=False,
                fix_parameters=False,
                category=category,
                label="fake_data_Background_corPdfFit",
                title="Background",
                save=True,
            )
            """
            ws_corepdf = rt.RooWorkspace("ws_corepdf", False)
            ws_corepdf.Import(
                my_fitter.workspace.pdf(
                    f"bwz_redux_{my_fitter.channel}_corepdf"
                )
            )
            ws_corepdf.Import(transfer_dataset)
            coreBWZRedux = rt.RooProdPdf(
                f"bkg_bwzredux_{my_fitter.channel}_{category}",
                f"bkg_bwzredux_{my_fitter.channel}_{category}",
                my_fitter.workspace.pdf(
                    f"bwz_redux_{my_fitter.channel}_corepdf"
                ),
                my_fitter.workspace.pdf(
                    f"{transfer_func_name}_{my_fitter.channel}_{category}"
                ),
            )
            ws_corepdf.Import(coreBWZRedux, rt.RooFit.RecycleConflictNodes())
            cat_dataSet = rt.RooDataHist(
                f"data_{category}",
                f"data_{category}",
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transfer_dataset,
            )
            ndata_cat = cat_dataSet.sumEntries()
            norm_cat = rt.RooRealVar(
                f"bkg_{category}_pdf_norm",
                f"bkg_{category}_pdf_norm",
                ndata_cat,
                -float("inf"),
                float("inf"),
            )
            ws_corepdf.Import(cat_dataSet)
            ws_corepdf.Import(norm_cat)
            ws_corepdf.Import(norm_Core)
            saveWorkspace(ws_corepdf, f"workspace_BackgroundFit{category}")
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
    parameters = {"ncpus": 20}
    # paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_*/*.parquet')
    paths = []
    paths.extend(glob.glob("/depot/cms/hmm/coffea/2016_sep26/ggh_amcPS/*.parquet"))
    paths.extend(glob.glob("/depot/cms/hmm/coffea/2016_sep26/data_D/*.parquet"))

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
