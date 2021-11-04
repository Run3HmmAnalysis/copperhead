from dask.distributed import Client
import ROOT as rt
import argparse
import dask.dataframe as dd
import pandas as pd
import glob
from fitter import Fitter

rt.gROOT.SetBatch(True)
rt.gStyle.SetOptStat(0)

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
    processName = args.process
    category = args.category
    tag = "_" + processName + "_" + category
    lumi = args.intLumi

    my_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        process_name=processName,
        category=category,
    )
    ws = my_fitter.workspace

    if args.doBackgroundFit:
        model_names = ["bwz_redux_model", "bwg_model"]
        my_fitter.add_data(df, isBlinded)
        ds_name = "ds"
        my_fitter.workspace.Print()

        for fitmodel in model_names:
            my_fitter.add_model(fitmodel)

        my_fitter.fit(
            ds_name,
            model_names,
            blinded=isBlinded,
            fix_parameters=False,
            name=f"data_BackgroundFit{args.ext}",
            title="Background",
            save=True,
        )

        saveWorkspace(my_fitter.workspace, "workspace_BackgroundFit" + tag + args.ext)

    if args.doSignalFit:
        isBlinded = False
        model_names = ["dcb_model"]
        ds_name = "ds"
        my_fitter.add_data(df, isBlinded)

        for fitmodel in model_names:
            my_fitter.add_model(fitmodel)

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
        saveWorkspace(my_fitter.workspace, "workspace_sigFit" + tag + args.ext)

    if args.getBkgModelFromMC:
        model_names = ["bwz_redux_model", "bwg_model"]
        isBlinded = False
        for fitmodel in model_names:
            my_fitter.add_model(fitmodel)
        fake_data = generateData(my_fitter.workspace, "bwz_redux_model", tag, 100, lumi)
        my_fitter.add_data(fake_data, False, "_fake", False)
        my_fitter.workspace.Print()
        my_fitter.fit(
            "ds_fake",
            model_names,
            blinded=isBlinded,
            fix_parameters=False,
            name=f"fake_data_BackgroundFit{args.ext}",
            title="Background",
            save=True,
        )
        saveWorkspace(my_fitter.workspace, "workspace_BackgroundFit" + tag + args.ext)

    if args.doCorePdfFit:
        # corePDF_results = {}
        hists_All = {}
        nCats = 5
        core_model_names = ["bwz_redux_model"]
        for fitmodel in core_model_names:
            my_fitter.add_model(fitmodel, tag=f"_{processName}_corepdf")
        isBlinded = False
        # fake_data = rt.RooDataSet("full_dataset", "full_dataset")
        # outputfile_fakedata = rt.TFile("data_histograms_EachCat_and_Full.root","recreate")
        dataStack = rt.THStack("full_data", "full_data")
        for icat in range(nCats):
            # tag = "_"+processName+"_cat"+icat
            hist_name = "hist" + "_" + processName + "_cat" + str(icat)
            # hist = rt.TH1F(hist_name,hist_name,80,110.,150.)
            ds = generateData(
                my_fitter.workspace,
                "bwz_redux_model",
                "_" + processName + "_corepdf",
                100,
                lumi,
            )
            hist = rt.RooAbsData.createHistogram(
                ds, hist_name, my_fitter.workspace.var("mass"), rt.RooFit.Binning(80)
            )
            # hist.Write()
            # print(hist.GetName())
            hists_All[hist_name] = hist
            print(hists_All)
            print(hists_All[hist_name].Integral())
            # fake_data.append(ds)
            dataStack.Add(hist)
            my_fitter.add_data(hist, False, hist_name + "_fake", False)
        print(hists_All)
        dataStack_Full = dataStack.GetStack().Last()
        hists_All[dataStack_Full.GetName()] = dataStack_Full
        # dataStack_Full.Write()
        # outputfile_fakedata.Close()
        print(hists_All)
        fullDataSet = rt.RooDataHist(
            "core_Data",
            "core_Data",
            rt.RooArgList(my_fitter.workspace.var("mass")),
            dataStack_Full,
        )
        my_fitter.add_data(fullDataSet, False, "_Core_fake", False)

        my_fitter.workspace.Print()
        my_fitter.fit(
            "ds_Core_fake",
            core_model_names,
            blinded=isBlinded,
            fix_parameters=False,
            tag=f"_{processName}_corepdf",
            name=f"fake_data_Background_corPdfFit{args.ext}",
            title="Background",
            save=True,
        )

        norm_Core = rt.RooRealVar(
            "bkg_norm_Core",
            "bkg_norm_Core",
            fullDataSet.sumEntries(),
            -float("inf"),
            float("inf"),
        )
        for icat in range(nCats):
            histName = "hist" + "_" + processName + "_cat" + str(icat)
            ws_corepdf = rt.RooWorkspace("ws_corepdf", False)
            ws_corepdf.Import(
                ws.pdf("bwz_redux_model" + "_" + processName + "_corepdf")
            )
            prefix = "cat" + str(icat)
            print(hists_All)
            transferHist = hists_All[histName].Clone()
            transferHist.Divide(dataStack_Full)
            transferHist.Scale(1 / transferHist.Integral())
            transferDataSet = rt.RooDataHist(
                "transfer_" + prefix,
                "transfer_" + prefix,
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transferHist,
            )
            transferDataName = transferDataSet.GetName()
            my_fitter.workspace.Import(transferDataSet)
            ws_corepdf.Import(transferDataSet)
            chebyOrder = 3 if icat < 1 else 2
            my_fitter.add_model(
                "chebychev_model", tag=f"_{processName}_{prefix}", order=chebyOrder
            )
            transferFuncName = "chebychev" + str(chebyOrder)
            my_fitter.fit(
                transferDataName,
                [transferFuncName],
                blinded=isBlinded,
                fix_parameters=False,
                tag=f"_{processName}_{prefix}",
                name=f"fake_data_Background_corPdfFit{args.ext}",
                title="Background",
                save=True,
            )
            coreBWZRedux = rt.RooProdPdf(
                "bkg_bwzredux_" + "_" + processName + "_" + prefix,
                "bkg_bwzredux_" + "_" + processName + "_" + prefix,
                my_fitter.workspace.pdf(
                    "bwz_redux_model" + "_" + processName + "_corepdf"
                ),
                my_fitter.workspace.pdf(
                    transferFuncName + "_" + processName + "_" + prefix
                ),
            )
            ws_corepdf.Import(coreBWZRedux, rt.RooFit.RecycleConflictNodes())
            cat_dataSet = rt.RooDataHist(
                "data_" + prefix,
                "data_" + prefix,
                rt.RooArgList(my_fitter.workspace.var("mass")),
                transferDataSet,
            )
            ndata_cat = cat_dataSet.sumEntries()
            norm_cat = rt.RooRealVar(
                "bkg_" + prefix + "_pdf_norm",
                "bkg_" + prefix + "_pdf_norm",
                ndata_cat,
                -float("inf"),
                float("inf"),
            )
            ws_corepdf.Import(cat_dataSet)
            ws_corepdf.Import(norm_cat)
            ws_corepdf.Import(norm_Core)
            saveWorkspace(ws_corepdf, "workspace_BackgroundFit" + prefix + args.ext)


def generateData(ws, pdfName, tag, cs, lumi):
    return ws.pdf(pdfName + tag).generate(rt.RooArgSet(ws.obj("mass")), cs * lumi)


def saveWorkspace(ws, outputFileName):
    outfile = rt.TFile(outputFileName + ".root", "recreate")
    ws.Write()
    outfile.Close()
    del ws


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
