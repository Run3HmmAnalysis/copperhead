from dask.distributed import Client
import ROOT as rt
import argparse
import dask.dataframe as dd
import pandas as pd
import glob
from fitmodels import *


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
colors = [
    rt.kRed,
    rt.kGreen,
    rt.kBlue,
    rt.kYellow,
    rt.kViolet,
    rt.kGray,
    rt.kOrange,
    rt.kPink,
    rt.kMagenta,
    rt.kAzure,
    rt.kCyan,
    rt.kTeal,
    rt.kSpring,
    rt.kRed + 1,
    rt.kGreen + 1,
    rt.kBlue + 1,
    rt.kYellow + 1,
    rt.kViolet + 1,
    rt.kGray + 1,
    rt.kOrange + 1,
    rt.kPink + 1,
    rt.kMagenta + 1,
]


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
    fittedmodels = {}
    isBlinded = args.isBlinded
    processName = args.process
    category = args.category
    tag = "_"+processName+"_"+category
    lumi = args.intLumi
    ws = createWorkspace()

    if args.doBackgroundFit:
        modelNames = ["bwz_redux_model", "bwg_model"]
        add_data(ws, df, isBlinded)
        ws.Print()
        fixparam = False
        if(isBlinded):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        plotter(ws,["ds"],isBlinded, processName, category, "ds_ggh_MC","MC")
        for fitmodel in modelNames:
            add_model(ws, fitmodel, tag)
        # plotter(ws,["ds","bwz_redux_model"+tag,"bwg_model"+tag], isBlinded, category, "ds_data_bwZreduxmodel","Data and BWZRedux model")        name = "data_BackgroundFit" + args.ext
        title = "Background"
        fit(ws, "ds", modelNames, isBlinded, fixparam, tag, True, name, title)
        #fit(ws, "ds", modelNames, tag, False, name, title)
    # plotter(ws,["ds","bwg_model"+tag], isBlinded, category, "ds_data_bwZreduxmodel","Data and BWZRedux model")
        saveWorkspace(ws, "workspace_BackgroundFit"+tag+args.ext)


    if args.doSignalFit:
        isBlinded = False
        fixparam = True
        modelNames = ["dcb_model"]
        add_data(ws, df, isBlinded)
        if(isBlinded):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #plotter(ws,["ds"],isBlinded, processName, category, "ds_ggh_MC","MC")
        for fitmodel in modelNames:
            add_model(ws, fitmodel, tag)
        # plotter(ws,["ds","bwz_redux_model"+tag,"bwg_model"+tag], isBlinded, category, "ds_data_bwZreduxmodel","Data and BWZRedux model")        name = "data_BackgroundFit" + args.ext
        name = "ggH_SignalFit" + args.ext
        title = "Signal"
        ws.Print()
        fit(ws, "ds", modelNames, isBlinded, fixparam, tag, True, name, title)
        saveWorkspace(ws, "workspace_sigFit"+tag+args.ext)

    if args.getBkgModelFromMC:
        modelNames = ["bwz_redux_model", "bwg_model"]
        fixparam = False
        isBlinded = False
        name = "fake_data_BackgroundFit" + args.ext
        title = "Background"
        for fitmodel in modelNames:
            add_model(ws, fitmodel, tag)
        fake_data = generateData(
            ws, "bwz_redux_model", tag, 100, lumi
        )
        add_data(ws, fake_data, False, "_fake", False)
        ws.Print()
        # plotter(ws,["ds_fake"], False, category, "data_bwZreduxmodel","BWZRedux model fake Data")
        fit(
            ws, "ds_fake", modelNames, isBlinded, fixparam, tag, True, name, title
        )
        saveWorkspace(ws, "workspace_BackgroundFit"+tag+args.ext)


    if args.doCorePdfFit:
        corePDF_results = {}
        hists_All = {}
        nCats=5
        coreModelNames = ["bwz_redux_model"]
        for fitmodel in coreModelNames:
            add_model(ws, fitmodel, "_"+processName+"_corepdf")
        fixparam = False
        isBlinded = False
        name = "fake_data_Background_corPdfFit" + args.ext
        title = "Background"
        #fake_data = rt.RooDataSet("full_dataset", "full_dataset")
        #outputfile_fakedata = rt.TFile("data_histograms_EachCat_and_Full.root","recreate")
        dataStack = rt.THStack("full_data","full_data")
        for icat in range(nCats):
            #tag = "_"+processName+"_cat"+icat
            hist_name = "hist"+"_"+processName+"_cat"+str(icat)
            #hist = rt.TH1F(hist_name,hist_name,80,110.,150.)
            ds = generateData(
                ws, "bwz_redux_model", "_"+processName+"_corepdf", 100, lumi
            )
            hist = rt.RooAbsData.createHistogram(ds,hist_name,ws.var("mass"),rt.RooFit.Binning(80))
            #hist.Write()
            #print(hist.GetName())
            hists_All[hist_name] = hist
            print(hists_All)
            print(hists_All[hist_name].Integral())
            #fake_data.append(ds)
            dataStack.Add(hist)
            add_data(ws, hist, False, hist_name+"_fake", False)
        print(hists_All)
        dataStack_Full = dataStack.GetStack().Last()
        hists_All[dataStack_Full.GetName()] = dataStack_Full
        #dataStack_Full.Write()
        #outputfile_fakedata.Close()
        print(hists_All)
        fullDataSet = rt.RooDataHist("core_Data","core_Data",rt.RooArgList(ws.var("mass")),dataStack_Full)
        add_data(ws, fullDataSet, False, "_Core_fake", False)
        #add_data(ws, fake_data, False, "_fake", False)
        ws.Print()
        # plotter(ws,["ds_fake"], False, category, "data_bwZreduxmodel","BWZRedux model fake Data")
        fit(
            ws, "ds_Core_fake", coreModelNames, isBlinded, fixparam, "_"+processName+"_corepdf", True, name, title
        )
        norm_Core = rt.RooRealVar("bkg_norm_Core", "bkg_norm_Core", fullDataSet.sumEntries(), -float('inf'), float('inf'))
        for icat in range(nCats):
            histName = "hist"+"_"+processName+"_cat"+str(icat)
            ws_corepdf = rt.RooWorkspace("ws_corepdf", False)
            ws_corepdf.Import(ws.pdf("bwz_redux_model"+"_"+processName+"_corepdf"))
            prefix = "cat"+str(icat)
            print(hists_All)
            transferHist = hists_All[histName].Clone()
            transferHist.Divide(dataStack_Full)
            transferHist.Scale( 1 / transferHist.Integral())
            transferDataSet = rt.RooDataHist("transfer_"+prefix,"transfer_"+prefix,rt.RooArgList(ws.var("mass")), transferHist)
            transferDataName = transferDataSet.GetName()
            ws.Import(transferDataSet)
            ws_corepdf.Import(transferDataSet)
            chebyOrder = 3 if icat < 1 else 2
            add_model(ws, "chebychev_"+str(chebyOrder)+"_model", "_"+processName+"_"+prefix)
            #transferFuncName = "chebychev_"+str(chebyOrder)+"_"+processName+"_"+prefix
            transferFuncName = "chebychev"+str(chebyOrder)
            fit(ws, transferDataName, [transferFuncName], isBlinded, fixparam, "_"+processName+"_"+prefix, True, name, title)
            coreBWZRedux = rt.RooProdPdf("bkg_bwzredux_"+"_"+processName+"_"+prefix,"bkg_bwzredux_"+"_"+processName+"_"+prefix, ws.pdf("bwz_redux_model"+"_"+processName+"_corepdf"), ws.pdf(transferFuncName+"_"+processName+"_"+prefix))
            ws_corepdf.Import(coreBWZRedux,rt.RooFit.RecycleConflictNodes())
            cat_dataSet = rt.RooDataHist("data_"+prefix,"data_"+prefix,rt.RooArgList(ws.var("mass")),transferDataSet)
            ndata_cat = cat_dataSet.sumEntries()
            norm_cat = rt.RooRealVar("bkg_"+prefix+"_pdf_norm", "bkg_"+prefix+"_pdf_norm", ndata_cat, -float('inf'), float('inf'))
            ws_corepdf.Import(cat_dataSet)
            ws_corepdf.Import(norm_cat)
            ws_corepdf.Import(norm_Core)
            saveWorkspace(ws_corepdf, "workspace_BackgroundFit"+prefix+args.ext)



def createWorkspace():
    w = rt.RooWorkspace("w", "w")
    fitrange = {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
    mass = rt.RooRealVar("mass", "mass", fitrange["low"], fitrange["high"])
    mass.setRange("unblindReg_left", fitrange["low"], fitrange["SR_left"] + 0.1)
    mass.setRange("unblindReg_right", fitrange["SR_right"] - 0.1, fitrange["high"])
    mass.setRange("window", fitrange["low"], fitrange["high"])
    mass.SetTitle("m_{#mu#mu}")
    mass.setUnit("GeV")
    w.Import(mass)
    w.Print()
    return w


def add_model(workspace, modelName, tag):
    if modelName == "bwz_redux_model":
        bwZredux_model, bwZredux_params = bwZredux(
            workspace.obj("mass"), tag
        )
        workspace.Import(bwZredux_model)
    elif modelName == "bwg_model":
        bwGamma_model, bwGamma_params = bwGamma(
            workspace.obj("mass"), tag
        )
        workspace.Import(bwGamma_model)
    elif modelName == "dcb_model":
        dcb_model, dcb_params = doubleCB(
            workspace.obj("mass"), tag
        )
        workspace.Import(dcb_model)
    elif "chebychev" in modelName:
        chebychev_model, chebychev_params = chebychev(
            workspace.obj("mass"), tag, int(modelName.split("_")[1])
        )
        workspace.Import(chebychev_model)
    else:
        print("The " + modelName + " does not exist!!!!!!!")


def add_data(workspace, data, isBlinded, name="", convertData=True):
    if convertData:
        ds = filldataset(
            data["dimuon_mass"].values, workspace.obj("mass"), dsName="ds" + name
        ) 
        if isBlinded:
            ds = ds.reduce(rt.RooFit.CutRange("unblindReg_left,unblindReg_right"))
        workspace.Import(ds)
    else:
        if isBlinded:
            data = data.reduce(CutRange("unblindReg_left,unblindReg_right"))
        workspace.Import(data, "ds" + name)


def filldataset(data, x, dsName="ds"):
    cols = rt.RooArgSet(x)
    ds = rt.RooDataSet(dsName, dsName, cols)
    for datum in data:
        if (datum < x.getMax()) and (datum > x.getMin()):
            x.setVal(datum)
            ds.add(cols)
    # ds.Print()
    return ds


def fit(ws, dsName, modelName, isBlinded, fixParameters, tag, save, name, title):
    print("In cat", tag.split("_")[2])
    pdfs = {}
    for model in modelName:
        print(model + tag)
        pdfs[model + tag] = ws.pdf(
            model + tag
        )
        if dsName == "ds":            
            result = pdfs[model + tag].fitTo(
                ws.data(dsName), rt.RooFit.Save()
            )                
        else:
            result = pdfs[model + tag].fitTo(
                ws.obj(dsName), rt.RooFit.Save()
            )
        if fixParameters:
            pdfs[model + tag].getParameters(rt.RooArgSet()).setAttribAll("Constant")
            #pdfs[model + tag].getParameters(rt.RooArgSet()).find("mean"+tag).setConstant()
    if save:
        plot(ws, dsName, pdfs, isBlinded, tag.split("_")[1], tag.split("_")[2], name, title)


def generateData(ws, pdfName, tag, cs, lumi):
    return ws.pdf(pdfName + tag).generate(
        rt.RooArgSet(ws.obj("mass")), cs * lumi
    )


def plotter(ws, objNames, isBlinded, processName, category, OutputFilename, title):
    c = rt.TCanvas("c_cat" + category, "c_cat" + category, 800, 800)
    xframe = ws.obj("mass").frame(rt.RooFit.Title(title + " in cat" + category))
    count = 0
    for name in objNames:
        if "model" in name:
            print(name)
            ws.pdf(name).plotOn(
                xframe,
                rt.RooFit.Range("window"),
                rt.RooFit.NormRange("window"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(name),
            )
            count += 1
            # ws.pdf(name).plotOn(xframe,rt.RooFit.Range("window"))
        elif "ds_fake" in name:
            ws.obj(name).plotOn(xframe, rt.RooFit.Binning(80))
        elif "ds" in name:
            ws.data(name).plotOn(xframe, rt.RooFit.Binning(80))
    xframe.Draw()
    c.Update()
    c.SaveAs(OutputFilename + "_cat" + category + ".root")
    c.SaveAs(OutputFilename + "_cat" + category + ".pdf")
    c.SaveAs(OutputFilename + "_cat" + category + ".png")
    c.SaveAs(OutputFilename + "_cat" + category + ".C")


def plot(ws, datasetName, models, isBlinded, processName, category, name, title):
    c = rt.TCanvas("c_cat" + category, "c_cat" + category, 800, 800)
    offset = 0.5
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    mass = ws.obj("mass")
    xframe = mass.frame(rt.RooFit.Title(title + " Fit in cat" + category))
    # dataset.plotOn(xframe,rt.RooFit.CutRange("unblindReg_left"))
    # dataset.plotOn(xframe,rt.RooFit.CutRange("unblindReg_right"))
    if datasetName == "ds":
        ws.data(datasetName).plotOn(xframe, rt.RooFit.Binning(80))
    else:
        ws.obj(datasetName).plotOn(xframe, rt.RooFit.Binning(80))
    leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
    leg0.SetFillStyle(0)
    leg0.SetLineColor(0)
    leg0.SetTextSize(0.03)
    # leg0.AddEntry(h_data,"Data","lep")
    if isBlinded:
        count = 0
        for model_key in models:
            models[model_key].plotOn(
                xframe,
                rt.RooFit.Range("window"),
                rt.RooFit.NormRange("unblindReg_left,unblindReg_right"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(models[model_key].GetName()),
            )
            leg0.AddEntry(
                models[model_key],
                "#splitline{" + models[model_key].GetName() + "}{model}",
                "l",
            )
            count += 1
    else:
        count = 0
        for model_key in models:
            models[model_key].plotOn(
                xframe,
                rt.RooFit.Range("window"),
                #rt.RooFit.NormRange("window"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(models[model_key].GetName()),
            )
            leg0.AddEntry(
                models[model_key],
                "#splitline{" + models[model_key].GetName() + "}{model}",
                "l",
            )
            count += 1
    # upper_pad = rt.TPad("up_cat"+category,"up_cat"+category,0.0,0.2,1.0,1.0,21)
    # lower_pad = rt.TPad("lp_cat"+category,"lp_cat"+category,0.0,0.0,1.0,0.2,22)
    xframe.SetMinimum(0.0001)
    xframe.Draw()
    if "ggH" in name:
        print("Fitting ggH signal")
        # Add TLatex to plot
        for model_key in models:
            h_pdf = models[model_key].createHistogram("h_pdf", mass, rt.RooFit.Binning(80))
        print(h_pdf.GetMaximum())
        effSigma = getEffSigma(h_pdf)
        effSigma_low, effSigma_high = (
            h_pdf.GetMean() - effSigma,
            h_pdf.GetMean() + effSigma,
        )
        h_effSigma = h_pdf.Clone()
        h_effSigma.GetXaxis().SetRangeUser(effSigma_low, effSigma_high)
        h_data = mass.createHistogram("h_data", rt.RooFit.Binning(80))
        lat0 = rt.TLatex()
        lat0.SetTextFont(42)
        lat0.SetTextAlign(11)
        lat0.SetNDC()
        lat0.SetTextSize(0.045)
        lat0.DrawLatex(0.15, 0.92, "#bf{CMS} #it{Simulation}")
        lat0.DrawLatex(0.77, 0.92, "13 TeV")
        lat0.DrawLatex(0.16 + 0.02, 0.83, "H#rightarrow#mu#mu")
        leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
        leg0.SetFillStyle(0)
        leg0.SetLineColor(0)
        leg0.SetTextSize(0.03)
        leg0.AddEntry(h_data, "Simulation", "lep")
        leg0.AddEntry(h_pdf, "#splitline{Double Crystal-Ball}{model}", "l")
        leg0.Draw("Same")

        leg1 = rt.TLegend(0.17 + offset, 0.45, 0.4 + offset, 0.61)
        leg1.SetFillStyle(0)
        leg1.SetLineColor(0)
        leg1.SetTextSize(0.03)
        leg1.AddEntry(
            h_pdf, "#scale[0.8]{#sigma_{eff} = %1.2f GeV}" % getEffSigma(h_pdf), "l"
        )
        # leg1.AddEntry(h_pdf_splitByYear['2017'],"2017: #scale[0.8]{#sigma_{eff} = %1.2f GeV}"%getEffSigma(h_pdf_splitByYear['2017']),"l")
        # leg1.AddEntry(h_pdf_splitByYear['2018'],"2018: #scale[0.8]{#sigma_{eff} = %1.2f GeV}"%getEffSigma(h_pdf_splitByYear['2018']),"l")
        # leg1.Draw("Same")

        leg2 = rt.TLegend(0.15 + offset, 0.3, 0.5 + offset, 0.45)
        leg2.SetFillStyle(0)
        leg2.SetLineColor(0)
        leg2.SetTextSize(0.03)
        leg2.AddEntry(
            h_effSigma,
            "#sigma_{eff} = %1.2f GeV" % (0.5 * (effSigma_high - effSigma_low)),
            "fl",
        )
        leg2.Draw("Same")
        h_effSigma.SetLineColor(15)
        h_effSigma.SetFillStyle(1001)
        h_effSigma.SetFillColor(15)
        h_effSigma.Draw("Same Hist F")
        vline_effSigma_low = rt.TLine(
            effSigma_low,
            0,
            effSigma_low,
            h_pdf.GetBinContent(h_pdf.FindBin(effSigma_low)),
        )
        vline_effSigma_high = rt.TLine(
            effSigma_high,
            0,
            effSigma_high,
            h_pdf.GetBinContent(h_pdf.FindBin(effSigma_high)),
        )
        vline_effSigma_low.SetLineColor(15)
        vline_effSigma_high.SetLineColor(15)
        vline_effSigma_low.SetLineWidth(2)
        vline_effSigma_high.SetLineWidth(2)
        vline_effSigma_low.Draw("Same")
        vline_effSigma_high.Draw("Same")
        fwhm_low = h_pdf.GetBinCenter(h_pdf.FindFirstBinAbove(0.5 * h_pdf.GetMaximum()))
        fwhm_high = h_pdf.GetBinCenter(h_pdf.FindLastBinAbove(0.5 * h_pdf.GetMaximum()))
        fwhmArrow = rt.TArrow(
            fwhm_low,
            0.5 * h_pdf.GetMaximum(),
            fwhm_high,
            0.5 * h_pdf.GetMaximum(),
            0.02,
            "<>",
        )
        fwhmArrow.SetLineWidth(2)
        fwhmArrow.Draw("Same <>")
        fwhmText = rt.TLatex()
        fwhmText.SetTextFont(42)
        fwhmText.SetTextAlign(11)
        fwhmText.SetNDC()
        fwhmText.SetTextSize(0.03)
        fwhmText.DrawLatex(
            0.17 + offset, 0.25, "FWHM = %1.2f GeV" % (fwhm_high - fwhm_low)
        )
    else:
        lat0 = rt.TLatex()
        lat0.SetTextFont(42)
        lat0.SetTextAlign(11)
        lat0.SetNDC()
        lat0.SetTextSize(0.045)
        lat0.DrawLatex(0.15, 0.92, "#bf{CMS} #it{2018D}")
        lat0.DrawLatex(0.77, 0.92, "13 TeV")
        lat0.DrawLatex(0.16 + 0.02, 0.83, "H#rightarrow#mu#mu")
        leg0.Draw("Same")

    hpull = xframe.pullHist()
    lower_pad.cd()
    xframe2 = mass.frame()
    xframe2.SetTitle("")
    xframe2.addPlotable(hpull, "P")
    xframe2.GetYaxis().SetTitle("Pull")
    if isBlinded:
        xframe2.GetYaxis().SetRangeUser(-4,4)
    xframe2.GetYaxis().SetTitleOffset(0.3)
    xframe2.GetYaxis().SetTitleSize(0.08)
    xframe2.GetYaxis().SetLabelSize(0.08)
    xframe2.GetXaxis().SetLabelSize(0.08)
    xframe2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    xframe2.GetXaxis().SetTitleSize(0.08)
    line = rt.TLine(0.1, 0.5, 0.9, 0.5)
    line.SetNDC(rt.kTRUE)
    line.SetLineWidth(2)
    line.Draw()
    xframe2.Draw()
    c.Modified()
    c.Update()
    c.SaveAs(processName + name + "_cat" + category + ".root")
    c.SaveAs(processName + name + "_cat" + category + ".pdf")
    c.SaveAs(processName + name + "_cat" + category + ".png")
    c.SaveAs(processName + name + "_cat" + category + ".C")

def saveWorkspace(ws, outputFileName):
    outfile = rt.TFile(outputFileName+".root","recreate")
    ws.Write()
    outfile.Close()
    del ws
    
def GOF(ws, pdfName, dsName, tag, ndata):
    normalization = rt.RooRealVar("normaliazation", "normalization", ndata, .5 * ndata, 2*ndata)
    model = rt.RooExtendPdf("ext", "ext", ws.pdf(pdfName), norm)
    xframe = ws.var("mass"+tag).frame()
    ds = ws.data(dsName)
    ds.plotOn(xframe, rt.RooFit.Name("ds"))
    model.plotOn(xframe, rt.RooFit.Name("model"))
    nparam = model.getParameters(ds).getSize()
    chi2 = xframe.chiSquare("model", "ds", nparam)
    nBins = ds.numEntries()
    if float(ndata) / nBins < 5:
        ntoys = 500
        print(" can't use asymptotic approximation!! need to run toys")
        prob = getPValue(chi2*(nBins-nparam), nBins-nparam)

    else:
        prob = getPValue(chi2*(nBins-nparam), nBins-nparam)

    print("chi2/ndof = "+str(chi2))
    print("chi2"+str(chi2*(nBins-nparam)))
    print("p-value = "+str(prob))

    return chi2, prob


def getPValue(chi2, ndof):
    prob = rt.TMath.Prob(chi2, ndof)
    return prob


def getEffSigma(_h):
    nbins, binw, xmin = (
        _h.GetXaxis().GetNbins(),
        _h.GetXaxis().GetBinWidth(1),
        _h.GetXaxis().GetXmin(),
    )
    mu, rms, total = _h.GetMean(), _h.GetRMS(), _h.Integral()
    # Scan round window of mean: window RMS/binWidth (cannot be bigger than 0.1*number of bins)
    nWindow = int(rms / binw) if (rms / binw) < 0.1 * nbins else int(0.1 * nbins)
    # Determine minimum width of distribution which holds 0.693 of total
    rlim = 0.683 * total
    wmin, iscanmin = 9999999, -999
    for iscan in range(-1 * nWindow, nWindow + 1):
        # Find bin idx in scan: iscan from mean
        i_centre = int((mu - xmin) / binw + 1 + iscan)
        x_centre = (i_centre - 0.5) * binw + xmin  # * 0.5 for bin centre
        x_up, x_down = x_centre, x_centre
        i_up, i_down = i_centre, i_centre
        # Define counter for yield in bins: stop when counter > rlim
        y = _h.GetBinContent(i_centre)  # Central bin height
        r = y
        reachedLimit = False
        for j in range(1, nbins):
            if reachedLimit:
                continue
            # Up:
            if (i_up < nbins) & (not reachedLimit):
                i_up += 1
                x_up += binw
                y = _h.GetBinContent(i_up)  # Current bin height
                r += y
                if r > rlim:
                    reachedLimit = True
            else:
                print(
                    " --> Reach nBins in effSigma calc: %s. Returning 0 for effSigma"
                    % _h.GetName()
                )
                return 0
            # Down:
            if not reachedLimit:
                if i_down > 0:
                    i_down -= 1
                    x_down -= binw
                    y = _h.GetBinContent(i_down)  # Current bin height
                    r += y
                    if r > rlim:
                        reachedLimit = True
                else:
                    print(
                        " --> Reach 0 in effSigma calc: %s. Returning 0 for effSigma"
                        % _h.GetName()
                    )
                    return 0
        # Calculate fractional width in bin takes above limt (assume linear)
        if y == 0.0:
            dx = 0.0
        else:
            dx = (r - rlim) * (binw / y)
        # Total width: half of peak
        w = (x_up - x_down + binw - dx) * 0.5
        if w < wmin:
            wmin = w
            iscanmin = iscan
        # Return effSigma
        return wmin


if __name__ == "__main__":
    parameters = {"ncpus": 40}
    # paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_*/*.parquet')
    #paths = glob.glob("/depot/cms/hmm/coffea/2016_sep26/data_D/*.parquet")
    paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/ggh_amcPS/*.parquet')

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
