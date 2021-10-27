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
    "--doBackgroundFit", default=True, help="Do you want to run bkg fit?"
)
parser.add_argument(
    "--doSignalFit", default=False, help="Do you want to run signal fit?"
)
parser.add_argument(
    "--generateBkgFromMC", default=False, help="Do you want to generate bkg from MC?"
)
parser.add_argument("--intLumi", default=3000, help="Integrated Luminosity")
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
    modelNames = ["bwz_redux_model", "bwg_model"]
    # modelNames = ["bwg_model"]
    fittedmodels = {}
    isBlinded = args.isBlinded
    processName = args.process
    category = args.category
    lumi = args.intLumi
    name = "data_BackgroundFit" + args.ext
    title = "Background"
    ws = createWorkspace(isBlinded)
    add_data(ws, df)
    ws.Print()
    # plotter(ws,["ds"],"0", "ds_data","Data")
    for fitmodel in modelNames:
        add_model(ws, fitmodel, processName, category)
        # background_fit(ws, fitmodel, "0", 1 , 0)
    # plotter(ws,["ds","bwz_redux_model"+processName+"_"+category,"bwg_model"+processName+"_"+category],category, "ds_data_bwZreduxmodel","Data and BWZRedux model")
    if args.doBackgroundFit:
        background_fit(ws, "ds", modelNames, processName, category, True, name, title)
    # plotter(ws,["ds","bwg_model"+processName+"_"+category],category, "ds_data_bwZreduxmodel","Data and BWZRedux model")
    if args.doSignalFit:
        signal_fit(df, fitrange, "0")
    if args.generateBkgFromMC:
        name = "fake_data_BackgroundFit" + ext
        fake_data = generateData(
            ws, "bwz_redux_model", processName, category, 100, lumi
        )
        add_data(ws, fake_data, "_fake", False)
        ws.Print()
        # plotter(ws,["ds_fake"], category, "data_bwZreduxmodel","BWZRedux model fake Data")
        background_fit(
            ws, "ds_fake", modelNames, processName, category, True, name, title
        )


def createWorkspace(blinded=True):
    w = rt.RooWorkspace("w", "w")
    fitrange = {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
    mass = rt.RooRealVar("mass", "mass", fitrange["low"], fitrange["high"])
    mass.setRange("window", fitrange["low"], fitrange["high"])
    if blinded:
        mass.setRange("unblindReg_left", fitrange["low"], fitrange["SR_left"] + 0.1)
        mass.setRange("unblindReg_right", fitrange["SR_right"] - 0.1, fitrange["high"])
    mass.SetTitle("m_{#mu#mu}")
    mass.setUnit("GeV")
    getattr(w, "import")(mass)
    w.Print()
    return w


def add_model(workspace, modelName, processName, category):
    if modelName == "bwz_redux_model":
        bwZredux_model, bwZredux_params = bwZredux(
            workspace.obj("mass"), processName, category
        )
        getattr(workspace, "import")(bwZredux_model)
    elif modelName == "bwg_model":
        bwGamma_model, bwGamma_params = bwGamma(
            workspace.obj("mass"), processName, category
        )
        getattr(workspace, "import")(bwGamma_model)
    else:
        print("The " + modelName + " does not exist!!!!!!!")


def add_data(workspace, data, name="", convertData=True):
    if convertData:
        ds = filldataset(
            data["dimuon_mass"].values, workspace.obj("mass"), dsName="ds" + name
        )
        getattr(workspace, "import")(ds)
    else:
        getattr(workspace, "import")(data, "ds" + name)


def filldataset(data, x, dsName="ds"):
    cols = rt.RooArgSet(x)
    ds = rt.RooDataSet(dsName, dsName, cols)
    for datum in data:
        if (datum < x.getMax()) and (datum > x.getMin()):
            x.setVal(datum)
            ds.add(cols)
    # ds.Print()
    return ds


def background_fit(ws, dsName, modelName, processName, category, save, name, title):
    print("In cat", category)
    pdfs = {}
    for model in modelName:
        print(model + processName + "_" + category)
        pdfs[model + processName + "_" + category] = ws.pdf(
            model + processName + "_" + category
        )
        if dsName == "ds":
            result = pdfs[model + processName + "_" + category].fitTo(
                ws.data(dsName), rt.RooFit.Save()
            )
        else:
            result = pdfs[model + processName + "_" + category].fitTo(
                ws.obj(dsName), rt.RooFit.Save()
            )
    if save:
        plot(ws, dsName, pdfs, processName, category, name, title)


def signal_fit(column, fitrange, category, save=True):
    print(column)
    print("In cat", category)
    model, params = doubleCB(mass)
    mass.setRange("window", fitrange["low"], fitrange["high"])
    ds = filldataset(column["dimuon_mass"].values, mass, dsName="ds")
    # ds.Print()
    result = model.fitTo(ds, rt.RooFit.Save())
    if save:
        name = "ggH_signalFit"
        title = "Signal"
        plot(mass, ds, model, category, name, title)


def generateData(ws, pdfName, processName, category, cs, lumi):
    return ws.pdf(pdfName + processName + "_" + category).generate(
        rt.RooArgSet(ws.obj("mass")), cs * lumi
    )


def plotter(ws, objNames, category, OutputFilename, title):
    c = rt.TCanvas("c_cat" + category, "c_cat" + category, 800, 800)
    xframe = ws.obj("mass").frame(rt.RooFit.Title(title + " in cat" + category))
    count = 0
    for name in objNames:
        if "model" in name:
            print(name)
            ws.pdf(name).plotOn(
                xframe,
                rt.RooFit.Range("window"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(name),
            )
            count += 1
            # ws.pdf(name).plotOn(xframe,rt.RooFit.Range("window"))
        elif "ds_fake" in name:
            ws.obj(name).plotOn(xframe)
        elif "ds" in name:
            ws.data(name).plotOn(xframe)
    xframe.Draw()
    c.Update()
    c.SaveAs(OutputFilename + "_cat" + category + ".root")
    c.SaveAs(OutputFilename + "_cat" + category + ".pdf")
    c.SaveAs(OutputFilename + "_cat" + category + ".png")
    c.SaveAs(OutputFilename + "_cat" + category + ".C")


def plot(ws, datasetName, models, processName, category, name, title):
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
        ws.data(datasetName).plotOn(xframe)
    else:
        ws.obj(datasetName).plotOn(xframe)
    leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
    leg0.SetFillStyle(0)
    leg0.SetLineColor(0)
    leg0.SetTextSize(0.03)
    # leg0.AddEntry(h_data,"Data","lep")
    count = 0
    for model_key in models:
        models[model_key].plotOn(
            xframe,
            rt.RooFit.Range("window"),
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
        h_pdf = model.createHistogram("h_pdf", mass, rt.RooFit.Binning(80))
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
    paths = glob.glob("/depot/cms/hmm/coffea/2016_sep26/data_D/*.parquet")
    # paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/ggh_amcPS/*.parquet')

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
