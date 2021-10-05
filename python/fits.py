from dask.distributed import Client
import ROOT as rt
import dask.dataframe as dd
import pandas as pd
import glob
from fitmodels import bwZredux
#from models import *
#from utils import filldataset

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
    df_future = [d[['dimuon_mass']] for d in df_future]
    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    df = df.compute()
    df.reset_index(inplace=True, drop=True)
    fitrange = {"low": 110,
                "high": 150,
                "SR_left": 120,
                "SR_right": 130
    }
    #modelNames = ["bwZRedux"]
    background_fit(df,fitrange,"0",0)

def filldataset(data, x, dsName = 'ds'):
    cols = rt.RooArgSet(x)
    ds = rt.RooDataSet(dsName, dsName, cols)
    #ds.Print()
    for datum in data:
        if (datum < x.getMax()) and (datum > x.getMin()):
            x.setVal(datum)
            ds.add(cols)
    #ds.Print()
    return ds

def background_fit(column, fitrange, category, blinded=True, save=True):
    #print(column)
    print("In cat",category)
    mass = rt.RooRealVar("mass","mass",fitrange["low"],fitrange["high"])
    model, params = bwZredux(mass)
    mass.setRange("window",fitrange["low"],fitrange["high"])
    if blinded:
        mass.setRange("unblindReg_left",fitrange["low"],fitrange["SR_left"]+0.1)
        mass.setRange("unblindReg_right",fitrange["SR_right"]-0.1,fitrange["high"])
    ds = filldataset(column['dimuon_mass'].values, mass, dsName = 'ds')
    #ds.Print()
    if blinded:
        result = model.fitTo(ds,rt.RooFit.Range("unblindReg_left,unblindReg_right"), rt.RooFit.Save())
    else:
        result = model.fitTo(ds, rt.RooFit.Save())
    if save:
        plot(mass,ds,model,category)

def plot(mass, dataset, model, category):
    c = rt.TCanvas("c_cat"+category,"c_cat"+category,800,800)
    upper_pad = rt.TPad("upper_pad","upper_pad",0,0.25,1,1);
    lower_pad = rt.TPad("lower_pad","lower_pad",0,0,1,0.35);
    upper_pad.SetBottomMargin(0.14);
    lower_pad.SetTopMargin(0.00001);
    lower_pad.SetBottomMargin(0.25);
    upper_pad.Draw();
    lower_pad.Draw();
    upper_pad.cd();
    xframe = mass.frame(rt.RooFit.Title("Background Fit in cat"+category))
    #dataset.plotOn(xframe,rt.RooFit.CutRange("unblindReg_left"))
    #dataset.plotOn(xframe,rt.RooFit.CutRange("unblindReg_right"))
    dataset.plotOn(xframe)
    model.plotOn(xframe,rt.RooFit.Range("window"),rt.RooFit.Name(model.GetName()))
    #upper_pad = rt.TPad("up_cat"+category,"up_cat"+category,0.0,0.2,1.0,1.0,21)
    #lower_pad = rt.TPad("lp_cat"+category,"lp_cat"+category,0.0,0.0,1.0,0.2,22)
    xframe.SetMinimum(0.0001)
    xframe.Draw()
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
    c.SaveAs("data2018D_invmass_Fit_cat"+category+".png")
    

if __name__ == '__main__':
    parameters = {'ncpus': 40}
    #paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_*/*.parquet')
    paths = glob.glob('/depot/cms/hmm/coffea/2016_sep26/data_D/*.parquet')

    client = Client(
        processes=True,
        n_workers=parameters['ncpus'],
        threads_per_worker=1,
        memory_limit='4GB'
    )

    workflow(client, paths, parameters)
