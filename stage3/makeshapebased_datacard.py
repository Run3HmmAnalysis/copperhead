# from uncertainty import *
import json


def buildDatacard(
    ws, bkgmodel, sigmodel, mass, tag, Channels, uncert_file, datacardpath
):
    fout = open(
        datacardpath
        + "/datacard_%s_%s_%s_%s_%s.txt"
        % (mass, sigmodel, bkgmodel, tag.split("_")[1], tag.split("_")[2]),
        "w",
    )
    fout.write("imax *\n")
    fout.write("jmax *\n")
    fout.write("kmax *\n")
    fout.write(("-" * 40) + "\n")
    for Channel in Channels:
        # print("shapes "+Channel+"_hmm "+tag.split("_")[2]+"_"+tag.split("_")[1]+" %s w:"% ("w.root")+Channel+"_cat0_ggh_pdf\n")
        fout.write(
            "shapes "
            + Channel
            + "_hmm cat"
            + tag.split("_")[2]
            + "_"
            + tag.split("_")[1]
            + " %s w:"
            % (
                "workspace_%s_%s_%s_cat%s.txt"
                % (mass, sigmodel, tag.split("_")[1], tag.split("_")[2])
            )
            + Channel
            + "_cat"
            + tag.split("_")[2]
            + "_"
            + tag.split("_")[1]
            + "_pdf\n"
        )
    fout.write(
        "shapes bkg cat"
        + tag.split("_")[2]
        + "_"
        + tag.split("_")[1]
        + " %s w:bkg_cat"
        % (
            "workspace_%s_%s_%s_cat%s.txt"
            % (mass, bkgmodel, tag.split("_")[1], tag.split("_")[2])
        )
        + tag.split("_")[2]
        + "_"
        + tag.split("_")[1]
        + "_pdf\n"
    )
    fout.write(
        "shapes data_obs cat"
        + tag.split("_")[2]
        + "_"
        + tag.split("_")[1]
        + " %s w:data_cat"
        % (
            "workspace_%s_%s_%s_cat%s.txt"
            % (mass, bkgmodel, tag.split("_")[1], tag.split("_")[2])
        )
        + tag.split("_")[2]
        + "_"
        + tag.split("_")[1]
        + "\n"
    )
    fout.write(("-" * 40) + "\n")
    fout.write("bin cat%s\n" % (tag.split("_")[2] + "_" + tag.split("_")[1]))
    fout.write("observation -1\n")
    fout.write(("-" * 40) + "\n")
    binstr = "bin "
    p1str = "process "
    p2str = "process "
    ratestr = "rate "
    isig = 1
    for Channel in Channels:
        processName = Channel
        binstr += "cat%s " % (tag.split("_")[2] + "_" + tag.split("_")[1])
        p1str += "%s_hmm " % processName
        p2str += "%d " % (-len(Channels) + isig)
        ratestr += "1 "
        isig += 1

    binstr += "cat%s\n" % (tag.split("_")[2] + "_" + tag.split("_")[1])
    p1str += "bkg\n"
    p2str += "1\n"
    ratestr += "1\n"

    unc_file = open(uncert_file)
    uncertainties = json.load(unc_file)
    uncStrings = []
    for unc in uncertainties:
        uncstr = unc + " " + uncertainties[unc]["type"]
        for sName in Channels:
            processName = sName
            for key in uncertainties[unc].keys():
                if processName in key:
                    uncstr += " %s" % uncertainties[unc][processName]
                if "bkg" in key:
                    uncstr += " %s" % uncertainties[unc]["bkg"]

        # append
        uncStrings.append(uncstr)

    #    binstr = "bin  %s  %s  %s\n" % (category, category, category)
    #    p1str = "process  %s  %s  %s\n" % ("smodel1", "smodel2", "bmodel")
    #    p2str = "process  -1  0  1\n"
    #    ratestr = "rate  1  1  1\n"
    fout.write(binstr)
    fout.write(p1str)
    fout.write(p2str)
    fout.write(ratestr)
    fout.write(("-" * 40) + "\n")
    for x in uncStrings:
        fout.write("%s\n" % x)
    fout.close()
