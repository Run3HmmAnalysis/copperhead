import multiprocessing as mp
from math import sqrt
import tqdm


def dynamic_binning(df, model, args):
    var = f"score_{model}"
    df = df[[var, "cls", "r", "year", "wgt_nominal"]].copy()
    parallel = True
    N = 1000
    min_score = 0
    max_score = 2.3
    bin_width = (max_score - min_score) / N
    if args["year"] == "":
        years = ["2016", "2017", "2018"]
    else:
        years = [args["year"]]
    binning = {}

    s = (
        []
    )  # will store the best significance for the category containing bins i through j
    best_splitting = (
        []
    )  # will store the best way to split the category containing bins i through j
    # Initialization
    for i in range(N):
        row = []
        row_bs = []
        for j in range(N):
            row.append(0)
            row_bs.append([])
        s.append(row)
        best_splitting.append(row_bs)

    def callback(a):
        #        pbar.update()
        i, j, s_ij, best_splitting_ij = a
        s[i][j] = s_ij
        best_splitting[i][j] = best_splitting_ij

    for year in years:
        binning[year] = []
        for l in tqdm.tqdm(range(1, N + 1)):
            subproblem_args = {
                "df": df,
                "var": var,
                "year": year,
                "l": l,
                "s": s,
                "N": N,
                "min_score": min_score,
                "max_score": max_score,
                "best_splitting": best_splitting,
            }
            if parallel:
                #                pbar = tqdm.tqdm(total=N-l+1)
                pool = mp.Pool(mp.cpu_count() - 2)
                a = [
                    pool.apply_async(
                        subproblem,
                        args=(i, i + l - 1, subproblem_args, False),
                        callback=callback,
                    )
                    for i in range(0, N - l + 1)
                ]
                for process in a:
                    process.wait()
                    # i, j, s_ij, best_splitting_ij = process.get()
                    # s[i][j] = s_ij
                    # best_splitting[i][j] = best_splitting_ij
                pool.close()

            else:  # if not parallel
                print(f"Solving subproblems of size {l}")
                for i in range(0, N - l + 1):  # j = i+l-1
                    i, j, s_ij, best_splitting_ij = subproblem(
                        i, i + l - 1, subproblem_args, True
                    )
                    s[i][j] = s_ij
                    best_splitting[i][j] = best_splitting_ij
            # print("S_ij so far:")
            # for ii in range(N):
            #    row = ""
            #    for jj in range(N):
            #        row = row + "%f "%s[ii][jj]
            #    print(row)
        binning[year] = [
            round(min_score + ibin * bin_width, 3) for ibin in best_splitting[0][N - 1]
        ]
        print(f"Best combined significance: {s[0][N-1]}")
        nom_sign = get_significance_(args["mva_bins"][model][year], subproblem_args)
        print(f"Nominal significance (Pisa bins): {nom_sign}")
    return binning


def get_significance(binning, args):
    df = args["df"]
    var = args["var"]
    significance2 = 0
    bin_width = (args["max_score"] - args["min_score"]) / args["N"]
    sig = df[(df.cls == "signal") & (df.r == "h-peak") & (df.year == int(args["year"]))]
    bkg = df[
        (df.cls == "background") & (df.r == "h-peak") & (df.year == int(args["year"]))
    ]
    for ibin in range(len(binning) - 1):
        bin_lo = args["min_score"] + binning[ibin] * bin_width
        bin_hi = args["min_score"] + binning[ibin + 1] * bin_width
        sig_yield = sig[(sig[var] >= bin_lo) & (sig[var] < bin_hi)]["wgt_nominal"].sum()
        bkg_yield = bkg[(bkg[var] >= bin_lo) & (bkg[var] < bin_hi)]["wgt_nominal"].sum()
        significance2 += (sig_yield * sig_yield) / (sig_yield + bkg_yield)
    return sqrt(significance2)


def get_significance_(binning, args):
    df = args["df"]
    var = args["var"]
    significance2 = 0
    sig = df[(df.cls == "signal") & (df.r == "h-peak") & (df.year == int(args["year"]))]
    bkg = df[
        (df.cls == "background") & (df.r == "h-peak") & (df.year == int(args["year"]))
    ]
    for ibin in range(len(binning) - 1):
        bin_lo = binning[ibin]
        bin_hi = binning[ibin + 1]
        sig_yield = sig[(sig[var] >= bin_lo) & (sig[var] < bin_hi)]["wgt_nominal"].sum()
        bkg_yield = bkg[(bkg[var] >= bin_lo) & (bkg[var] < bin_hi)]["wgt_nominal"].sum()
        significance2 += (sig_yield * sig_yield) / (sig_yield + bkg_yield)
    return sqrt(significance2)


def subproblem(i, j, args, debug):
    df = args["df"]
    var = args["var"]
    year = args["year"]
    l = args["l"]
    s = args["s"]
    N = args["N"]
    bin_width = (args["max_score"] - args["min_score"]) / args["N"]
    best_splitting = args["best_splitting"]
    # for N=50:
    # penalty = 1 # for 2017 set to 1.4
    # for N=100:
    penalty = 1.2
    if debug:
        print(f"Solving subproblem P_{i}{j}!")
    s_ij = 0
    best_splitting_ij = []
    for k in range(i, j + 1):
        consider_this_option = True
        can_decrease_nCat = False
        if k == i:
            bins = [
                i,
                j + 1,
            ]  # here the numbers count not bins, but boundaries between bins, hence j+1
            significance = get_significance(bins, args)
            s_ij = significance
            best_splitting_ij = bins
            if debug:
                print(f"First approach to P_{i}{j}: merge all bins.")
                print(f"Merge bins from #{i} to #{j} into a single category")
                b = bins_to_illustration(i, j + 1, bins)
                print(f"Splitting is {b}")
                print(
                    f"Calculated significance for merged bins! Significance = {round(significance,3)}"
                )

        else:
            # sorted union of lists will provide the correct category boundaries
            bins = sorted(
                list(set(best_splitting[i][k - 1]) | set(best_splitting[k][j]))
            )
            significance = sqrt(s[i][k - 1] * s[i][k - 1] + s[k][j] * s[k][j])
            if s_ij:
                gain = (significance - s_ij) / s_ij * 100.0
            else:
                gain = 999

            if debug:
                print(f"Continue solving P_{i}{j}!")
                print(f"Cut between bins #{k-1} and #{k}")
                print(f"Combine the optimal solutions of P_{i}{k-1} and P_{k}{j}")
                b = bins_to_illustration(i, j + 1, bins)
                print(f"Splitting is {b}")
                b = bins_to_illustration(i, j + 1, best_splitting_ij)
                print(
                    f"Before this option the best s[{i}][{j}] was {round(s_ij,3)} for splitting {b}"
                )
                print(
                    f"This option gives s[{i}][{j}] = {round(significance,3)} for splitting {bins}"
                )
                print(f"We gain {round(gain,2)} %% if we use the new option.")
                print(
                    f"The required gain if {round(penalty,2)} %% per additional category."
                )
            ncat_diff = abs(len(bins) - len(best_splitting_ij))

            if (len(bins) > len(best_splitting_ij)) & (gain < penalty * ncat_diff):
                if debug:
                    print(
                        f"This option increases number of subcategories by {len(bins)-len(best_splitting_ij)} from {len(best_splitting_ij)-1} to {len(bins)-1}, but the improvement is just {round(gain,2)} %%, so skip."
                    )
                consider_this_option = False

            elif (len(bins) < len(best_splitting_ij)) & (gain > -penalty * ncat_diff):
                if debug:
                    print(
                        f"This option decreases number of subcategories by {len(best_splitting_ij)-len(bins)} from {len(best_splitting_ij)-1} to {len(bins)-1}, and the change in significance is just {-round(gain,2)} %%, so keep it."
                    )
                can_decrease_nCat = True

            elif (len(bins) == len(best_splitting_ij)) & (gain > 0):
                if debug:
                    print(
                        f"This option keeps the same number of categories as the bes option so far, and the significance is increased by {round(gain,2)} %%, so keep it."
                    )

            if ((gain > 0) & (consider_this_option)) or can_decrease_nCat:
                s_ij = significance
                best_splitting_ij = bins
                if debug:
                    print(
                        f"Updating best significance: now s[{i}][{j}] = {round(s_ij,3)}"
                    )
            else:
                if debug:
                    print("Not updating best significance.")

    if debug:
        print(f"Problem P_{i}{j} solved! Here's the best solution:")
        b = bins_to_illustration(i, j + 1, best_splitting_ij)
        print(
            f"Highest significance for P_{i}{j} is {round(s_ij,3)} and achieved when the splitting is {b}"
        )
        boundaries = [
            args["min_score"] + ibin * bin_width for ibin in best_splitting_ij
        ]
        print(f"Corresponding binning is {boundaries}")
        print("-" * 30)
    s[i][j] = s_ij
    best_splitting[i][j] = best_splitting_ij
    return i, j, s_ij, best_splitting_ij


def bins_to_illustration(min, max, bins):
    result = ""
    for iii in range(min, max):
        if iii in bins:
            result = result + "| "
        result = f"{result} {iii}"
    result = result + "| "
    return result
