import numpy as np


def add_pdf_variations(do_pdf, df, year, dataset, parameters, output, weights):
    if "2016" in year:
        max_replicas = 0
        if "dy" in dataset:
            max_replicas = 100
        elif "ewk" in dataset:
            max_replicas = 33
        else:
            max_replicas = 100
        if do_pdf:
            pdf_wgts = df.LHEPdfWeight[:, 0 : parameters["n_pdf_variations"]]
        for i in range(100):
            if (i < max_replicas) and do_pdf:
                output[f"pdf_mcreplica{i}"] = pdf_wgts[:, i]
            else:
                output[f"pdf_mcreplica{i}"] = np.nan
    else:
        if do_pdf:
            pdf_wgts = df.LHEPdfWeight[:, 0 : parameters["n_pdf_variations"]][0]
            pdf_wgts = np.array(pdf_wgts)
            pdf_vars = {
                "up": (1 + 2 * pdf_wgts.std()),
                "down": (1 - 2 * pdf_wgts.std()),
            }
            weights.add_weight("pdf_2rms", pdf_vars, how="only_vars")
        else:
            weights.add_weight("pdf_2rms", how="dummy_vars")
