from ROOT import (
    gSystem,
    RooRealVar,
    RooGenericPdf,
    RooArgList,
    RooAddPdf,
    RooFormulaVar,
    RooExponential,
    RooChebychev,
    RooBernstein,
    RooDoubleCB,
)


def linear(x, tag):
    m = RooRealVar("slope" + tag, "slope", -0.33, -10, 0)
    b = RooRealVar("offset" + tag, "offset", 15, 2, 1000)

    linear_model = RooGenericPdf(
        "linear_model" + tag, "@1*(@0-140)+@2", RooArgList(x, m, b)
    )
    return linear_model, [m, b]


# --------------------------------------------------------
# breit weigner for photons scaled by falling exp
# no breit weigner for the Z
# --------------------------------------------------------
def bwGamma(x, tag):
    expParam = RooRealVar("bwg_expParam" + tag, "expParam", -1e-03, -1e-01, 1e-01)
    bwmodel = RooGenericPdf(
        "bwg_model" + tag, "exp(@0*@1)*pow(@0,-2)", RooArgList(x, expParam)
    )

    return bwmodel, [expParam]


# --------------------------------------------------------
# breit weigner Z scaled by falling exp
# no mixture, no photon contribution
# --------------------------------------------------------
def bwZ(x, tag):
    bwWidth = RooRealVar("bwz_Width" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = RooRealVar("bwz_mZ" + tag, "mZ", 91.2, 90, 92)
    expParam = RooRealVar("bwz_expParam" + tag, "expParam", -1e-03, -1e-02, 1e-02)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    bwmodel = RooGenericPdf(
        "bwz_model" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        RooArgList(x, bwmZ, bwWidth, expParam),
    )
    return bwmodel, [bwWidth, bwmZ, expParam]


# --------------------------------------------------------
# breit weigner mixture scaled by falling exp (run1 bg)
# --------------------------------------------------------
def bwZGamma(x, tag, mix_min=0.001):
    bwWidth = RooRealVar("bwzg_Width" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = RooRealVar("bwzg_mZ" + tag, "mZ", 91.2, 90, 92)

    expParam = RooRealVar("bwzg_expParam" + tag, "expParam", -0.0053, -0.0073, -0.0033)
    mixParam = RooRealVar("bwzg_mixParam" + tag, "mix", 0.379, 0.2, 1)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    phoExpMmumu = RooGenericPdf(
        "phoExpMmumu" + tag, "exp(@0*@1)*pow(@0,-2)", RooArgList(x, expParam)
    )
    bwExpMmumu = RooGenericPdf(
        "bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        RooArgList(x, bwmZ, bwWidth, expParam),
    )
    bwmodel = RooAddPdf(
        "bwzg_model" + tag,
        "bwzg_model",
        RooArgList(bwExpMmumu, phoExpMmumu),
        RooArgList(mixParam),
    )

    return bwmodel, [bwWidth, bwmZ, expParam, mixParam, phoExpMmumu, bwExpMmumu]


# ----------------------------------------
# perturbed exponential times bwz
# with an off power for the breit weigner
# ----------------------------------------
def bwZredux(x, tag):
    a1 = RooRealVar("bwz_redux_a1" + tag, "a1", 1.39, 0.7, 2.1)
    a2 = RooRealVar("bwz_redux_a2" + tag, "a2", 0.46, 0.30, 0.62)
    a3 = RooRealVar("bwz_redux_a3" + tag, "a3", -0.26, -0.40, -0.12)

    # a1.setConstant()
    # a2.setConstant()
    # a3.setConstant()

    f = RooFormulaVar(
        "bwz_redux_f" + tag, "(@1*(@0/100)+@2*(@0/100)^2)", RooArgList(x, a2, a3)
    )
    # expmodel = RooGenericPdf("bwz_redux_model", "exp(@2)*(2.5)/(pow(@0-91.2,@1)+0.25*pow(2.5,@1))", RooArgList(x, a1, f))
    expmodel = RooGenericPdf(
        "bwz_redux_model" + tag,
        "bwz_redux_model",
        "exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))",
        RooArgList(x, a1, f),
    )
    return expmodel, [a1, a2, a3, f]


# ----------------------------------------
# perturbed exponential times bwz
# with an off power for the breit weigner
# ----------------------------------------
def bwZreduxFixed(x, tag):
    a1 = RooRealVar("bwz_redux_fixed_a1" + tag, "a1", 2.0, 0.7, 2.1)
    a2 = RooRealVar("bwz_redux_fixed_a2" + tag, "a2", 0.36, 0.0, 50.0)
    a3 = RooRealVar("bwz_redux_fixed_a3" + tag, "a3", -0.36, -50.0, 0)
    bwmZ = RooRealVar("bwz_redux_fixed_mZ" + tag, "mZ", 91.2, 89, 93)
    w = RooRealVar("bwz_redux_fixed_w" + tag, "w", 2.5, 0, 10)

    a1.setConstant()
    # a2.setConstant()
    # a3.setConstant()
    bwmZ.setConstant()
    w.setConstant()

    f = RooFormulaVar(
        "bwz_redux_fixed_f" + tag, "(@1*(@0/100)+@2*(@0/100)^2)", RooArgList(x, a2, a3)
    )
    # expmodel = RooGenericPdf("bwz_redux_model", "exp(@2)*(2.5)/(pow(@0-91.2,@1)+0.25*pow(2.5,@1))", RooArgList(x, a1, f))
    expmodel = RooGenericPdf(
        "bwz_redux_fixed_model" + tag,
        "bwz_redux_fixed_model",
        "exp(@2)*(2.5)/(pow(@0-@3,@1)+pow(@4/2,@1))",
        RooArgList(x, a1, f, bwmZ, w),
    )
    return expmodel, [a1, a2, a3, f, bwmZ, w]


# ----------------------------------------
# hgg falling exponential
# ----------------------------------------
def higgsGammaGamma(x, tag):
    a1 = RooRealVar("hgg_a1" + tag, "a1", -5, -1000, 1000)
    a2 = RooRealVar("hgg_a2" + tag, "a2", -5, -1000, 1000)
    one = RooRealVar("hgg_one" + tag, "one", 1.0, -10, 10)
    one.setConstant()

    # a1.setConstant(True)

    f = RooFormulaVar("hgg_f" + tag, "@1*(@0/100)+@2*(@0/100)^2", RooArgList(x, a1, a2))
    expmodel = RooExponential(
        "hggexp_model" + tag, "hggexp_model", f, one
    )  # exp(1*f(x))

    return expmodel, [a1, a2, one, f]


# ----------------------------------------
# chebychev
# ----------------------------------------
def chebychev(x, tag, order=7):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)
    # c1 = RooRealVar("c1","c1", 1.0,-1.0,1.0)
    # c2 = RooRealVar("c2","c2", 1.0,-1.0,1.0)

    args = RooArgList()
    params = []
    for i in range(0, order):
        c = RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2 ** i, -1.0, 1.0)
        args.add(c)
        params.append(c)

    chebychev = RooChebychev(
        "chebychev" + str(order) + tag, "chebychev" + str(order), x, args
    )
    return chebychev, params


# ----------------------------------------
# bernstein
# ----------------------------------------
def bernstein(x, tag, order=5):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)
    # c1 = RooRealVar("c1","c1", 1.0,-1.0,1.0)
    # c2 = RooRealVar("c2","c2", 1.0,-1.0,1.0)

    args = RooArgList()
    params = []
    for i in range(0, order):
        c = RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2 ** i, -1.0, 1.0)
        args.add(c)
        params.append(c)

    bernstein = RooBernstein(
        "bernstein" + str(order) + tag, "bernstein" + str(order), x, args
    )
    return bernstein, params


# ----------------------------------------
# h2mupoly
# ----------------------------------------
def h2mupoly(x, tag, order=5):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = RooArgList()
    args.add(x)
    params = []
    poly_str = ""
    for i in range(0, order):
        c = RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2 ** i, -1.0, 1.0)
        args.add(c)
        params.append(c)
        if i == 0:
            poly_str += "(@%d)^2" % (i + 1)
        else:
            poly_str += "+ pow(@%d,2)*((160-@0)/50)^%d" % ((i + 1), i)

    # print "h2mupoly = "+poly_str

    h2mupoly = RooGenericPdf(
        "h2mu" + tag + "poly%d" % order, "h2mupoly%d" % order, poly_str, args
    )
    return h2mupoly, params


# ----------------------------------------
# h2mupolyf
# ----------------------------------------
def h2mupolyf(x, tag, order=10):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = RooArgList()
    args.add(x)
    params = []
    poly_str = ""
    for i in range(0, order):
        c = RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2, -1.0, 1.0)
        args.add(c)
        params.append(c)
        if i == 0:
            poly_str += "(@%d)^2" % (i + 1)
        else:
            poly_str += "+ pow(@%d,2)*sqrt(pow((160-@0)/50,%d))" % ((i + 1), i)

    # print "h2mupolyf = "+poly_str

    h2mupolyf = RooGenericPdf(
        "h2mu" + tag + "polyf%d" % order, "h2mupolyf%d" % order, poly_str, args
    )
    return h2mupolyf, params


# ----------------------------------------
# h2mupolypow
# ----------------------------------------
def h2mupolypow(x, tag, order=6):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = RooArgList()
    args.add(x)
    params = []
    poly_str = ""

    ic = 1
    ib = 2
    for o in range(0, order):
        c = RooRealVar("c" + str(o) + tag, "c" + str(o), 1.0 / 2, -1.0, 1.0)
        b = RooRealVar("b" + str(o) + tag, "b" + str(o), 1.0 / 2, -3.14, 3.14)
        args.add(c)
        args.add(b)
        params.append(c)
        params.append(b)
        if o == 0:
            poly_str += "(@%d)^2" % (ic)
        else:
            poly_str += (
                "+ TMath::Power(@%d,2)*TMath::Power((160-@0)/50,%d+(cos(@%d))^2)"
                % (ic, o, ib)
            )

        ic += 2
        ib += 2

    # print "h2mupolypow = "+poly_str

    h2mupolypow = RooGenericPdf(
        "h2mu" + tag + "polypow%d" % order, "h2mupolypow%d" % order, poly_str, args
    )
    return h2mupolypow, params


# --------------------------------------------------------
# breit weigner scaled by falling exp, then add a line
# for ttbar
# --------------------------------------------------------
def bwZPlusLinear(x, tag):
    bwWidth = RooRealVar("bwzl_widthZ" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = RooRealVar("bwzl_mZ" + tag, "mZ", 91.2, 85, 95)
    expParam = RooRealVar("bwzl_expParam" + tag, "expParam", -1e-03, -1e-01, 1e-01)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    slopeParam = RooRealVar("bwzl_slope" + tag, "slope", -0.2, -50, 0)
    offsetParam = RooRealVar("bwzl_offset" + tag, "offset", 39, 0, 1000)

    mix1 = RooRealVar("bwzl_mix1" + tag, "mix1", 0.95, 0, 1)

    linMmumu = RooGenericPdf(
        "bwzl_linMmumu" + tag, "@1*@0+@2", RooArgList(x, slopeParam, offsetParam)
    )
    bwExpMmumu = RooGenericPdf(
        "bwzl_bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        RooArgList(x, bwmZ, bwWidth, expParam),
    )
    model = RooAddPdf(
        "bwzl_model" + tag,
        "bwzl_model",
        RooArgList(bwExpMmumu, linMmumu),
        RooArgList(mix1),
    )

    return (
        model,
        [bwWidth, bwmZ, expParam, mix1, slopeParam, offsetParam, bwExpMmumu, linMmumu],
    )


# --------------------------------------------------------------------
# breit weigner mixture (z + photons) scaled by falling exp (run1 bg)
# then add a line for ttbar
# --------------------------------------------------------------------
def bwZGammaPlusLinear(x, tag):
    bwWidth = RooRealVar("bwzgl_widthZ" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = RooRealVar("bwzgl_mZ" + tag, "mZ", 91.2, 85, 95)
    expParam = RooRealVar("bwzgl_expParam" + tag, "expParam", -0.0053, -0.0073, -0.0033)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    slopeParam = RooRealVar("bwl_slope" + tag, "slope", -0.2, -50, 0)
    offsetParam = RooRealVar("bwl_offset" + tag, "offset", 39, 0, 1000)

    mix1 = RooRealVar("bwzgl_mix1" + tag, "mix1", 0.10, 0.01, 0.20)
    mix2 = RooRealVar("bwzgl_mix2" + tag, "mix2", 0.39, 0.1, 1)

    expParam.setConstant(True)
    mix1.setConstant(True)
    mix2.setConstant(True)

    linMmumu = RooGenericPdf(
        "bwzgl_linMmumu" + tag, "@1*@0+@2", RooArgList(x, slopeParam, offsetParam)
    )
    phoExpMmumu = RooGenericPdf(
        "bwzgl_phoExpMmumu" + tag, "exp(@0*@1)*pow(@0,-2)", RooArgList(x, expParam)
    )
    bwExpMmumu = RooGenericPdf(
        "bwzgl_bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        RooArgList(x, bwmZ, bwWidth, expParam),
    )
    model = RooAddPdf(
        "bwzgl_model" + tag,
        "bwl_model",
        RooArgList(linMmumu, bwExpMmumu, phoExpMmumu),
        RooArgList(mix1, mix2),
    )

    return (
        model,
        [
            bwWidth,
            bwmZ,
            expParam,
            mix1,
            mix2,
            slopeParam,
            offsetParam,
            phoExpMmumu,
            bwExpMmumu,
            linMmumu,
        ],
    )


def doubleCB(x, tag):
    # gSystem.Load("libHiggsAnalysisCombinedLimit")
    gSystem.Load("python/RooDoubleCB/RooDoubleCB")
    mean = RooRealVar("mean" + tag, "mean", 125.0, 120.0, 130.0)
    sigma = RooRealVar("sigma" + tag, "sigma", 2, 0.0, 5.0)
    alpha1 = RooRealVar("alpha1" + tag, "alpha1", 2, 0.001, 25)
    n1 = RooRealVar("n1" + tag, "n1", 1.5, 0, 25)
    alpha2 = RooRealVar("alpha2" + tag, "alpha2", 2.0, 0.001, 25)
    n2 = RooRealVar("n2" + tag, "n2", 1.5, 0, 25)
    model = RooDoubleCB(
        "dcb_model" + tag, "dcb_model", x, mean, sigma, alpha1, n1, alpha2, n2
    )
    return model, [mean, sigma, alpha1, n1, alpha2, n2]
