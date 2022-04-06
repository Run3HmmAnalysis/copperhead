import ROOT as rt

rt.gSystem.Load("lib/RooDoubleCB/RooDoubleCB")


def linear(x, tag):
    m = rt.RooRealVar("slope" + tag, "slope", -0.33, -10, 0)
    b = rt.RooRealVar("offset" + tag, "offset", 15, 2, 1000)

    linear_model = rt.RooGenericPdf(
        "linear" + tag, "@1*(@0-140)+@2", rt.RooArgList(x, m, b)
    )
    return linear_model, [m, b]


# --------------------------------------------------------
# breit weigner for photons scaled by falling exp
# no breit weigner for the Z
# --------------------------------------------------------
def bwGamma(x, tag):
    expParam = rt.RooRealVar("bwg_expParam" + tag, "expParam", -1e-03, -1e-01, 1e-01)
    bwmodel = rt.RooGenericPdf(
        "bwgamma" + tag, "exp(@0*@1)*pow(@0,-2)", rt.RooArgList(x, expParam)
    )

    return bwmodel, [expParam]


# --------------------------------------------------------
# breit weigner Z scaled by falling exp
# no mixture, no photon contribution
# --------------------------------------------------------
def bwZ(x, tag):
    bwWidth = rt.RooRealVar("bwz_Width" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwz_mZ" + tag, "mZ", 91.2, 90, 92)
    expParam = rt.RooRealVar("bwz_expParam" + tag, "expParam", -1e-03, -1e-02, 1e-02)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    bwmodel = rt.RooGenericPdf(
        "bwz" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        rt.RooArgList(x, bwmZ, bwWidth, expParam),
    )
    return bwmodel, [bwWidth, bwmZ, expParam]


# --------------------------------------------------------
# breit weigner mixture scaled by falling exp (run1 bg)
# --------------------------------------------------------
def bwZGamma(x, tag, mix_min=0.001):
    bwWidth = rt.RooRealVar("bwzg_Width" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwzg_mZ" + tag, "mZ", 91.2, 90, 92)

    expParam = rt.RooRealVar(
        "bwzg_expParam" + tag, "expParam", -0.0053, -0.0073, -0.0033
    )
    mixParam = rt.RooRealVar("bwzg_mixParam" + tag, "mix", 0.379, 0.2, 1)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    phoExpMmumu = rt.RooGenericPdf(
        "phoExpMmumu" + tag, "exp(@0*@1)*pow(@0,-2)", rt.RooArgList(x, expParam)
    )
    bwExpMmumu = rt.RooGenericPdf(
        "bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        rt.RooArgList(x, bwmZ, bwWidth, expParam),
    )
    bwmodel = rt.RooAddPdf(
        "bwzgamma" + tag,
        "bwzgamma",
        rt.RooArgList(bwExpMmumu, phoExpMmumu),
        rt.RooArgList(mixParam),
    )

    return bwmodel, [bwWidth, bwmZ, expParam, mixParam, phoExpMmumu, bwExpMmumu]


# ----------------------------------------
# perturbed exponential times bwz
# with an off power for the breit weigner
# ----------------------------------------
def bwZredux(x, tag):
    a1 = rt.RooRealVar("bwz_redux_a1" + tag, "a1", 1.39, 0.7, 2.1)
    a2 = rt.RooRealVar("bwz_redux_a2" + tag, "a2", 0.46, 0.30, 0.62)
    a3 = rt.RooRealVar("bwz_redux_a3" + tag, "a3", -0.26, -0.40, -0.12)

    # a1.setConstant()
    # a2.setConstant()
    # a3.setConstant()

    f = rt.RooFormulaVar(
        "bwz_redux_f" + tag, "(@1*(@0/100)+@2*(@0/100)^2)", rt.RooArgList(x, a2, a3)
    )
    # expmodel = RooGenericPdf("bwz_redux", "exp(@2)*(2.5)/(pow(@0-91.2,@1)+0.25*pow(2.5,@1))", RooArgList(x, a1, f))
    expmodel = rt.RooGenericPdf(
        "bwz_redux" + tag,
        "bwz_redux",
        "exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))",
        rt.RooArgList(x, a1, f),
    )
    return expmodel, [a1, a2, a3, f]


# ----------------------------------------
# perturbed exponential times bwz
# with an off power for the breit weigner
# ----------------------------------------
def bwZreduxFixed(x, tag):
    a1 = rt.RooRealVar("bwz_redux_fixed_a1" + tag, "a1", 2.0, 0.7, 2.1)
    a2 = rt.RooRealVar("bwz_redux_fixed_a2" + tag, "a2", 0.36, 0.0, 50.0)
    a3 = rt.RooRealVar("bwz_redux_fixed_a3" + tag, "a3", -0.36, -50.0, 0)
    bwmZ = rt.RooRealVar("bwz_redux_fixed_mZ" + tag, "mZ", 91.2, 89, 93)
    w = rt.RooRealVar("bwz_redux_fixed_w" + tag, "w", 2.5, 0, 10)

    a1.setConstant()
    # a2.setConstant()
    # a3.setConstant()
    bwmZ.setConstant()
    w.setConstant()

    f = rt.RooFormulaVar(
        "bwz_redux_fixed_f" + tag,
        "(@1*(@0/100)+@2*(@0/100)^2)",
        rt.RooArgList(x, a2, a3),
    )
    # expmodel = RooGenericPdf("bwz_redux", "exp(@2)*(2.5)/(pow(@0-91.2,@1)+0.25*pow(2.5,@1))", RooArgList(x, a1, f))
    expmodel = rt.RooGenericPdf(
        "bwz_redux_fixed" + tag,
        "bwz_redux_fixed",
        "exp(@2)*(2.5)/(pow(@0-@3,@1)+pow(@4/2,@1))",
        rt.RooArgList(x, a1, f, bwmZ, w),
    )
    return expmodel, [a1, a2, a3, f, bwmZ, w]


# ----------------------------------------
# hgg falling exponential
# ----------------------------------------
def higgsGammaGamma(x, tag):
    a1 = rt.RooRealVar("hgg_a1" + tag, "a1", -5, -1000, 1000)
    a2 = rt.RooRealVar("hgg_a2" + tag, "a2", -5, -1000, 1000)
    one = rt.RooRealVar("hgg_one" + tag, "one", 1.0, -10, 10)
    one.setConstant()

    # a1.setConstant(True)

    f = rt.RooFormulaVar(
        "hgg_f" + tag, "@1*(@0/100)+@2*(@0/100)^2", rt.RooArgList(x, a1, a2)
    )
    expmodel = rt.RooExponential("hggexp" + tag, "hggexp", f, one)  # exp(1*f(x))

    return expmodel, [a1, a2, one, f]


# ----------------------------------------
# chebyshev
# ----------------------------------------
def chebyshev(x, tag, order=7):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)
    # c1 = RooRealVar("c1","c1", 1.0,-1.0,1.0)
    # c2 = RooRealVar("c2","c2", 1.0,-1.0,1.0)

    args = rt.RooArgList()
    params = []
    for i in range(0, order):
        c = rt.RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2**i, -1.0, 1.0)
        args.add(c)
        params.append(c)

    chebyshev = rt.RooChebychev(f"chebyshev{order}" + tag, f"chebyshev{order}", x, args)
    return chebyshev, params


# ----------------------------------------
# bernstein
# ----------------------------------------
def bernstein(x, tag, order=5):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)
    # c1 = RooRealVar("c1","c1", 1.0,-1.0,1.0)
    # c2 = RooRealVar("c2","c2", 1.0,-1.0,1.0)

    args = rt.RooArgList()
    params = []
    for i in range(0, order):
        c = rt.RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2**i, -1.0, 1.0)
        args.add(c)
        params.append(c)

    bernstein = rt.RooBernstein(f"bernstein{order}" + tag, f"bernstein{order}", x, args)
    return bernstein, params


# ----------------------------------------
# h2mupoly
# ----------------------------------------
def h2mupoly(x, tag, order=5):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = rt.RooArgList()
    args.add(x)
    params = []
    poly_str = ""
    for i in range(0, order):
        c = rt.RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2**i, -1.0, 1.0)
        args.add(c)
        params.append(c)
        if i == 0:
            poly_str += "(@%d)^2" % (i + 1)
        else:
            poly_str += "+ pow(@%d,2)*((160-@0)/50)^%d" % ((i + 1), i)

    # print "h2mupoly = "+poly_str

    h2mupoly = rt.RooGenericPdf(
        "h2mu" + tag + f"poly{order}", f"h2mupoly{order}", poly_str, args
    )
    return h2mupoly, params


# ----------------------------------------
# h2mupolyf
# ----------------------------------------
def h2mupolyf(x, tag, order=10):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = rt.RooArgList()
    args.add(x)
    params = []
    poly_str = ""
    for i in range(0, order):
        c = rt.RooRealVar("c" + str(i) + tag, "c" + str(i), 1.0 / 2, -1.0, 1.0)
        args.add(c)
        params.append(c)
        if i == 0:
            poly_str += "(@%d)^2" % (i + 1)
        else:
            poly_str += "+ pow(@%d,2)*sqrt(pow((160-@0)/50,%d))" % ((i + 1), i)

    # print "h2mupolyf = "+poly_str

    h2mupolyf = rt.RooGenericPdf(
        "h2mu" + tag + f"poly{order}", f"h2mupoly{order}", poly_str, args
    )
    return h2mupolyf, params


# ----------------------------------------
# h2mupolypow
# ----------------------------------------
def h2mupolypow(x, tag, order=6):
    # c0 = RooRealVar("c0","c0", 1.0,-1.0,1.0)

    args = rt.RooArgList()
    args.add(x)
    params = []
    poly_str = ""

    ic = 1
    ib = 2
    for o in range(0, order):
        c = rt.RooRealVar("c" + str(o) + tag, "c" + str(o), 1.0 / 2, -1.0, 1.0)
        b = rt.RooRealVar("b" + str(o) + tag, "b" + str(o), 1.0 / 2, -3.14, 3.14)
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

    h2mupolypow = rt.RooGenericPdf(
        "h2mu" + tag + f"polypow{order}", f"h2mupolypow{order}", poly_str, args
    )
    return h2mupolypow, params


# --------------------------------------------------------
# breit weigner scaled by falling exp, then add a line
# for ttbar
# --------------------------------------------------------
def bwZPlusLinear(x, tag):
    bwWidth = rt.RooRealVar("bwzl_widthZ" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwzl_mZ" + tag, "mZ", 91.2, 85, 95)
    expParam = rt.RooRealVar("bwzl_expParam" + tag, "expParam", -1e-03, -1e-01, 1e-01)

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    slopeParam = rt.RooRealVar("bwzl_slope" + tag, "slope", -0.2, -50, 0)
    offsetParam = rt.RooRealVar("bwzl_offset" + tag, "offset", 39, 0, 1000)

    mix1 = rt.RooRealVar("bwzl_mix1" + tag, "mix1", 0.95, 0, 1)

    linMmumu = rt.RooGenericPdf(
        "bwzl_linMmumu" + tag, "@1*@0+@2", rt.RooArgList(x, slopeParam, offsetParam)
    )
    bwExpMmumu = rt.RooGenericPdf(
        "bwzl_bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        rt.RooArgList(x, bwmZ, bwWidth, expParam),
    )
    model = rt.RooAddPdf(
        "bwzl" + tag, "bwzl", rt.RooArgList(bwExpMmumu, linMmumu), rt.RooArgList(mix1)
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
    bwWidth = rt.RooRealVar("bwzgl_widthZ" + tag, "widthZ", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwzgl_mZ" + tag, "mZ", 91.2, 85, 95)
    expParam = rt.RooRealVar(
        "bwzgl_expParam" + tag, "expParam", -0.0053, -0.0073, -0.0033
    )

    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    slopeParam = rt.RooRealVar("bwl_slope" + tag, "slope", -0.2, -50, 0)
    offsetParam = rt.RooRealVar("bwl_offset" + tag, "offset", 39, 0, 1000)

    mix1 = rt.RooRealVar("bwzgl_mix1" + tag, "mix1", 0.10, 0.01, 0.20)
    mix2 = rt.RooRealVar("bwzgl_mix2" + tag, "mix2", 0.39, 0.1, 1)

    expParam.setConstant(True)
    mix1.setConstant(True)
    mix2.setConstant(True)

    linMmumu = rt.RooGenericPdf(
        "bwzgl_linMmumu" + tag, "@1*@0+@2", rt.RooArgList(x, slopeParam, offsetParam)
    )
    phoExpMmumu = rt.RooGenericPdf(
        "bwzgl_phoExpMmumu" + tag, "exp(@0*@1)*pow(@0,-2)", rt.RooArgList(x, expParam)
    )
    bwExpMmumu = rt.RooGenericPdf(
        "bwzgl_bwExpMmumu" + tag,
        "exp(@0*@3)*(@2)/(pow(@0-@1,2)+0.25*pow(@2,2))",
        rt.RooArgList(x, bwmZ, bwWidth, expParam),
    )
    model = rt.RooAddPdf(
        "bwzgl" + tag,
        "bwl",
        rt.RooArgList(linMmumu, bwExpMmumu, phoExpMmumu),
        rt.RooArgList(mix1, mix2),
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
    mean = rt.RooRealVar("mean" + tag, "mean", 125.0, 120.0, 130.0)
    sigma = rt.RooRealVar("sigma" + tag, "sigma", 2, 0.0, 5.0)
    alpha1 = rt.RooRealVar("alpha1" + tag, "alpha1", 2, 0.001, 25)
    n1 = rt.RooRealVar("n1" + tag, "n1", 1.5, 0, 25)
    alpha2 = rt.RooRealVar("alpha2" + tag, "alpha2", 2.0, 0.001, 25)
    n2 = rt.RooRealVar("n2" + tag, "n2", 1.5, 0, 25)
    model = rt.RooDoubleCB("dcb" + tag, "dcb", x, mean, sigma, alpha1, n1, alpha2, n2)
    return model, [mean, sigma, alpha1, n1, alpha2, n2]
