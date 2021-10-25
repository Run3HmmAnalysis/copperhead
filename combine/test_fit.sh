tolerance=0.01

combineTool.py -M Impacts -d $1_$2/combined.root --doInitialFit -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance $tolerance --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic

combineTool.py -M Impacts -d $1_$2/combined.root --doFits -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance $tolerance --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic

combineTool.py -M Impacts -d $1_$2/combined.root -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance $tolerance --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic -o impacts.json

plotImpacts.py -i impacts.json -o impacts_test_$1_$2_aug18
