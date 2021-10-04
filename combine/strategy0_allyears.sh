#label=jul7_score_dnn_allyears_128_64_32
#label=jul7_score_bdt_jul15_earlystop50
#label=jul7_score_bdt_nest10000_bestmodel_31July
label=jul7_score_bdt_nest10000_weightCorrAndShuffle_2Aug
combineCards.py 2016_$label/combined.txt 2017_$label/combined.txt 2018_$label/combined.txt > combined.txt

text2workspace.py combined.txt --channel-masks

combineTool.py -M Impacts -d combined.root --doInitialFit -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic

combineTool.py -M Impacts -d combined.root --doFits -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic

combineTool.py -M Impacts -d combined.root -m 125 --autoBoundsPOIs r --autoMaxPOIs r --cminDefaultMinimizerStrategy 0 --saveWorkspace --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 -t -1 --toysFrequentist --expectSignal 1 --X-rtd MINIMIZER_analytic -o impacts.json

#plotImpacts.py -i impacts.json -o impacts_dnn_aug10
plotImpacts.py -i impacts.json -o impacts_bdt_aug11
