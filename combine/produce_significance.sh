echo "Prefit Significance:" >> $1_significance.txt
combineTool.py -d $1/combined_datacard$1.txt -M Significance -m 125 --expectSignal=1 -n _$1_ -t -1 --rMin -2 --rMax 5 > $1_prefitsignificance.log
cat $1_prefitsignificance.log | grep "Significance:" >> $1_significance.txt
echo "Postfit Significance:" >> $1_significance.txt
combineTool.py -d $1/combined_datacard$1.txt -M Significance -m 125 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams --expectSignal=1 -n _$1_ -t -1 --toysFrequentist --rMin -2 --rMax 5 > $1_postfitsignificance.log
cat $1_postfitsignificance.log | grep "Significance:" >> $1_significance.txt
