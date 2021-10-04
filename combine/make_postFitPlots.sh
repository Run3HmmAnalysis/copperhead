echo 'combine -M FitDiagnostics $1/combined_datacard$1.root -m 125 --rMin -1 --rMax 2 --saveShapes --saveWithUncertainties'
#combine -M FitDiagnostics $1/combined_datacard$1.root -m 125 --rMin -1 --rMax 2 --saveShapes --saveWithUncertainties -n _$1_new_analytic --plots --robustFit=1 --X-rtd MINIMIZER_analytic
GENERATE='n;toysFrequentist;t;;NAME_prefit_asimov,!,-1;NAME_postfit_asimov, ,-1'
DRYRUN="--dry-run"
JOBOPTS0="+JobFlavour = \"microcentury\"\nrequirements = (OpSysAndVer =?= \"CentOS7\")"
JOBOPTS="$JOBOPTS0"
#combine -M FitDiagnostics $1/combined.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic --robustFit=1 --X-rtd MINIMIZER_analytic --expectSignal=1 --robustHesse 1 --setParameters mask_ch1=1

#Last used
combine -M FitDiagnostics $1/combined_datacard$1.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -5 --rMax 5  --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic$2 --X-rtd MINIMIZER_analytic --expectSignal=1 --setParameters mask_ch1=1

#According to gitlab
#combine -M FitDiagnostics $1/combined_datacard$1.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -5 --rMax 5  --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic_gitlab --X-rtd MINIMIZER_analytic --expectSignal=1 --skipBOnlyFit --robustHesse 1 #--setParameters mask_ch1=1

#combine -M FitDiagnostics $1/combined.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -2 --rMax 2 --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic --plots --robustFit=1 --X-rtd MINIMIZER_analytic --job-mode condor --generate "${GENERATE//NAME/$name}" --task-name FitDiagnostics_${dc##*/} --sub-opts="$JOBOPTS\n+JobBatchName=\"FitDiagnostics\"" $DRYRUN --expectSignal=1  --skipBOnlyFit --robustHesse 1
echo 'python postfitplot.py'
#python postfitplot_AP.py
