year='2018'
#combine -M GoodnessOfFit -d dc_"$year"_combined.root --algo=saturated -n "$year"_CRfit_bonly --setParametersForFit mask_ch1=1 --setParametersForEval mask_ch1=0 --freezeParameters r --setParameters r=0
 
combine -M FitDiagnostics --saveShapes --saveWithUncertainties dc_"$year"_combined.root -n "$year"_CRfit  --setParameters mask_ch1=1
