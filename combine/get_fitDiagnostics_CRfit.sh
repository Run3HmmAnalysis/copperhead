echo $1 $2

combine -M FitDiagnostics --saveShapes --saveWithUncertainties $1_$2/combined.root -n $1_$2  --setParameters mask_ch1=1
