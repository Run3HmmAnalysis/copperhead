mass_points=(1250 1251 1252 1253 1254 1255 1256 1257 1258 1259 1260)
for m in "${mass_points[@]}"; do
    path=massScan/$m/$1_$2/
    echo $path
    cd $path
    combineCards.py datacard_vbf_SR.txt datacard_vbf_SB.txt > combined.txt
    text2workspace.py combined.txt --channel-masks
    cd -
done

for m in "${mass_points[@]}"; do
    path=massScan/$m/$1_$2/
    echo $path
    cd $path
#    combine -M Significance combined.txt -t -1 --expectSignal=1
#    combine -M Significance combined.txt
    cd -
done
