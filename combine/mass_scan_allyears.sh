mass_points=(1250 1251 1252 1253 1254 1255 1256 1257 1258 1259 1260)
#mass_points=(1256 1257 1258 1259 1260)
for m in "${mass_points[@]}"; do
    path=massScan/$m/
    echo $path
    cd $path
#    combineCards.py 2016_$1/combined.txt 2017_$1/combined.txt 2018_$1/combined.txt > allyears.txt
#    text2workspace.py allyears.txt --channel-masks
    cd -
done

for m in "${mass_points[@]}"; do
    path=massScan/$m/
    echo $path
    cd $path
    combine -M Significance allyears.txt -t -1 --expectSignal=1
    #combine -M Significance allyears.txt
    cd -
done

