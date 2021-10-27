
lbl_global=jul7
#lbl=jul7_score_dnn_allyears_128_64_32
lbl=jul7_score_bdt_nest10000_weightCorrAndShuffle_2Aug
cwd=$(pwd)

cd massScan_"$lbl_global"
for dir in ./*;
do
    cd $dir
    massPoint=${dir##*/}
    for year in 2016 2017 2018;
    do
	cd "$year"_"$lbl"
	name=combined_"$massPoint"_"$year".txt
	name_nominal=combined_"$massPoint"_"$year"_nominal.txt
	combineCards.py datacard_vbf_SR.txt datacard_vbf_SB.txt > $name
	combineCards.py datacard_vbf_SR_nominal.txt datacard_vbf_SB_nominal.txt > $name_nominal
	ls
	cd ../
    done
    combineCards.py 2016_"$lbl"/combined_"$massPoint"_2016.txt 2017_"$lbl"/combined_"$massPoint"_2017.txt 2018_"$lbl"/combined_"$massPoint"_2018.txt > combined_"$lbl"_"$massPoint".txt
    combineCards.py 2016_"$lbl"/combined_"$massPoint"_2016_nominal.txt 2017_"$lbl"/combined_"$massPoint"_2017_nominal.txt 2018_"$lbl"/combined_"$massPoint"_2018_nominal.txt > combined_"$lbl"_"$massPoint"_nominal.txt
    ls
    cd ../
done
cd ../
