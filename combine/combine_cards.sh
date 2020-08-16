mkdir 2016 2017 2018 allyears

combineCards.py $1/2016_$2/datacard_vbf_SR.txt $1/2016_$2/datacard_vbf_SB.txt > 2016/combined_datacard2016.txt
combineCards.py $1/2017_$2/datacard_vbf_SR.txt $1/2017_$2/datacard_vbf_SB.txt > 2017/combined_datacard2017.txt
combineCards.py $1/2018_$2/datacard_vbf_SR.txt $1/2018_$2/datacard_vbf_SB.txt > 2018/combined_datacard2018.txt
combineCards.py $1/2016_$2/datacard_vbf_SR.txt $1/2016_$2/datacard_vbf_SB.txt $1/2017_$2/datacard_vbf_SR.txt $1/2017_$2/datacard_vbf_SB.txt $1/2018_$2/datacard_vbf_SR.txt $1/2018_$2/datacard_vbf_SB.txt > allyears/combined_datacardallyears.txt
