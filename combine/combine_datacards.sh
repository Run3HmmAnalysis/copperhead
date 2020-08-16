echo $1 $2
cd $1_$2
combineCards.py datacard_vbf_SR.txt datacard_vbf_SB.txt > combined.txt
text2workspace.py combined.txt --channel-masks
cd -
