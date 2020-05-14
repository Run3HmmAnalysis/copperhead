echo $1 $2

combineCards.py $1_$2/datacard_vbf_h-peak.txt $1_$2/datacard_vbf_h-sidebands.txt > $1_$2/combined.txt
text2workspace.py $1_$2/combined.txt --channel-masks
