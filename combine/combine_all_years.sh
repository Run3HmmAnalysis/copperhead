echo $1

combineCards.py 2016_$1/datacard_vbf_h-peak.txt 2016_$1/datacard_vbf_h-sidebands.txt 2017_$1/datacard_vbf_h-peak.txt 2017_$1/datacard_vbf_h-sidebands.txt 2018_$1/datacard_vbf_h-peak.txt 2018_$1/datacard_vbf_h-sidebands.txt  > combined_$1.txt
text2workspace.py combined_$1.txt --channel-masks
