year='2018'
combineCards.py datacard_test_m125_SR_"$year".txt datacard_test_m125_CR_"$year".txt > dc_"$year"_combined.txt
text2workspace.py dc_"$year"_combined.txt --channel-masks
