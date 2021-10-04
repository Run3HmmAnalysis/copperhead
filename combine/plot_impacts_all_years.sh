dc_name=combined_$1
echo "$dc_name"
text2workspace.py "$dc_name".txt -m 125
combineTool.py -M Impacts -d "$dc_name".root -m 125 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d "$dc_name".root -m 125 --robustFit 1 --doFits
combineTool.py -M Impacts -d "$dc_name".root -m 125 -o impacts.json
plotImpacts.py -i impacts.json -o impacts_$year
