## Instructions for the ggH channel

This README provides a step-by-step guide on how to perform the stage3 of the ggH analysis. Starting from producing workspaces, and datacards to performing the maximum likelihood fit, bias study, goodness of fit test, and finally producing significance using HiggsCombine for the ggH channel are describen in this readme.<br/>

**Prerequisites**:<br/>
Before proceeding, ensure you have prepared a CMSSW environment with following:<br/>
HiggsCombine (Follow the instrucitons here: [HiggsCombine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit))<br/>
ROOT framework<br/>



### Goodness of fit study

Steps:<br/>

1. Create a workspace combining all the datacards for the 5 categories using the command:<br/>
        `combineCards.py category1=datacard1.txt category2=datacard2.txt category3=datacard3.txt category4=datacard4.txt category5=datacard5.txt > combined_datacard.txt`<br/>

2. Generate the workspace using the command:<br/>
	`text2workspace.py combined_datacard.txt -o combined_workspace.root`<br/>

3. Run the goodness of fit test using the command:<br/>
    	`combine -M GoodnessOfFit combined_workspace.root --algo=saturated -m 125 --freezeParameters MH -n .goodnessOfFit_data`<br/>
    	`combine -M GoodnessOfFit combined_workspace.root --algo=saturated -m 125 --freezeParameters MH -n .goodnessOfFit_toys -t 1000`<br/>

4. Obtain the p-value from the output.<br/>
   	 `combineTool.py -M CollectGoodnessOfFit --input higgsCombine.goodnessOfFit_data.GoodnessOfFit.mH125.root higgsCombine.goodnessOfFit_toys.GoodnessOfFit.mH125.123456.root -m 125.0 -o gof.json`<br/>

	 `plotGof.py gof.json --statistic saturated --mass 125.0 -o part2_gof`<br/>
5. Interpret the results based on the p-value. A p-value greater than 0.05 indicates a good fit, while a p-value less than or equal to 0.05 indicates a poor fit.<br/>

6. Conclusion<br/>
   This guide provided a simple method to perform the goodness of fit test using the saturated Chi-square method in Higgs Combine for 5 categories with their corresponding datacards.<br/>