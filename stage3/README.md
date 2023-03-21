##Goodness of Fit using Chi-square in Higgs Combine

This README provides a step-by-step guide on how to perform the goodness of fit test using the Chi-square method in Higgs Combine for 5 categories with their corresponding datacards.
Prerequisites

Before proceeding, ensure you have the following:

    Higgs Combine
    ROOT framework

Steps:

Create a workspace combining all the datacards for the 5 categories using the command:
        combineCards.py category1=datacard1.txt category2=datacard2.txt category3=datacard3.txt category4=datacard4.txt category5=datacard5.txt > combined_datacard.txt

Generate the workspace using the command:
	text2workspace.py combined_datacard.txt -o combined_workspace.root

Run the goodness of fit test using the command:
    	combine -M GoodnessOfFit combined_workspace.root --algo=saturated

Obtain the p-value from the output.

Interpret the results based on the p-value. A p-value greater than 0.05 indicates a good fit, while a p-value less than or equal to 0.05 indicates a poor fit.

Conclusion

This guide provided a simple method to perform the goodness of fit test using the Chi-square method in Higgs Combine for 5 categories with their corresponding datacards.