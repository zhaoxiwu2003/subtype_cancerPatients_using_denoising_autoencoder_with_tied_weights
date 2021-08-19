# cancer_subtyping

This is the new upgrade for this paper: https://www.nature.com/articles/s41598-019-53048-x. The current codes can be run using CPU and GPU (using tensorflow 2.0).

The purpose of the project is to stratify the cancer patients into clinical associated subgroups to assist the personalized diagnosis and treatment. The algorithm used in this study is denoising autoencoders with tied weights. We used correlation coefficient as our second evaluation metric.

To obtain the optimal parameters for the architecture of this model, we tested a serial of parameters in a wide range: the learning rate at 0.01, 0.005, 0.001, 0.0005, and 0.0001; the corruption level at 0.9, 0.7, 0.5, 0.3 and 0.1.

The first program (training_cancer_type_with_tiedWeightsAE.py) used in this project is to encode the five omics data (gene, miRNA and protein expression, CNA and methylation). The script can automatically select the proper architecture based on the scale of the input omics data. Different parameters were examined. The best model can be selected based on the training loss, training correlation coefficient, test loss or test correlation coefficient. In this project, the test loss was used.

The outputs (encoded data) of the first program were then combined to input to the second program (combined_data_with_tied_weights.py). Different model parameters were examined again. The best model were selected based on the test loss. The final encoded data (output of combined_data_with_tied_weights.py) can be grouped by K-means to subtype the patients. 

All the affiliated functions and classes are in utils_cancer_type.py.

If you meet any problems or have any comments, please contact zhao xi wu 2003 at gmail . com (no space).

