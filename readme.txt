# cancer_subtyping

This is the new upgrade for this paper: https://www.nature.com/articles/s41598-019-53048-x

The purpose of the project is to stratify the cancer patients into clinical associated subgroups to assist the personalized diagnosis and treatment. The algorithm used in this study is autoencoders with tied weights. We used correlation coefficient as our second metric.

To obtain the optimal parameters for the architecture of this model, we tested a serial of parameters in a wide range: the learning rate at 0.01, 0.005, 0.001, 0.0005, and 0.0001; the corruption level at 0.9, 0.7, 0.5, 0.3 and 0.1; the batch size at 10, 20, 50 and 100.

The first program used in this project is training_cancer_type_with_tiedWeightsAE.py. In the output of this code (result), you will choose what 5 neck data will be used for the second program (combined_data_with_tied_weights.py) according to the loss, correlation coefficient, test loss or test correlation coefficient. All the functions and classes can be called from utils_cancer_type.py.

The final result (output of combined_data_with_tied_weights.py) will be used of a K-means to subtype the data, which you can get from the paper link.
