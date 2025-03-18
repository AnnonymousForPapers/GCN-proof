# GCN-proof
Code for experiment in Derivation of Back-propagation for Graph Convolutional Networks using Matrix Calculus and its Application to Explainable Artificial Intelligence

## Element-wise backpropagation for binary node classification on heterophilic graph
The code for the experiment of element-wise backpropagation is in Code/Node classification/5-layer GCN/(2-class) (loop) (einsum) GCN_Heterophilic_Random_Kronecker_compare.py

To change the number of nodes of the hetelophilic graph, change the number on the right of 
```
num_nodes = 
```
at line 61. The averaged computational time and memory usage will be shown in the output.

## Matrix-based backpropagation for binary node classification on KarateClub graph using 5-layer GCN
For Figure 1 in the supplementary file, the code is located at Code/Node classification/5-layer GCN/(2-class) (loop) GCN_KarateClub_Kronecker_compare.py

## Matrix-based backpropagation for link prediction on drug-drug interaction network using 5-layer GCN
For Figure 2 in the supplementary file, the code is located at Code/Link prediction/5-layer GCN/(small_general_loop) Kronecker_vs_auto_5_layer.py
To run (small_general_loop) Kronecker_vs_auto_5_layer.py, the user needs to download the DDI_100_nodes.pkl located at Code/Link prediction/ and replace the string on the right-hand side of the equal sign at the 10th line of the code to the file location of your downloaded DDI_100_nodes.pkl.

## Matrix-based backpropagation for binary node classification on KarateClub graph using 1-layer GCN
For Figure 1 (a), Figure 2, Figure 3 (a), and Figure 4 (a), the code is located at Code/Node classification/1-layer GCN/2 class of GCN_KarateClub_feature.ipynb

## Matrix-based backpropagation for link prediction on drug-drug interaction network using 2-layer GCN
For Figure 1 (b), Figure 3 (b), Figure 4 (b), Figure 5, the code is located at Code/Node classification/2-layer GCN/Small_sens_of_Link_prediction_Kronecker_vs_auto_drug.ipynb
The file DDI_100_nodes.pkl located at Code/Link prediction/ needs to be uploaded when runing Small_sens_of_Link_prediction_Kronecker_vs_auto_drug.ipynb.
