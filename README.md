# SSGraphDTI
SSGraphDTI:Operating at the intersection of sequential structural data and network-level interaction insights of drug-target pairs, SSGraphDTI orchestrates a harmonious fusion to drive DTI predictions. On the one hand, the model harnesses a CNN-based architecture to distill sequence features, encapsulating the intrinsic structural characteristics. On the other hand, the incorporation of a GNN-based model adeptly captures the dynamic network attributes of interaction networks linked to the input drug-target pairs. This strategic amalgamation of sequential particulars and network dynamics imparts heightened effectiveness to the feature set, thereby elevating the model's predictive performance.

## SSGraphDTI

<div align="center">
<p><img src="model.jpg" width="600" /></p>
</div>

## Setup and dependencies 

Dependencies:
- python 3.9.16
- pytorch >=1.12
- pyg	2.2.0
- rdkit	2022.9.5
- numpy
- sklearn
- tqdm
- tensorboardX
- prefetch_generator
- matplotlib

## Resources:
+ README.md: this file.
+ data: The datasets used in paper.
	+ **Systems biology dataset**: Construct a heterogeneous network related to drug-target pairs
 		+ **Dataset_KB**：	all *.csv files in DrugKB folder	
   		+ **Dataset_Yamanishi**: 	drug_smiles.csv + protein_fasta.csv + drug_drug.csv + protein_protein.csv + drug_protein.csv
  	+ **Structural biology dataset**：They are used as input for training the model.
  		+ **DrugBank**:	DrugBank35022.txt
  	 	+ **Dataset_in_ne**t:DrugBank7710.txt
  	  	+ **dataset after remove related data**:new7710.txt
  	  	+ **Dataset_out_net**:out_net_data.txt
  	  	+ **Dataset_less**:DrugBank2570.txt

+ dataset.py: data process.
+ hyperparameter.py: set the hyperparameters of SSGraphDTI.
+ model.py: SSGraphDTI model architecture and Read systems biology data.
+ pytorchtools: early stopping.
+ SSGraphDTI.py: train and test the model.



# Run:

python SSGraphDTI.py
