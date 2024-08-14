import pandas as pd

def creat_dict(dict, file):
    for item1, item2 in file.values:
        if item1 not in dict.keys():
            dict.update({item1: [item2]})
        else:
            dict[item1].append(item2)
    return

def readfile():
    drug = pd.read_csv('data/DrugKB/drug.csv', header=0)
    protein = pd.read_csv('data/DrugKB/protein.csv',header= 0)
    DPI = pd.read_csv('data/DrugKB/DPI.csv', header=0)
    PPI = pd.read_csv('data/DrugKB/PPI.csv', header=0)
    DDI_drugbank = pd.read_csv('data/DrugKB/DDI_drugbank.csv', header=0)
    DDI_drugs = pd.read_csv('data/DrugKB/DDI_drugs.csv', header=0)
    DDI_pharmgkb_pathways = pd.read_csv('data/DrugKB/DDI_pharmgkb_pathways.csv', header=0)
    DDI_pharmgkb_relationships = pd.read_csv('data/DrugKB/DDI_pharmgkb_relationships.csv', header=0)
    DDI_transformer_activation = pd.read_csv('data/DrugKB/DDI_transformer_activation.csv', header=0)

    drug_dict = {}
    for drug_name, smiles in drug.values:
        drug_dict.update({drug_name: smiles})
    print(f'the length of drug dict is {len(drug_dict)}')

    protein_dict = {}
    for protein_name, sequence in protein.values:
       protein_dict.update({protein_name: sequence})
    print(f'the length of protein dict is {len(protein_dict)}')

    DPI_dict = {}
    creat_dict(dict= DPI_dict , file= DPI)
    print(f'the length of DPI dict is {len(DPI_dict)}')

    PPI_dict = {}
    creat_dict(dict= PPI_dict, file= PPI)
    print(f'the length of PPI dict is {len(PPI_dict)}')

    DDI_dict = {}
    ddi_flag = "1+2+3+4+5"
    if ddi_flag == "1+2+3+4+5":
        creat_dict(dict=DDI_dict, file=DDI_drugbank)
        creat_dict(dict=DDI_dict, file=DDI_drugs)
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_pathways)
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_relationships)
        creat_dict(dict=DDI_dict, file=DDI_transformer_activation)
    if ddi_flag == "2+3+4+5":
        creat_dict(dict=DDI_dict, file=DDI_drugs)
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_pathways)
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_relationships)
        creat_dict(dict=DDI_dict, file=DDI_transformer_activation)
    if ddi_flag == "3+4+5":
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_pathways)
        creat_dict(dict=DDI_dict, file=DDI_pharmgkb_relationships)
        creat_dict(dict=DDI_dict, file=DDI_transformer_activation)
    print(f'the length of DDI dict is {len(DDI_dict)}')

    return drug_dict, protein_dict, DDI_dict, PPI_dict, DPI_dict