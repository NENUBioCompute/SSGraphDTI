# -*- coding: utf-8 -*-
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import  HeteroConv, SAGEConv
import warnings
import sys, os
from dataset import get_numSMILES, get_numFASTA
from readfromKB import readfile
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/mol2vec')
warnings.filterwarnings('ignore')

#读取数据，为创建subgraph做准备
time_start = time.time()

# 读取KB数据集
drug_dict ,protein_dict ,DDI_dict,PPI_dict,DPI_dict = readfile()
initial_length_of_drug_dict = len(drug_dict)
initial_length_of_protein_dict = len(protein_dict)

# 对于每一个batch的数据，整理数据为input:[DrugID,ProteinID,numSMILES,numFASTA,label]
def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [],[]
    MAX_len_SMILES = 100
    MAX_len_FASTA = 1000
    numSMILESs = torch.zeros((N, MAX_len_SMILES), dtype=torch.long)
    numFASTAs = torch.zeros((N, MAX_len_FASTA), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id,protein_id, smiles, fasta, label = pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]

        drug_ids.append(drug_id)
        if drug_id not in drug_dict.keys():
            drug_dict.update({drug_id: smiles})

        protein_ids.append(protein_id)
        if protein_id not in protein_dict.keys():
            protein_dict.update({protein_id: fasta})

        compoundint = torch.from_numpy(get_numSMILES(smiles, MAX_len_SMILES))
        numSMILESs[i] = compoundint

        proteinint = torch.from_numpy(get_numFASTA(fasta, MAX_len_FASTA))
        numFASTAs[i] = proteinint

        label = float(label)
        labels_new[i] = np.int(label)

    return drug_ids, protein_ids,  numSMILESs, numFASTAs, labels_new

class SSGraphDTI(nn.Module):
    def __init__(self, hp):
        super(SSGraphDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv

        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel

        self.drug_MAX_LENGH = hp.drug_MAX_LENGH
        self.protein_MAX_LENGH = hp.protein_MAX_LENGH

        self.num_of_neighbourlayers_drug = hp.num_of_neighbourlayers_drug
        self.num_of_neighbours_drug = hp.num_of_neighbours_drug
        self.num_of_neighbourlayers_protein = hp.num_of_neighbourlayers_protein
        self.num_of_neighbours_protein = hp.num_of_neighbours_protein
        self.gnn_layers = hp.gnn_layers


        self.drug_embed = nn.Embedding(71, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)

        #采用embedding进行编码
        self.creat_drug_node = nn.Embedding(num_embeddings=71,embedding_dim=self.drug_MAX_LENGH,padding_idx=0)
        self.creat_protein_node = nn.Embedding(num_embeddings=26, embedding_dim=self.protein_MAX_LENGH, padding_idx=0)

        self.graphConv = HeteroConv(
            {('drug', 'DDI', 'drug'):SAGEConv((-1, -1), self.drug_MAX_LENGH),
             ('protein', 'PPI', 'protein'):SAGEConv((-1, -1), self.protein_MAX_LENGH),
             ('drug', 'DPI', 'protein'):SAGEConv((-1, -1), self.protein_MAX_LENGH),
             ('protein', 'PDI', 'drug'):SAGEConv((-1, -1), self.drug_MAX_LENGH)}, aggr='mean')

        self.dropout1 = nn.Dropout(hp.FC_Dropout)
        self.dropout2 = nn.Dropout(hp.FC_Dropout)
        self.dropout3 = nn.Dropout(hp.FC_Dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(1420, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    # 创建与输入相关的子网络；输入：药物ID，药物邻居跳数，层药物邻居最大数，蛋白ID，蛋白邻居跳数，层蛋白邻居最大数
    # 返回：ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist
    def creat_subgraph(self, drugID, num_NeighbourLayers_of_drug, num_Neighbours_of_drug,
                       proID, num_NeighbourLayers_of_protein,num_Neighbours_of_protein):
        # find temp_druglist
        temp_druglist = []
        temp_druglist.append(drugID)
        if drugID in DDI_dict.keys():
            for i in range(num_NeighbourLayers_of_drug):
                addlist_drug = []
                for item in temp_druglist:
                    if item in DDI_dict.keys():
                        if len(DDI_dict[item])>= num_Neighbours_of_drug:
                            add_drugs_indexs = random.sample(range(0,len(DDI_dict[item])),num_Neighbours_of_drug)
                            for index_drug in add_drugs_indexs:
                                addlist_drug.append(DDI_dict[item][index_drug])
                        else:
                            addlist_drug.extend(DDI_dict[item])
                addlist_drug = set(addlist_drug).difference(temp_druglist)
                temp_druglist.extend(addlist_drug)

        # find temp_prolist
        temp_prolist = []
        temp_prolist.append(proID)
        if proID in PPI_dict.keys():
            for i in range(num_NeighbourLayers_of_protein):
                addlist_pro = []
                for item in temp_prolist:
                    if item in PPI_dict.keys():
                        if len(PPI_dict[item])>=num_Neighbours_of_protein:
                            add_proteins_indexs = random.sample(range(0, len(PPI_dict[item])), num_Neighbours_of_protein)
                            for index_protein in add_proteins_indexs:
                                addlist_pro.append(PPI_dict[item][index_protein])
                        else:
                            addlist_pro.extend(PPI_dict[item])
                addlist_pro = set(addlist_pro).difference(temp_prolist)
                temp_prolist.extend(addlist_pro)

        # create DDI index
        ddi_edgeindex = None
        if len(temp_druglist) > 1:
            DDI_coo = []
            for drug1 in temp_druglist:
                if drug1 in DDI_dict.keys():
                    drug2list = DDI_dict[drug1]
                    for item in drug2list:
                        if item in temp_druglist:
                            newpoint = [[temp_druglist.index(drug1), temp_druglist.index(item)]]
                            DDI_coo.extend(newpoint)
            ddi_edgeindex = torch.tensor(DDI_coo).t()

        # create PPI index
        ppi_edgeindex = None
        if len(temp_prolist) > 1:
            PPI_coo = []
            for pro1 in temp_prolist:
                if pro1 in PPI_dict.keys():
                    pro2list = PPI_dict[pro1]
                    for item in pro2list:
                        if item in temp_prolist:
                            newpoint = [[temp_prolist.index(pro1), temp_prolist.index(item)]]
                            PPI_coo.extend(newpoint)
            ppi_edgeindex = torch.tensor(PPI_coo).t()

        # create DPI index
        DPI_coo = []
        for drug1 in temp_druglist:
            if drug1 in DPI_dict.keys():
                pro2list = DPI_dict[drug1]
                for item in pro2list:
                    if item in temp_prolist:
                        newpoint = [[temp_druglist.index(drug1), temp_prolist.index(item)]]
                        DPI_coo.extend(newpoint)
        if len(DPI_coo) == 0:
            dpi_edgeindex = None
        else:
            dpi_edgeindex = torch.tensor(DPI_coo).t()
        # create PDI index
        if dpi_edgeindex == None:
            pdi_edgeindex = None
        else:
            pdi_edgeindex = dpi_edgeindex[[1, 0]]

        return ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist

    #
    def DP_Pair_graphConv(self, drugIDs, proteinIDs, graph_model_layers_num):
        result = torch.zeros((len(drugIDs), self.protein_MAX_LENGH+self.drug_MAX_LENGH))
        for i in range(len(drugIDs)):
            ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist = \
                self.creat_subgraph(drugIDs[i],self.num_of_neighbourlayers_drug,self.num_of_neighbours_drug,
            proteinIDs[i],self.num_of_neighbourlayers_protein,self.num_of_neighbours_protein)
            data = HeteroData()
            data['drug', 'DDI', 'drug'].edge_index = ddi_edgeindex
            data['protein', 'PPI', 'protein'].edge_index = ppi_edgeindex
            data['drug', 'DPI', 'protein'].edge_index = dpi_edgeindex
            data['protein', 'PDI', 'drug'].edge_index = pdi_edgeindex
            data['drug'].x = torch.zeros((len(temp_druglist),self.drug_MAX_LENGH ))
            data['protein'].x = torch.zeros((len(temp_prolist),self.protein_MAX_LENGH))

            for j, drugname in enumerate(temp_druglist):
                data['drug'].x[j] = torch.from_numpy((get_numSMILES(drug_dict[drugname], self.drug_MAX_LENGH)))
            data['drug'].x = data['drug'].x.int().cuda()
            data['drug'].x = self.creat_drug_node(data['drug'].x)
            data['drug'].x = torch.mean(data['drug'].x, dim=1)

            for k, proname in enumerate(temp_prolist):
                data['protein'].x[k] =  torch.from_numpy((get_numFASTA(protein_dict[proname],self.protein_MAX_LENGH)))
            data['protein'].x = data['protein'].x.int().cuda()
            data['protein'].x = self.creat_protein_node( data['protein'].x)
            data['protein'].x = torch.mean( data['protein'].x,dim=1)

            data = data.cuda()
            graphconv_out = self.graphConv(data.collect('x'),data.collect("edge_index"))

            if graph_model_layers_num > 1:
                for conv_num in range(graph_model_layers_num - 1):
                    if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) == 0:# drug、protein均不更新
                        result[i] = torch.cat(( data['drug'].x[0],  data['protein'].x[0]), dim=0)

                    if len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) == 0:# drug 更新
                        data['drug'].x = graphconv_out['drug']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((graphconv_out['drug'][0], data['protein'].x[0]), dim=0)

                    if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) > 0:# protein 更新
                        data['protein'].x = graphconv_out['protein']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((data['drug'].x[0], graphconv_out['protein'][0]), dim=0)

                    if  len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) > 0:# drug、protein均更新
                        data['drug'].x = graphconv_out['drug']
                        data['protein'].x = graphconv_out['protein']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((graphconv_out['drug'][0], graphconv_out['protein'][0]), dim=0)

            if graph_model_layers_num ==1 :
                if len(graphconv_out['drug']) == 0  and len(graphconv_out['protein']) == 0:
                    result[i] = torch.cat(( data['drug'].x[0],  data['protein'].x[0]), dim=0)
                if len(graphconv_out['drug']) > 0  and len(graphconv_out['protein']) == 0:
                    result[i] = torch.cat((graphconv_out['drug'][0],  data['protein'].x[0]), dim=0)
                if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) > 0:
                    result[i] = torch.cat((data['drug'].x[0], graphconv_out['protein'][0]), dim=0)
                if  len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) > 0:
                    result[i] = torch.cat((graphconv_out['drug'][0], graphconv_out['protein'][0]), dim=0)
        return result

    def forward(self, drugIDs, proIDs, numSMILES,numFASTA):
        drugembed = self.drug_embed(numSMILES)
        proteinembed = self.protein_embed(numFASTA)

        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        graphConv = self.DP_Pair_graphConv(drugIDs, proIDs,self.gnn_layers)
        graphConv = graphConv.cuda().float()

        pair = torch.cat([drugConv, proteinConv, graphConv], dim=1)

        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))

        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))

        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))

        predict = self.out(fully3)
        return predict
