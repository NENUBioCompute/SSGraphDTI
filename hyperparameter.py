# -*- coding: utf-8 -*-

from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        self.Learning_rate = 5e-5
        self.Epoch = 50
        self.Batch_size = 128
        self.Patience = 5

        self.drug_kernel = [4, 6, 8]
        self.drug_MAX_LENGH = 100
        self.num_of_neighbourlayers_drug =2
        self.num_of_neighbours_drug = 7

        self.protein_kernel = [4, 8, 12]
        self.protein_MAX_LENGH=1000
        self.num_of_neighbourlayers_protein = 2
        self.num_of_neighbours_protein = 5

        self.gnn_layers = 1

        self.conv = 40
        self.char_dim = 64
        self.FC_Dropout = 0.1
