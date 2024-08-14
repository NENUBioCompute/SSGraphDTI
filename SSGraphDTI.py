# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import *
from dataset import CustomDataSet
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")


# 计算test集上的5个参数列表K折平均值，并生成./DATASET/results.txt文件，并打印结果
def show_result(DATASET, lable, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    F1_score_list = []
    for num in range(len(Accuracy_List)):
        F1_score_list.append(2.0 * Precision_List[num] * Recall_List[num]/(Precision_List[num] + Recall_List[num]))
    F1_score_mean,F1_score_var = np.mean(F1_score_list), np.var(F1_score_list)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
        f.write('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var) + '\n')
        f.write(f'the length of drug_dict :initial:{initial_length_of_drug_dict}, final:{len(drug_dict)}, add:{len(drug_dict) - initial_length_of_drug_dict}'+ '\n')
        f.write(f'the length of protein_dict :initial:{initial_length_of_protein_dict}, final:{len(protein_dict)}, add:{len(protein_dict) - initial_length_of_protein_dict}'+ '\n')


    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
    print('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var))
    print(f'the length of drug_dict :initial:{initial_length_of_drug_dict}, final:{len(drug_dict)}, add:{len(drug_dict) - initial_length_of_drug_dict}')
    print(f'the length of protein_dict :initial:{initial_length_of_protein_dict}, final:{len(protein_dict)}, add:{len(protein_dict) - initial_length_of_protein_dict}')

# 计算model在数据pbar下的指标数值
def test_precess(model, pbar, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            drugIDs, proIDs, numSMILESs, numFASTAs,labels = data
            numSMILESs = numSMILESs.cuda()
            numFASTAs = numFASTAs.cuda()
            labels = labels.cuda()
            predicted_scores = model(drugIDs, proIDs, numSMILESs, numFASTAs)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy() # 口口
            predicted_labels = np.argmax(predicted_scores, axis=1) # 口：0 or 1
            predicted_scores = predicted_scores[:, 1]
            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P , test_loss, Accuracy, Precision, Recall, AUC, PRC


# 调用了test_process函数，提供创建文件以及返回结果字符串
def test_model(dataset_load, save_path, DATASET, LOSS,dataset="Train", lable="best", save=True):
    test_pbar = tqdm(enumerate(BackgroundGenerator(dataset_load)),total=len(dataset_load))
    T, P , loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test= test_precess(model, test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET, dataset, lable), 'w') as f:
            for i in range(len(T)):
                #str(T[i])  + " " +
                f.write(str(T[i])  + " " +str(P[i]) + '\n')
    #result 是字符串
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.'.format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    #  testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test

# 划分K折数据集
def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset

# 打乱数据集
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    # DATASET = "KIBA"
    # DATASET = "Davis"
    # DATASET = "DrugBank2570" # 2570 seq data  out graph
    # DATASET = "out_net_data"# 7712 seq data  out graph
    DATASET = "new7710"
    # DATASET = "DrugBank3846" # 1923*2 data in graph
    # DATASET = "DrugBank7710" #  7710 seq data in graph
    # DATASET = "DrugBank9615" #  1923 in graph + 1923*4 out graph
    # DATASET = "DrugBank10282" # 7712+2570 3:1 in : out
    # DATASET = "DrugBank35022" # original sequence data
    # DATASET = "DrugBank38868" # 35022+3846
    # DATASET = "AsiaticAcid_metabolicm"
    # DATASET = "AsiaticAcid_transmem"
    # DATASET = "Asiaticside_metabolicm"
    # DATASET = "Asiaticside_transmem"

    print("Train in " + DATASET)
    weight_CE = None
    dir_input = ('./data/{}.txt'.format(DATASET))
    print("load data")
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("load finished")


    dataset = shuffle_dataset(train_data_list, SEED)
    print("Data shuffle finished !")
    K_Fold = 5

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset = CustomDataSet(test_dataset)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
        """ create model"""
        model = SSGraphDTI(hp).cuda()

        """load trained model"""
        optimizer = optim.AdamW(model.parameters(), lr=hp.Learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,cycle_momentum=False,step_size_up=train_size // hp.Batch_size)
        Loss = nn.CrossEntropyLoss(weight=weight_CE)

        save_path = "./" + DATASET + "/{}".format(i_fold)
        note = ''
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 0/1/2/3/4 The_results_of_whole_dataset.txt
        file_results = save_path + 'The_results_of_whole_dataset.txt'
        with open(file_results, 'w') as f:
            hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
            f.write(hp_attr + '\n')

        early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0)
        """Start training."""
        print('Training...')

        plt_x = []
        plt_trainloss = []
        plt_vaildloss =[]
        plt_Precision = []
        plt_Recall = []

        for epoch in range(1, hp.Epoch + 1):
            plt_x = list(range(0, hp.Epoch))
            trian_pbar = tqdm(enumerate(BackgroundGenerator(train_dataset_load)),total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for trian_i, train_data in trian_pbar:
                '''data preparation '''
                train_drugids, train_proids, train_numSMILES, train_numFASTAs, trian_labels = train_data
                train_numSMILES = train_numSMILES.cuda()
                train_numFASTAs = train_numFASTAs.cuda()
                trian_labels = trian_labels.cuda()

                optimizer.zero_grad()
                predicted_interaction = model(train_drugids, train_proids, train_numSMILES, train_numFASTAs)

                train_loss = Loss(predicted_interaction, trian_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)

            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
            plt_trainloss.append(train_loss_a_epoch)

            """valid"""
            valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_dataset_load)),total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_drugids, valid_proids, valid_numSMILES,  valid_numFASTAs,  valid_labels = valid_data
                    valid_numSMILES = valid_numSMILES.cuda()
                    valid_numFASTAs = valid_numFASTAs.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_scores = model(valid_drugids, valid_proids, valid_numSMILES, valid_numFASTAs)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]
                    valid_losses_in_epoch.append(valid_loss.item())
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
            Precision_dev = precision_score(Y, P)# 根据标签和输出计算精度
            Recall_dev = recall_score(Y, P)# 根据标签和输出计算recall
            Accuracy_dev = accuracy_score(Y, P)# 根据标签和输出计算Acc
            AUC_dev = roc_auc_score(Y, S)# 根据标签和预测值计算ROC曲线的AUC数值
            tpr, fpr, _ = precision_recall_curve(Y, S)# 计算precision recall，并不是TPR FPR!!!
            # 根据给定曲线上数值计算面积,fpr是recall, tpr是precision
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)
            plt_vaildloss.append(valid_loss_a_epoch)

            epoch_len = len(str(hp.Epoch))
            # f string > 对齐方式
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Recall: {Recall_dev:.5f} ')
            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid AUC', AUC_dev, epoch)
            writer.add_scalar('Valid AUPR', PRC_dev, epoch)
            writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
            writer.add_scalar('Valid Precision', Precision_dev, epoch)
            writer.add_scalar('Valid Recall', Recall_dev, epoch)
            writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)
            print(print_msg)
            early_stopping(valid_loss_a_epoch, model, epoch)
            #早停，修改x边界，break
            if early_stopping.early_stop:
                plt_x = list(range(0, epoch))
                break

        # 绘制train_loss and vaildloss曲线
        fig = plt.figure()
        ax = plt.axes()
        plt.plot(plt_x, plt_trainloss, linestyle='dashdot', label='train_loss', color="green")
        plt.plot(plt_x, plt_vaildloss, linestyle='dashed', label='vaild_loss', color="deeppink")
        plt.legend()
        plt.xlim(0, len(plt_trainloss))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig('{}/fold{}_{}epoch.jpg'.format(DATASET, i_fold,hp.Epoch))
        plt.show()

        trainset_test_stable_results, _, _, _, _, _ = test_model(train_dataset_load, save_path, DATASET, Loss,dataset="Train", lable="stable")
        validset_test_stable_results, _, _, _, _, _ = test_model(valid_dataset_load, save_path, DATASET, Loss,dataset="Valid", lable="stable")
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test =test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test", lable="stable")
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')


    # 展示最终结果
    show_result(DATASET, "stable",Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                AUC_List_stable, AUPR_List_stable)
#
    time_end = time.time()
    print(f'All accomplished !')
    print(f'The total time is {time_end - time_start:.4f}s!!!')