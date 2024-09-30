

import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD, RMSprop,AdamW,Adadelta,Adagrad,SparseAdam,Adamax,ASGD,LBFGS,Rprop
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import roc_curve
from sklearn.preprocessing import Normalizer

sys.path.append("../")

from TabMap.loss import NTXent, NTXent1, cross_entropy
from TabMap.model import TabMap, fine_tune, baseline, Encoder, c_head

from example.dataset import ExampleDataset
#from example.utils import dataset_embeddings,  train_epoch #,fix_seed
from TabMap.early_stop import EarlyStopping

from sklearn.metrics import roc_auc_score
#seed = 1234
#fix_seed(seed)

from sklearn.preprocessing import StandardScaler, RobustScaler
from numpy.random import seed
import mat4py
import os
from TabMap.data_loader import Data_Loader
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler #




device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

def f1_calculator(classes, losses):
    classes=classes#.numpy()
    losses=losses#.numpy()
    df_version_classes = pd.DataFrame(data=classes)
    df_version_losses = pd.DataFrame(losses).astype(np.float64)
    Na = df_version_classes[df_version_classes.iloc[:, 0] == 1].shape[0]
    anomaly_indices = df_version_losses.nlargest(Na, 0).index.values
    picked_anomalies = df_version_classes.iloc[anomaly_indices]
    true_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 1].shape[0]
    false_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 0].shape[0]
    f1 = true_pos / (true_pos + false_pos)
    return (f1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def f_embeddings(new_model, loader, device): 
    setup_seed(random_state)
    new_model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for anchor, _, target in tqdm(loader):
            anchor = anchor.to(device)
            embeddings.append(new_model.get_embeddings(anchor)) 
            targets.append(target)

    embeddings = torch.cat(embeddings).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    return embeddings, targets

seed=[]
auc_allseed=[]
f1_allseed=[]
name='arrhythmia.mat'

from imblearn.datasets import fetch_datasets

if name in ['yeast_me2','isolet','yeast_ml8','us_crime','ozone_level','oil','scene','libras_move','webpage','mammography']:
    ddd=fetch_datasets()[name]
    df=pd.concat([pd.DataFrame(ddd.data),pd.DataFrame(ddd.target)],axis=1)
    df.iloc[:,-1]=[0 if df.iloc[i,-1]==-1 else 1 for i in range(len(df))]
    df.columns=['col_'+str(i) for i in range(df.shape[1])]
else:
    df=Data_Loader(name).get_dataset(name)

print(df.shape)

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

for i in df.columns[:-1]:
    df[i] = rob_scaler.fit_transform(df[i].values.reshape(-1,1))


lst1np.random.randint(1000, 10000, 30)
for random_state in lst1:
    auc=[]
    f1=[]
    seed.append(random_state)
    setup_seed(random_state)
    def train_test_val_split(df,train_ratio = 0.7,validation_ratio = 0.1,test_ratio = 0.2,random_state=random_state):
        # random_state for reproduction
        # shuffle must be 'True'
        [x_train, x_test, y_train, y_test] = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=validation_ratio+test_ratio, random_state=random_state, shuffle=True)
        [x_val, x_test, y_val, y_test] = train_test_split(
        x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state)
        return x_train,y_train, x_test, y_test, x_val, y_val

    x_train,y_train, x_test, y_test, x_val, y_val=train_test_val_split(df,train_ratio = 0.7,validation_ratio = 0.1,test_ratio = 0.2,random_state=random_state)
    
    setup_seed(random_state)
    # to torch dataset
    train_ds = ExampleDataset(
        x_train,#.to_numpy(), 
        y_train,#.to_numpy(), 
        columns=df.columns[:-1]
    )
    test_ds = ExampleDataset(
        x_test,#.to_numpy(), 
        y_test,#.to_numpy(), 
        columns=df.columns[:-1]
    )
    val_ds = ExampleDataset(
        x_val,#.to_numpy(), 
        y_val,#.to_numpy(), 
        columns=df.columns[:-1]
    )

    setup_seed(random_state)

    batch_size = 128
    epochs = 20
    early_stopping = EarlyStopping(patience=4, verbose=True) 

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = TabMap(
        input_dim=train_ds.shape[1], 
        anchor_rate=0.6
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_history = []
    valid_losses=[]
    ep=[]
    for epoch in range(1, epochs + 1):
        ep.append(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_loss_val=0.0
        train_batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) 
        val_batch = tqdm(val_loader, desc=f"Epoch {epoch}", leave=False) 

        for anchor, positive, _ in train_batch: 
            anchor = anchor.to(device)

            optimizer.zero_grad()

            # get embeddings
            emb_anchor, emb_positive = model(anchor) 

            # compute loss
            loss = criterion(emb_anchor, emb_positive, device)  
            loss.backward()

            # update model weights
            optimizer.step()

            # log progress
            epoch_loss += anchor.size(0) * loss.item() 
            train_batch.set_postfix({"loss": loss.item()})
        epoch_loss=epoch_loss / len(train_loader.dataset) 
        loss_history.append(epoch_loss)


        model.eval()
        for anchor_val, positive_val,_ in val_batch:
            anchor_val, positive_val=anchor_val.to(device), positive_val.to(device)
            emb_anchor_val, emb_positive_val = model(anchor_val)
            loss_val = criterion(emb_anchor_val, emb_positive_val, device)
            epoch_loss_val += anchor_val.size(0) * loss_val.item()
        epoch_loss_val=epoch_loss_val / len(val_loader.dataset)
        valid_losses.append(epoch_loss_val)

        early_stopping(epoch_loss_val, model)
        if early_stopping.early_stop: 
            model.save_pre_encoder_model()
            print("Early stopping")
            print(epoch)
            break
    

    setup_seed(random_state)

    train_embeddings, train_targets =f_embeddings(model, train_loader, device)#.cpu() #训练好的model，通过encoder f得到emb，g丢了
    val_embeddings, val_targets =f_embeddings(model, val_loader, device)#.cpu()
    test_embeddings, test_targets =f_embeddings(model, test_loader, device)#.cpu()
 
    #mlp
    setup_seed(random_state)
    train_embeddings_ed=ExampleDataset(
        train_embeddings,#.to_numpy(), 
        train_targets#,#.to_numpy(), 
    )

    val_embeddings_ed=ExampleDataset(
        val_embeddings,#.to_numpy(), 
        val_targets#,#.to_numpy(), 
    )
    test_embeddings_ed=ExampleDataset(
        test_embeddings,#.to_numpy(), 
        test_targets#,#.to_numpy(), 
    )
    setup_seed(random_state)
    train_emb_loader = DataLoader(train_embeddings_ed, batch_size=batch_size, shuffle=True) #128
    val_emb_loader = DataLoader(val_embeddings_ed, batch_size=batch_size, shuffle=True)
    test_emb_loader = DataLoader(test_embeddings_ed, batch_size=batch_size, shuffle=True)

    model_baseline = baseline(
        input_dim=train_embeddings.shape[1]#train_ds

    ).to(device)
    optimizer_baseline = Adam(model_baseline.parameters(), lr=0.001) 
    criterion_baseline=nn.BCEWithLogitsLoss()
    early_stopping_baseline= EarlyStopping(patience=3, verbose=True)

    setup_seed(random_state)
    loss_history_baseline = []
    valid_losses_baseline=[]
    epochs_baseline=100

    for epoch in range(1, epochs_baseline + 1):
        model_baseline.train()
        epoch_loss_baseline = 0.0
        epoch_loss_val_baseline=0.0
        train_emb_batch = tqdm(train_emb_loader, desc=f"Epoch {epoch}", leave=False) 
        val_emb_batch = tqdm(val_emb_loader, desc=f"Epoch {epoch}", leave=False) 

        for anchor, _, target in train_emb_batch: 
            anchor=anchor.to(device)

            optimizer_baseline.zero_grad()

            # get embeddings
            output = model_baseline(anchor) 

            # compute loss
            loss_baseline = criterion_baseline(output, target.unsqueeze(1).to(device))  
            loss_baseline.backward()

            # update model weights
            optimizer_baseline.step()

            # log progress
            epoch_loss_baseline += anchor.size(0) * loss_baseline.item() #每个batch的
        epoch_loss_baseline = epoch_loss_baseline / len(train_loader.dataset) 
        loss_history_baseline.append(epoch_loss_baseline)

        model_baseline.eval()
        for anchor_val, _, target_val in val_emb_batch:
            anchor_val, target_val = anchor_val.to(device), target_val.to(device)
            output_val = model_baseline(anchor_val)
            loss_val_baseline = criterion_baseline(output_val, target_val.unsqueeze(1).to(device))
            epoch_loss_val_baseline += anchor_val.size(0) * loss_val_baseline.item()
        epoch_loss_val_baseline=epoch_loss_val_baseline / len(val_loader.dataset)

        early_stopping_baseline(epoch_loss_val_baseline, model_baseline)
        if early_stopping_baseline.early_stop:
            print("Early stopping")
            print(epoch)
            break

    #test
    setup_seed(random_state)
    model_baseline.eval()
    test_emb_batch = tqdm(test_emb_loader, desc=f"Epoch {epoch}", leave=False) 
    test_output_list=torch.FloatTensor()#.to(device)
    test_target_list=torch.FloatTensor()#.to(device)
    for anchor_test, _, target_test in test_emb_batch: 
        anchor_test= anchor_test.to(device)#, target_test.to(device)
        output_test = model_baseline(anchor_test)
        test_output_list=torch.cat((test_output_list,output_test.detach().to('cpu')),0)
        test_target_list=torch.cat((test_target_list,target_test.detach().to('cpu')),0)

    auc.append(roc_auc_score(test_target_list.numpy(), test_output_list.numpy()))
    f1.append(f1_calculator(test_target_list.numpy(), test_output_list.numpy()))


    train_emb_d=pd.DataFrame(train_embeddings)#.numpy())
    test_emb_d=pd.DataFrame(test_embeddings)#.numpy())
    #lr
    setup_seed(random_state)
    lr = LogisticRegression()

    lr.fit(train_emb_d, train_targets)#.numpy())
    auc.append(roc_auc_score(test_targets, lr.predict_proba(test_emb_d)[:,1]))
    f1.append(f1_calculator(test_targets, lr.predict_proba(test_emb_d)[:,1]))


    #dt
    setup_seed(random_state)
    dtree = DecisionTreeClassifier(
        class_weight="balanced",
        min_weight_fraction_leaf=0.01,
        random_state=random_state)
    dtree = dtree.fit(train_emb_d, train_targets)#.numpy())
    auc.append(roc_auc_score(test_targets, dtree.predict_proba(test_emb_d)[:,1]))#dtree3.predict(test_embeddings)) rf_emb_add.predict_proba(test_embeddings)[:,1]
    f1.append(f1_calculator(test_targets, dtree.predict_proba(test_emb_d)[:,1]))

    #rf
 
    rf_5 = RandomForestClassifier(
        n_estimators=100, 
        criterion= 'entropy',
        max_depth=10 # 

        )
    rf_5.fit(train_emb_d, train_targets)

    auc.append(roc_auc_score(test_targets, rf_5.predict_proba(test_emb_d)[:,1]))
    f1.append(f1_calculator(test_targets, rf_5.predict_proba(test_emb_d)[:,1]))
    
#invertable net augmentation 
    def get_index(lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]
    

    train_fraud_emb=train_embeddings[get_index(train_targets, 1)] 
    train_nonfraud_emb=train_embeddings[get_index(train_targets, 0)] 
    # sampling_strategy = {0: train_fraud_emb.shape[0]+int((train_nonfraud_emb.shape[0]-train_fraud_emb.shape[0])/1.5), 1: train_fraud_emb.shape[0]}
    # under = RandomUnderSampler(random_state=random_state,sampling_strategy=sampling_strategy) 
    # train_embeddings, train_targets= under.fit_resample(train_embeddings,train_targets)
    

    # train_fraud_emb=train_embeddings[get_index(train_targets, 1)] 
    # train_nonfraud_emb=train_embeddings[get_index(train_targets, 0)] 
    num=train_embeddings[get_index(train_targets, 0)].shape[0]-train_fraud_emb.shape[0] 
    np.savetxt('new_num.txt',[num])
    ma=train_fraud_emb.max()
    mi=train_fraud_emb.min()
    train_fraud_emb_scale=(2*(train_fraud_emb-mi)/(ma-mi))-1 #0~1

    ma_all=train_embeddings.max()
    mi_all=train_embeddings.min()
    train_all_emb_scale=(2*(train_embeddings-mi_all)/(ma_all-mi_all))-1 

    np.savetxt('train_fraud_emb_new.txt',train_fraud_emb_scale) 
    np.savetxt('train_all_emb_new.txt',train_all_emb_scale) 
    np.savetxt('train_all_targets.txt',train_targets)
    #normal emb scale

    os.system('python ffjord/train_tabular1.py') #在z上面tomek
    new_new=np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/ffjord/generated_samples_new1.txt',dtype=np.float32)
    new=(new_new+1)*(ma-mi)/2+mi# 


#tomek
    ind=np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/ffjord/ind.txt',dtype=np.float32)
    print(len(ind))
    print(new.shape)

    train_embeddings_new=np.append(train_embeddings,new,axis=0)
    train_labels_new=np.append(train_targets,[1]*new.shape[0],axis=0) 

    train_embeddings_new=train_embeddings_new[list(map(int, ind.tolist())),]
    train_labels_new=train_labels_new[list(map(int, ind.tolist())),]



    setup_seed(random_state)
    train_embeddings_new_set=ExampleDataset(
        train_embeddings_new,#.to_numpy(), 
        train_labels_new#,#.to_numpy(), 
    )

    val_embeddings_set=ExampleDataset(
        val_embeddings,#.to_numpy(), 
        val_targets#,#.to_numpy(), 
    )
    test_embeddings_set=ExampleDataset(
        test_embeddings,#.to_numpy(), 
        test_targets#,#.to_numpy(), 
    )

    setup_seed(random_state)
    train_emb_new_loader = DataLoader(train_embeddings_new_set, batch_size=batch_size, shuffle=True) #128
    val_emb_loader = DataLoader(val_embeddings_set, batch_size=batch_size, shuffle=True)
    test_emb_loader = DataLoader(test_embeddings_set, batch_size=batch_size, shuffle=True)

    setup_seed(random_state)
    model_baseline_aug = baseline(
        input_dim=train_embeddings_new.shape[1]#train_ds#, 

    ).to(device)
    optimizer_baseline_aug = Adam(model_baseline_aug.parameters(), lr=0.001) 
    criterion_baseline_aug=nn.BCEWithLogitsLoss()
    early_stopping_baseline= EarlyStopping(patience=3, verbose=True)

    setup_seed(random_state)
    loss_history_baseline = []
    valid_losses_baseline=[]
    epochs_baseline=100

    for epoch in range(1, epochs + 1):

        model_baseline_aug.train()
        epoch_loss_baseline = 0.0
        epoch_loss_val_baseline=0.0
        train_batch = tqdm(train_emb_new_loader, desc=f"Epoch {epoch}", leave=False) 
        val_batch = tqdm(val_emb_loader, desc=f"Epoch {epoch}", leave=False) 

        for anchor, _, target in train_batch: 
            anchor=anchor.to(device)

            optimizer_baseline_aug.zero_grad()

            # get embeddings
            output = model_baseline_aug(anchor) 

            # compute loss
            loss_baseline = criterion_baseline_aug(output, target.unsqueeze(1).to(device))  
            loss_baseline.backward()

            # update model weights
            optimizer_baseline_aug.step()

            # log progress
            epoch_loss_baseline += anchor.size(0) * loss_baseline.item() 
        epoch_loss_baseline = epoch_loss_baseline / len(train_loader.dataset) 
        loss_history_baseline.append(epoch_loss_baseline)

        model_baseline_aug.eval()
        for anchor_val, _, target_val in val_batch:
            anchor_val, target_val = anchor_val.to(device), target_val.to(device)
            output_val = model_baseline_aug(anchor_val)
            loss_val_baseline = criterion_baseline_aug(output_val, target_val.unsqueeze(1).to(device))
            epoch_loss_val_baseline += anchor_val.size(0) * loss_val_baseline.item()
        epoch_loss_val_baseline=epoch_loss_val_baseline / len(val_loader.dataset)

        early_stopping_baseline(epoch_loss_val_baseline, model_baseline_aug)
        if early_stopping_baseline.early_stop:
            print("Early stopping")
            print(epoch)
            break

    #test
    setup_seed(random_state)
    model_baseline_aug.eval()
    test_batch = tqdm(test_emb_loader, desc=f"Epoch {epoch}", leave=False) 
    test_output_list=torch.FloatTensor()#.to(device)
    test_target_list=torch.FloatTensor()#.to(device)
    for anchor_test, _, target_test in test_batch: 
        anchor_test= anchor_test.to(device)
        output_test = model_baseline_aug(anchor_test)
        test_output_list=torch.cat((test_output_list,output_test.detach().to('cpu')),0)
        test_target_list=torch.cat((test_target_list,target_test.detach().to('cpu')),0)

    auc.append(roc_auc_score(test_target_list.numpy(), test_output_list.numpy()))
    f1.append(f1_calculator(test_target_list.numpy(), test_output_list.numpy()))

    train_emb_d=pd.DataFrame(train_embeddings_new)#.numpy())
    test_emb_d=pd.DataFrame(test_embeddings)#.numpy())

    #lr
    setup_seed(random_state)
    lr = LogisticRegression()
    lr.fit(train_emb_d, train_labels_new)#.numpy())
    auc.append(roc_auc_score(test_targets, lr.predict_proba(test_emb_d)[:,1]))
    f1.append(f1_calculator(test_targets, lr.predict_proba(test_emb_d)[:,1]))


    #dt
    setup_seed(random_state)
    dtree = DecisionTreeClassifier(
        class_weight="balanced",
        min_weight_fraction_leaf=0.01,
        random_state=random_state)
    dtree = dtree.fit(train_emb_d, train_labels_new)#.numpy())
    auc.append(roc_auc_score(test_targets, dtree.predict_proba(test_emb_d)[:,1]))
    f1.append(f1_calculator(test_targets, dtree.predict_proba(test_emb_d)[:,1]))

    #rf

    rf_5 = RandomForestClassifier(
        n_estimators=100, 
        criterion= 'entropy',
        max_depth=10 # 

        )
    rf_5.fit(train_emb_d, train_labels_new)

    auc.append(roc_auc_score(test_targets, rf_5.predict_proba(test_emb_d)[:,1]))
    f1.append(f1_calculator(test_targets, rf_5.predict_proba(test_emb_d)[:,1]))

#result

    auc=pd.DataFrame(auc).transpose()
    auc_allseed=pd.concat([pd.DataFrame(auc_allseed),auc],axis=0)
    save_path='/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/result/auc_arrhythmia'+'.csv' 
    auc_allseed.to_csv(save_path,index=True)  #arr 16

    f1=pd.DataFrame(f1).transpose()
    f1_allseed=pd.concat([pd.DataFrame(f1_allseed),f1],axis=0)
    f1_allseed.to_csv('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/result/f1_arrhythmia'+'.csv',index=True) 

    print(df.shape)

auc_allseed.index=seed
auc_allseed.columns=['crop_mlp','crop_lr','crop_dt','crop_rf','crop_aug_mlp','crop_aug_lr','crop_aug_dt','crop_aug_rf']
save_path='/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/result/auc_arrhythmia'+'.csv'
auc_allseed.to_csv(save_path,index=True)   

f1_allseed.index=seed
f1_allseed.columns=['crop_mlp','crop_lr','crop_dt','crop_rf','crop_aug_mlp','crop_aug_lr','crop_aug_dt','crop_aug_rf']#['crop_mlp','crop_lr','crop_dt','crop_aug_mlp','crop_aug_lr','crop_aug_dt']
f1_allseed.to_csv('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/result/f1_arrhythmia'+'.csv',index=True)  