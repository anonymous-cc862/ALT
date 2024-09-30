import torch
import numpy as np
import random
import pandas as pd
import scipy.io
from scipy.io import arff
import os
import sys
import zipfile
from sklearn.preprocessing import StandardScaler, RobustScaler
import mat4py


class Data_Loader():

    def __init__(self, n_trains=None):
        self.n_train = n_trains
        #self.name=name

    def get_dataset(self, dataset_name):

        rel_path = os.path.join("Data/",dataset_name)
        mat_files=['annthyroid.mat','arrhythmia.mat','breastw.mat','cardio.mat','forest_cover.mat','glass.mat','ionosphere.mat','letter.mat','lympho.mat','mammography.mat','mnist.mat','musk.mat',
                   'optdigits.mat','pendigits.mat','pima.mat','satellite.mat','satimage.mat','shuttle.mat','speech.mat','thyroid.mat','vertebral.mat','vowels.mat','wbc.mat','wine.mat']

        if dataset_name in mat_files :
            return self.build_train_test_generic_matfile(rel_path)

  

        if dataset_name == 'seismic.arff':
            #print('seismic')
            return self.build_train_test_seismic(rel_path)

        if dataset_name == 'mulcross.arff':
            #print('mullcross')
            return self.build_train_test_mulcross(rel_path)

        if dataset_name == 'abalone.data':
            #print('abalone')
            return self.build_train_test_abalone(rel_path)

        if dataset_name == 'ecoli.data':
            #print('ecoli')
            return self.build_train_test_ecoli(rel_path)

        if dataset_name == 'kddcup.data_10_percent_corrected.zip':
            #print ('kdd')
            return self.build_train_test_kdd('Data/kddcup.data_10_percent_corrected.zip') 

        if dataset_name == 'kddrev':
            #print ('kddrev')
            return self.build_train_test_kdd_rev('/Data/kddcup.data_10_percent_corrected.zip')

        if dataset_name == 'credit':
            return self.build_train_test_credit('/home/rliuaj/creditcard 2.csv')

        if dataset_name == 'adult':
            return self.build_train_test_adult('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/Data/adult')
        
        if dataset_name == 'mnist28':
            return self.build_train_test_mnist28('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/Data/mnist28.txt')


        sys.exit ('No such dataset!')

    def build_train_test_generic_matfile(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        da = mat4py.loadmat(name_of_file)
        col=['X'+str(i) for i in range(len(da['X'][0]))]
        df = pd.DataFrame(da.get('X'), columns=col)
        df['y']=pd.DataFrame(da.get('y'),columns=['y'])

        return df


    def build_train_test_credit(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        df = pd.read_csv('/home/rliuaj/creditcard 2.csv')
        df = df.loc[:, 'V1':] 


        return df
    
    def build_train_test_mnist28(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        df = pd.read_csv('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/Data/mnist28.txt') 


        return df
    
    def build_train_test_adult(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        df = pd.read_csv('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/Data/adult')


        return df

    def build_train_test_seismic(self,name_of_file): 
        dataset, _ = arff.loadarff(name_of_file)   
        dataset = pd.DataFrame(dataset)
        classes = dataset.iloc[:, -1]
        dataset = pd.get_dummies(dataset.iloc[:, :-1])
        c=[0 if x==b'0' else 1 for x in classes ] #0:2414, 1:170
        dataset = pd.concat((dataset, pd.DataFrame(c)), axis=1)

        return dataset


    def build_train_test_mulcross(self,name_of_file): 
        dataset, _ = arff.loadarff(name_of_file)   
        dataset = pd.DataFrame(dataset)
        classes = dataset.iloc[:, -1]
        dataset = pd.get_dummies(dataset.iloc[:, :-1])
        c=[0 if x==b'Normal' else 1 for x in classes ]
        dataset = pd.concat((dataset, pd.DataFrame(c)), axis=1)

        return dataset

    def build_train_test_ecoli(self,name_of_file):  
        dataset = dataset.iloc[:, 1:]
        anomalies = np.array(
            dataset[(dataset.iloc[:, 7] == 'omL') | (dataset.iloc[:, 7] == 'imL') | (dataset.iloc[:, 7] == 'imS')])[:,
                    :-1]
        normals = np.array(dataset[(dataset.iloc[:, 7] == 'cp') | (dataset.iloc[:, 7] == 'im') | (
                    dataset.iloc[:, 7] == 'pp') | (dataset.iloc[:, 7] == 'imU') | (dataset.iloc[:, 7] == 'om')])[:, :-1]
        normals = torch.tensor(normals.astype('double'))
        anomalies = torch.tensor(anomalies.astype('double'))
        normals = torch.cat((normals, torch.zeros(normals.shape[0], 1,dtype=torch.double)), dim=1)
        anomalies = torch.cat((anomalies, torch.ones(anomalies.shape[0], 1,dtype=torch.double)), dim=1)
        normals = normals[torch.randperm(normals.shape[0])]
        anomalies = anomalies[torch.randperm(anomalies.shape[0])]
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, :-1]
        test = test[:, :-1]
        return (train, test, test_classes)

    def build_train_test_abalone(self,path):  
        df = pd.read_csv('/home/rliuaj/fraud_contrastive/TabMap_model/Data/abalone_new') 

        return df



    def build_train_test_kdd(self,name_of_file):  
        
        zf = zipfile.ZipFile(name_of_file)
        kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
        entire_set = np.array(kdd_loader)
        revised_pd = pd.DataFrame(entire_set)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix='new1')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix='new2')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix='new3')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix='new6')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix='new11')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix='new21')), axis=1)
        revised_pd.drop(revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1)
        new_columns = [0, 'new1_icmp', 'new1_tcp', 'new1_udp', 'new2_IRC', 'new2_X11', 'new2_Z39_50', 'new2_auth',
                        'new2_bgp',
                        'new2_courier', 'new2_csnet_ns', 'new2_ctf', 'new2_daytime', 'new2_discard', 'new2_domain',
                        'new2_domain_u', 'new2_echo', 'new2_eco_i', 'new2_ecr_i', 'new2_efs', 'new2_exec', 'new2_finger',
                        'new2_ftp', 'new2_ftp_data', 'new2_gopher', 'new2_hostnames', 'new2_http', 'new2_http_443',
                        'new2_imap4',
                        'new2_iso_tsap', 'new2_klogin', 'new2_kshell', 'new2_ldap', 'new2_link', 'new2_login', 'new2_mtp',
                        'new2_name', 'new2_netbios_dgm', 'new2_netbios_ns', 'new2_netbios_ssn', 'new2_netstat', 'new2_nnsp',
                        'new2_nntp', 'new2_ntp_u', 'new2_other', 'new2_pm_dump', 'new2_pop_2', 'new2_pop_3', 'new2_printer',
                        'new2_private', 'new2_red_i', 'new2_remote_job', 'new2_rje', 'new2_shell', 'new2_smtp',
                        'new2_sql_net',
                        'new2_ssh', 'new2_sunrpc', 'new2_supdup', 'new2_systat', 'new2_telnet', 'new2_tftp_u', 'new2_tim_i',
                        'new2_time', 'new2_urh_i', 'new2_urp_i', 'new2_uucp', 'new2_uucp_path', 'new2_vmnet', 'new2_whois',
                        'new3_OTH', 'new3_REJ', 'new3_RSTO', 'new3_RSTOS0', 'new3_RSTR', 'new3_S0', 'new3_S1', 'new3_S2',
                        'new3_S3', 'new3_SF', 'new3_SH', 4, 5, 'new6_0', 'new6_1', 7, 8, 9, 10, 'new11_0', 'new11_1', 12, 13,
                        14,
                        15, 16, 17, 18, 19, 'new21_0', 'new21_1', 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                        35, 36, 37, 38, 39, 40, 41]
        revised_pd = revised_pd.reindex(columns=new_columns)
        revised_pd.loc[revised_pd[41] != 'normal.', 41] = 0
        revised_pd.loc[revised_pd[41] == 'normal.', 41] = 1

        df=revised_pd
 
        df.iloc[:,-1]=df.iloc[:,-1].astype(int)
        return df

