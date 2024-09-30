import numpy as np
import h5py

import datasets


class MYDATA:
    """
    A dataset of patches from mydata.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):

            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self):

        # load dataset /home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy4/train_fraud_emb1.txt
        f = np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/train_fraud_emb_new.txt',dtype=np.float32)#h5py.File('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy4/train_fraud_emb1.txt', 'r')
        new_num=np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/new_num.txt',dtype=np.float32)
        self.trn = self.Data(f[:int(f.shape[0]*0.85),:])#self.Data(f['train'])
        self.val = self.Data(f[int(f.shape[0]*0.85):,:])
        # self.tst = self.Data(f['test'])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2
        self.num=f.shape[0]
        self.new_num=int(new_num)
        #f.close()
