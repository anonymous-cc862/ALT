import numpy as np
import h5py

import datasets


class BSDS300:
    """
    A dataset of patches from BSDS300.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):

            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self):

        # load dataset
        f = h5py.File(datasets.root + 'BSDS300/BSDS300.hdf5', 'r')

        self.trn = self.Data(f['train'])
        self.val = self.Data(f['validation'])
        self.tst = self.Data(f['test'])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2

        f.close()

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

        # load dataset
        f = np.loadtxt('train_fraud_emb.txt1',dtype=np.float32)#h5py.File('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy4/train_fraud_emb1.txt', 'r')

        self.trn = self.Data(f[:int(f.shape[0]*0.9),:])#self.Data(f['train'])
        self.val = self.Data(f[int(f.shape[0]*0.9):,:])
        # self.tst = self.Data(f['test'])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2
        self.num=f.shape[0]

        f.close()
