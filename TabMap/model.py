import torch
import torch.nn as nn
import random
import numpy as np
from random import sample



class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim)) 
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers) 

class Encoder(torch.nn.Sequential): #anchor full data & positive random crop
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim=[256,256,256,256], n_layers=4, dropout=0.0): 
        layers = []
        in_dim = input_dim
        out_dim=hidden_dim[0]
        for i in range(n_layers - 1): 
            layers.append(torch.nn.Linear(in_dim, out_dim))  
            layers.append(nn.LeakyReLU(0.2))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim[i] 
            out_dim = hidden_dim[i+1] 

        layers.append(torch.nn.Linear(in_dim, out_dim)) 

        super().__init__(*layers) 


class g_head(torch.nn.Sequential): 
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim=256, hidden_dim=[256,256], n_layers=2, dropout=0.0):
        layers = []
        in_dim = input_dim #128
        out_dim=hidden_dim[0] #50
        for i in range(n_layers - 1): 
            layers.append(torch.nn.Linear(in_dim, out_dim))  
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim[i] 
            out_dim=hidden_dim[i+1] 

        layers.append(torch.nn.Linear(in_dim, out_dim)) 

        super().__init__(*layers) 

class c_head(torch.nn.Sequential): #fine tune classification head
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim=256, hidden_dim=[256,256,1], n_layers=3, dropout=0.0):
        layers = []
        in_dim = input_dim 
        out_dim=hidden_dim[0] 
        for i in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, out_dim))  
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim[i] 
            out_dim=hidden_dim[i+1] 

        layers.append(torch.nn.Linear(in_dim, out_dim)) 

        super().__init__(*layers)

class classification(torch.nn.Sequential): #baseline head
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim=256, hidden_dim=[256,256,1], n_layers=3, dropout=0.0):
        layers = []
        in_dim = input_dim #256
        out_dim=hidden_dim[0] #256(0)
        for i in range(n_layers - 1): #0 
            layers.append(torch.nn.Linear(in_dim, out_dim))  
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim[i] 
            out_dim=hidden_dim[i+1] 

        layers.append(torch.nn.Linear(in_dim, out_dim)) 

        super().__init__(*layers) 


class TabMap(nn.Module): 
    def __init__(
        self,
        input_dim,

        anchor_rate=0.5,
        encoder=None,
        encoder2=None,
        pretraining_head=None#,

    ):

        super().__init__()

        # initialize weights

        self.input_dim=input_dim
        self.anchor_len = int(anchor_rate * input_dim) 

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(self.input_dim)
        if encoder2:
            self.encoder2 = encoder2
        else:
            self.encoder2 = Encoder(self.anchor_len)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
            self.pretraining_head = g_head()
        self.encoder.apply(self._init_weights)
        self.encoder2.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor):
        batch_size, m = anchor.size() 


        for i in range(batch_size): 
            l = range(m)
            pos_index=sample(l, self.anchor_len) 

            pos_index=torch.sort(torch.tensor(pos_index)).values 
            if i==0:
                positive=anchor[i,pos_index].unsqueeze(0)
            else:
                positive = torch.cat((positive,anchor[i,pos_index].unsqueeze(0)),0)


        emb_anchor = self.encoder(anchor) 
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder2(positive) 
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive

    def save_pre_encoder_model(self):
        torch.save(self.encoder.state_dict(),'TabMap/pre_encoder_checkpoint.dict')
        print('save pre_encoder model')
        return self.encoder.state_dict()

    def get_embeddings(self, input):
        #self.encoder.eval()
        return self.encoder(input)

class fine_tune(nn.Module): 
    def __init__(
        self,
        input_dim,

        encoder=None,
        classification_head=None,
    ):
       
        super().__init__()

        encoder=torch.load('TabMap/pre_encoder_checkpoint.dict')
        self.encoder=Encoder(input_dim)
        self.encoder.load_state_dict(encoder)

        if classification_head:
            self.classification_head = classification_head
        else:
            self.classification_head = c_head()

        # initialize weights
        self.classification_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, inputs):

        # compute embeddings
        emb = self.encoder(inputs) 
        out = self.classification_head(emb)
        return out 
    
    def save_finetune_model(self):
        #return self.encoder(input)
        torch.save(self.encoder.state_dict(),'TabMap/finetune_encoder_checkpoint.dict')
        torch.save(self.classification_head.state_dict(),'TabMap/finetune_classification_checkpoint.dict')
        print('save encoder&classification model')
        return self.encoder.state_dict(), self.classification_head.state_dict()

    def get_embeddings(self, input):
        #self.encoder.eval()
        return self.encoder(input)
    
    def get_prediction(self, emb):
        return self.classification_head(emb)

class baseline(nn.Module): 
    def __init__(
        self,
        input_dim,

        classification_head=None,
    ):
       
        super().__init__()

        if classification_head:
            self.classification_head = classification_head
        else:
            self.classification_head = classification(input_dim)

        # initialize weights
        self.classification_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, inputs):

        out = self.classification_head(inputs)
        return out 
    

