import torch
import torch.nn as nn
import torch.nn.functional as F



class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j,device):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0) 
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) 
        sim_ij = torch.diag(similarity, batch_size) 
        sim_ji = torch.diag(similarity, -batch_size) 
        positives = torch.cat([sim_ij, sim_ji], dim=0) 

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(device) 
        numerator = torch.exp(positives / self.temperature) 
        denominator = mask * torch.exp(similarity / self.temperature)        
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))

        loss = torch.sum(all_losses) / (2 * batch_size) 
        return loss

class NTXent1(nn.Module):
    def __init__(self, temperature=1.5):#1.0):

        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j,device):

        batch_size = z_i.size(0)

        similarity = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) 

        sim_ij = torch.diag(similarity)
        positives = sim_ij
        mask = (~torch.eye(batch_size , batch_size, dtype=torch.bool)).float().to(device) 
        numerator = torch.exp(positives / self.temperature) 
        denominator = mask * torch.exp(similarity / self.temperature)        
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (batch_size) 
        return loss    

class cross_entropy(nn.Module):
    def __init__(self):#1.0):

        super().__init__()

    def forward(self, output, target,device):

        loss=nn.BCEWithLogitsLoss()
        target=target.to(device)

        final_loss = loss(output,target)
        return final_loss    
