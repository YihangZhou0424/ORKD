import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RKDLoss', 'GCN']


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class RKDLoss(nn.Module):
    def __init__(self):
        super(RKDLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.kd = DistillKL(T=0.1)

    def forward(self, s_embeddings):
        s_angle_relations = []
        t_angle_relations = []
        for i in range(len(s_embeddings)):
            s_embeddings[i] = s_embeddings[i].view(s_embeddings[i].shape[0], -1)
            s_relation = self.pdist(s_embeddings[i], squared=False)
            s_mean_td = s_relation[s_relation > 0].mean()
            s_relation = s_relation / s_mean_td
            
            s_norm_embeddings = F.normalize(s_embeddings[i], p=2, dim=1)
            s_angle = torch.mm(s_norm_embeddings, s_norm_embeddings.transpose(0, 1))
            s_angle_relations.append(s_angle)

        loss_a = 0.
                    
        for i in range(len(s_angle_relations)):
            for j in range(len(s_angle_relations)):
                loss_a += F.mse_loss(s_angle_relations[i], s_angle_relations[j].detach())

        return loss_a/len(s_angle_relations)**2
    
        

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res