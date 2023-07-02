import torch
from torch.nn import Parameter
from utils import negative_sampling
import torch.nn.functional as F

from torch_geometric.nn.inits import reset, uniform
import math
import numpy as np

EPS = 1e-15


class LC(torch.nn.Module):
    r"""Local Clustering
    Args:
        use_curri (bool): For curriculum learning
        total_epoch (int): For curriculum learning
        curri_round (int): For curriculum learning

    """

    def __init__(self, args):
        super().__init__()
        self.curri_round = args.curri_round
        self.cut_round = math.ceil(args.epochs/(args.curri_round + 0.))
        self.use_curri = args.use_curri
        self.bilinear = args.bilinear

    def forward(self, z_tuple, mu=1., contrast_weight=None, epoch=0):
        z, _ = z_tuple
        P_mat = self.calculate_P_matrix(z)
        _, index = torch.topk(P_mat, 5, dim = 1)
        neg_list = torch.randperm(z.size(0)).to(z.device)

        pos_z = torch.mean(z[index], dim=1)
        neg_z = torch.index_select(z, 0, neg_list)

        if self.use_curri:
            # curriculum learning
            epoch = epoch-1
            if epoch%(self.curri_round) == 0:
                r = math.floor(epoch/self.cut_round) + 1.
                S = int(np.floor((r*1.0/self.curri_round)*(z.size(0))))
                P_mat = self.calculate_P_matrix(z)
                with torch.no_grad():
                    entropy_list = (-P_mat*torch.log(P_mat + EPS)).sum(dim = 1)

                _, entropy_index = torch.topk(-entropy_list, S, dim = 0)


            pos_z = torch.index_select(pos_z, 0, entropy_index)
            neg_z = torch.index_select(neg_z, 0, entropy_index)
            z = torch.index_select(z, 0, entropy_index)

        z = z.unsqueeze(-1)
        if not self.bilinear:
            pos_loss = -torch.log(
                torch.sigmoid(torch.matmul(pos_z.unsqueeze(1), z/mu)) + EPS).mean().squeeze()
            neg_loss = -torch.log(
                1-torch.sigmoid(torch.matmul(neg_z.unsqueeze(1), z/mu)) + EPS).mean().squeeze()
        else:
            pos_loss = -torch.log(
                torch.sigmoid(torch.matmul(pos_z.unsqueeze(1), torch.matmul(contrast_weight, z/mu))) + EPS).mean().squeeze()
            neg_loss = -torch.log(
                1-torch.sigmoid(torch.matmul(neg_z.unsqueeze(1), torch.matmul(contrast_weight, z/mu))) + EPS).mean().squeeze()

        return pos_loss+neg_loss

    def calculate_P_matrix(self, x):
        x = F.normalize(x, p=2., dim=-1)
        self_P_matrix = torch.matmul(x, torch.transpose(x, 0, 1))
        self_P_matrix = F.normalize(self_P_matrix, p=1., dim=-1)
        return self_P_matrix

class ML(torch.nn.Module):
    r"""Multi Level representation alignment

    """

    def __init__(self, args):
        super().__init__()
        self.bilinear = args.bilinear
        pass

    def forward(self, z_tuple, mu=1., contrast_weight=None):
        # mu is used to test the baseline of l2 with tempetature coeffient.
        pos_z, prop_z = z_tuple
        neg_list = torch.randperm(pos_z.size(0))
        neg_z = pos_z[neg_list]
        prop_z = prop_z.unsqueeze(-1)
        if not self.bilinear:
            pos_loss = -torch.log(
                torch.sigmoid(torch.matmul(pos_z.unsqueeze(1), prop_z/mu)) + EPS).mean()
            neg_loss = -torch.log(
                1-torch.sigmoid(torch.matmul(neg_z.unsqueeze(1), prop_z/mu)) + EPS).mean()
        else:
            pos_loss = -torch.log(
                torch.sigmoid(torch.matmul(pos_z.unsqueeze(1), torch.matmul(contrast_weight, prop_z/mu))) + EPS).mean()
            neg_loss = -torch.log(
                1-torch.sigmoid(torch.matmul(neg_z.unsqueeze(1), torch.matmul(contrast_weight, prop_z/mu))) + EPS).mean() 

        return pos_loss + neg_loss

class CO(torch.nn.Module):
    r""" Co-Occurance defined by first order proximity

    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bilinear = args.bilinear
        pass

    def logits(self, x, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        if self.args.sig_contrast:
            x_j = torch.sigmoid(x_j)
        if self.args.norm_contrast:
            x_j = F.normalize(x_j, p=2, dim=1)
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                  neg_edge_index.size(1)).float()
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def forward(self, z_tuple, mu=1., contrast_weight=None, pos_edge_index=None, neg_edge_index=None):
        z, _ = z_tuple
        link_logits = self.logits(z, pos_edge_index, neg_edge_index)
        link_labels = self.get_link_labels(pos_edge_index, neg_edge_index).to(z.device)

        loss_contrast = F.binary_cross_entropy_with_logits(link_logits/mu, link_labels)
        return loss_contrast
