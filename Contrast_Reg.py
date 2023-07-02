import torch
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from utils import negative_sampling
import torch.nn.functional as F

from torch_geometric.nn.inits import reset, uniform
import math
import numpy as np

EPS = 1e-15


class Contrast_Reg(torch.nn.Module):
    r"""Contrast-Reg
    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        reg_vec (callable): The vector used in .
        corruption (callable): Shuffling node features among nodes.
        contrast_loss (Module): Contrast loss
        reg (bool): Indicator of adding reg loss
        prop (bool): Indicator of calculating propogation for ML contrast loss

    """

    def __init__(self, encoder, reg_vec, corruption, contrast_loss, args):
        super().__init__()
        self.hidden_channels = args.hidden
        self.encoder = encoder
        self.reg_vec = reg_vec
        self.corruption = corruption
        self.reg = args.reg
        self.prop = args.prop
        self.bilinear = args.bilinear

        if self.reg:
            self.weight = Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))
        
        if self.bilinear :
            self.contrast_weight = Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))
        else:
            self.contrast_weight = None

        self.contrast_loss = contrast_loss

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        if self.reg:
            uniform(self.hidden_channels, self.weight)
        if self.bilinear:
            uniform(self.hidden_channels, self.contrast_weight)


    def forward(self, data, data_partial=None):
        """Returns the latent space for the input arguments, their
        corruptions and their reg_vec representation."""
        if data_partial is not None:
            pos_z = self.encoder(data_partial.x, data_partial.train_pos_edge_index)
        else:
            pos_z = self.encoder(data.x, data.edge_index)

        if self.prop:
            prop_z = self.encoder.propagation(data.x, data.edge_index)
            pos_z_tuple = (pos_z, prop_z)
        else:
            pos_z_tuple = (pos_z, None)

        if not self.reg:
            reg_vec = None
            neg_z = None
        else:
            cor = self.corruption(data.x, data.edge_index)
            cor = cor if isinstance(cor, tuple) else (cor, )
            neg_z = self.encoder(*cor)
            reg_vec = self.reg_vec(pos_z)

        return pos_z_tuple, neg_z, reg_vec

    def inference(self, data):
        z = self.encoder(data.x, data.edge_index)
        return z

    def discriminate(self, z, reg_vec, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, reg_vec))
        return torch.sigmoid(value) if sigmoid else value

    def loss_reg(self, pos_z, neg_z, reg_vec):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, reg_vec, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, reg_vec, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def loss(self, pos_z_tuple, neg_z, reg_vec, args, **kwargs):
        mu = args.mu
        ratio = args.ratio

        penalty_ratio = args.penalty_ratio

        pos_z, prop_z = pos_z_tuple
        if args.sig_contrast and prop_z is not None:
            prop_z_sig = torch.sigmoid(prop_z)
            pos_z_tuple = (pos_z, prop_z_sig/args.mu_contrast)
        elif args.norm_contrast and prop_z is not None:
            prop_z_norm = F.normalize(prop_z, p=2, dim=1)
            pos_z_tuple = (pos_z, prop_z_norm/args.mu_contrast)
        contrast_loss = 0.
        if self.contrast_loss is not None:
            contrast_loss = self.contrast_loss(pos_z_tuple, mu=mu, contrast_weight=self.contrast_weight, **kwargs)

        reg_loss = 0
        if self.reg:
            reg_loss = self.loss_reg(pos_z, neg_z, reg_vec.to(pos_z.device))

        norm_loss = 0.
        if args.norm_penalty:
            norm_loss = torch.norm(pos_z.squeeze(), p=2, dim=1).mean().squeeze()
            print("norm loss: {:.4f}".format(norm_loss.item()))
        loss = contrast_loss+ratio*reg_loss+penalty_ratio*norm_loss
        return loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
