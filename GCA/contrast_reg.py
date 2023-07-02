import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform

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

    def __init__(self, model, reg_vec, corruption, hidden, reg):
        super().__init__()
        self.model = model
        self.reg_vec = reg_vec
        self.corruption = corruption
        self.hidden_channels = hidden
        self.reg = reg

        if self.reg:
            self.weight = Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.reg:
            uniform(self.hidden_channels, self.weight)

    def forward(self, x, edge_index, return_neg=True):
        """Returns the latent space for the input arguments, their
        corruptions and their reg_vec representation."""
        pos_z = self.model.encoder(x, edge_index)

        if not self.reg and not return_neg:
            reg_vec = None
            neg_z = None
        else:
            cor = self.corruption(x, edge_index)
            cor = cor if isinstance(cor, tuple) else (cor, )
            neg_z = self.model.encoder(*cor)
            reg_vec = self.reg_vec(pos_z)

        return pos_z, neg_z, reg_vec

    def inference(self, data):
        z = self.model.encoder(data.x, data.edge_index)
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

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
