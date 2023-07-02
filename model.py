import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from iconv import IConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers_to=1, activate="prelu"):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.num_layers_to = num_layers_to
        self.convs.append(GCNConv(in_channels, 512))
        if activate == "prelu":
            self.activates.append(nn.PReLU(512))
        elif activate == "relu":
            self.activates.append(nn.ReLU())
        
        if self.num_layers_to > 1:
            for i in range(num_layers_to - 1):
                self.convs.append(GCNConv(512, 512))
                if activate == "prelu":
                    self.activates.append(nn.PReLU(512))
                elif activate == "relu":
                    self.activates.append(nn.ReLU())

        self.hidden_channels = hidden_channels
        if self.hidden_channels < 512:
            self.convs.append(GCNConv(512, hidden_channels))
            if activate == "prelu":
                self.activates.append(nn.PReLU(hidden_channels))
            elif activate == "relu":
                self.activates.append(nn.ReLU())

    def forward(self, x, edge_index):
        for i in range(self.num_layers_to):
            x = self.convs[i](x, edge_index)
            x = self.activates[i](x)
        if self.hidden_channels < 512:
            x = self.convs[self.num_layers_to](x, edge_index)
            x = self.activates[self.num_layers_to](x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers_to=1, normalization=False, activate="prelu"):
        super(Encoder, self).__init__()
        self.m1 = GCN(in_channels, hidden_channels, num_layers_to=num_layers_to, activate=activate)
        self.prop = IConv(hidden_channels)
        self.norm = normalization

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        if self.norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    def propagation(self, x, edge_index):
        x = self.m1(x, edge_index)
        reg = self.prop(x, edge_index)
        if self.norm:
            reg = F.normalize(reg, p=2, dim=1)

        return reg

