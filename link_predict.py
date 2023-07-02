import torch
import torch.nn as nn


class LinkPred(nn.Module):
    def __init__(self, dataset, load_emb, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.):
        super(LinkPred, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.dataset = da

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def _mlp(self, x_i, x_j):
        """Calculating the probability of existance of one edge in pair (x_i, x_j)
            First do the element-wise multiplication and then apply one mlp network.
            Inherent from the example in ogb.
        """
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def train(self):




    def forward(self):

    @torch.no_grad()
    def evaluate(self, embeds, data):


    def load_emb(self):




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argumemt("--model", type=str, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--hidden-channels", type=int, required=True)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--load-emb", type=str, required=True)
    args = parser.parse_args()
    task = LinkPred(
        args.
    )



