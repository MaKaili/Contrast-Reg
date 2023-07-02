from sklearn.linear_model import LogisticRegression
import torch
from utils import load_dataset, load_split, create_csv, mkdir_
import numpy as np
import argparse
import os
from model import Encoder
import csv
import torch_geometric.transforms as T
import json
import time

class Node_Class(torch.nn.Module):
    r"""Node classification task."""
    def __init__(self, solver="lbfgs", multi_class="auto", max_iter=150):
        super().__init__()
        self.solver = solver
        self.multi_class = multi_class
        self.max_iter = max_iter

    def forward(self, z, data, splits):
        (train_mask, val_mask, test_mask) = splits
        train_mask = train_mask.cpu().numpy()
        val_mask   = val_mask.cpu().numpy()
        test_mask  = test_mask.cpu().numpy()

        data.y = data.y
        train_z = z[train_mask]
        train_y = data.y[train_mask]

        train_y = train_y.cpu()
        clf = LogisticRegression(solver=self.solver, multi_class=self.multi_class, max_iter=self.max_iter).fit(train_z, train_y)

        val_z = z[val_mask]
        val_y = data.y[val_mask]
        val_y = val_y.cpu()

        test_z = z[test_mask]
        test_y = data.y[test_mask]
        test_y = test_y.cpu()
        
        test_acc = clf.score(test_z, test_y)
        val_acc = clf.score(val_z, val_y)

        return test_acc, val_acc

    def load_model(self, model, data, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["encoder"])
        model.eval()
        # inference
        z = model(data.x, data.edge_index)
        z = z.detach().cpu().numpy()
        return z

    def load_embed(self, path):
        embed = np.load(path)
        return embed