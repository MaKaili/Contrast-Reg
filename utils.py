import argparse
import json
import os
import pickle
import csv
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, Yelp, Flickr
from wikics import WikiCS
from wikipedia_network import WikipediaNetwork


EPS=1e-15


class Mask(object):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


def load_dataset(dataset, transform=None):
    if dataset.lower() in ["cora", "citeseer", "pubmed"]:
        path = os.path.join(".datasets", "Plantoid")
        dataset = Planetoid(path, dataset.lower(), transform=transform)
    elif dataset.lower() in ["cs", "physics"]:
        path = os.path.join(".datasets", "Coauthor", dataset.lower())
        dataset = Coauthor(path, dataset.lower(), transform=transform)
    elif dataset.lower() in ["computers", "photo"]:
        path = os.path.join(".datasets", "Amazon", dataset.lower())
        dataset = Amazon(path, dataset.lower(), transform=transform)
    elif dataset.lower() in ["reddit"]:
        path = os.path.join(".datasets", "Reddit")
        dataset = Reddit(path, transform=transform)
    elif dataset.lower() in ["wikics"]:
        path = os.path.join(".datasets", "WikiCS")
        dataset = WikiCS(path, transform=transform)
    elif dataset.lower() in ["chameleon", "squirrel"]:
        path = os.path.join(".datasets", "WikipediaNetwork")
        dataset = WikipediaNetwork(path, dataset.lower(), transform=transform)
        #change /data2/home/hcyang/anaconda3/envs/py38-torch18/lib/python3.8/site-packages/torch_geometric/datasets/wikipedia_network.py for this
    elif dataset.lower() in ["yelp"]:
        path = os.path.join(".datasets", "Yelp")
        dataset = Yelp(path, transform=transform)
    elif dataset.lower() in ["flickr"]:
        path = os.path.join(".datasets", "Flickr")
        dataset = Flickr(path, transform=transform)
    else:
        print("Dataset not supported!")
        assert False
    return dataset



def generate_split(dataset, seed=0, train_num_per_c=20, val_num_per_c=30):
    torch.manual_seed(seed)
    dataset = load_dataset(dataset)
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def generate_percent_split(dataset, seed, train_percent=70, val_percent=20):
    torch.manual_seed(seed)
    dataset = load_dataset(dataset)
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero().flatten()
        num_c = all_c_idx.size(0)
        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask


def load_split(path, name, data, split):
    if name in ["chameleon", "squirrel"]:
        splits = data.train_mask[:,split], data.val_mask[:,split], data.test_mask[:,split]
    elif name == "wikics":
        splits = data.train_mask[:,split], data.val_mask[:,split], data.test_mask
    else:
        file_path = osp.join(path, name+str(split)+'.pt')
        if not osp.exists(file_path):
            splits = generate_split(name, split)
        else:
            splits = torch.load(osp.join(path, name+str(split)+'.pt'))
    return splits

def negative_sampling(num_nodes, sample_times):
    sample_list = []
    for j in range(sample_times):
        sample_iter = []
        i = 0
        while True:
            randnum = np.random.randint(0,num_nodes)
            if randnum!=i:
                sample_iter.append(randnum)
                i = i+1
            if len(sample_iter)==num_nodes:
                break
        sample_list.append(sample_iter)
    return sample_list

def mkdir_(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_TSNE(file_name, pos_embed, neg_embed, pos_label, num_classes, weight=None):
    from sklearn.manifold import TSNE
    emb_dim = pos_embed.size(0)
    num_class = len(list(set(pos_label)))
    sns.set(rc={'figure.figsize':(11.7, 8.27)})
    palette = sns.color_palette("bright", num_classes)

    pos_embed, _ = pos_embed
    if weight is not None:
        pos_embed = torch.matmul(pos_embed, weight)
        neg_embed = torch.matmul(neg_embed, weight)

    tsne = TSNE()
    embed = torch.cat((pos_embed, neg_embed), dim=0).to('cpu').detach().numpy()
    label = torch.cat((pos_label.to('cpu'), num_classes*torch.ones(emb_dim).long())).detach().numpy()
    emb = tsne.fit_transform(embed)
    np.save(file_name+"_embed.npy", embed)
    np.save(file_name+"_label.npy", label)
    np.save(file_name+"_emb.npy", emb)

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.scatter(emb[:emb_dim, 0], emb[:emb_dim, 1], c=label[:emb_dim])
    ax = fig.add_subplot(2,2,2)
    ax.scatter(emb[emb_dim:, 0], emb[emb_dim:, 1], c=label[emb_dim:])
    ax = fig.add_subplot(1,1,1)
    ax.scatter(emb[:, 0], emb[:, 1], c=label)
    plt.savefig(file_name+".pdf")

def create_csv(csv_path, whole=True):
    # whole to control the first line
    with open(csv_path,'w') as f:
        f.seek(0)
        f.truncate()
        csv_write = csv.writer(f)
        if whole is True:
            csv_head = ["test_acc_mean", "test_acc_std", "val_acc_mean", "val_acc_std",\
                        "args.contrast_model", "args.hidden", "args.epochs", "args.patience", "args.activate", "args.lr", "args.weight_decay", "args.reg", "args.ratio", \
                        "args.pre_norm", "args.final_norm", "args.normalization", "args.mu", "args.norm_penalty", "args.penalty_ratio", "args.dgi", "args.dgi_contrast", "args.sig_contrast", "args.norm_contrast", \
                        "args.mu_contrast", "args.num_layers_to", "args.noise_struct", "args.modify_ratio"]
        else:
            csv_head = ["seed", "test_acc", "val_acc", "best_loss_test_acc", "best_loss_val_acc", "best_loss_epoch"]
        csv_write.writerow(csv_head)
