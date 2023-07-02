import os
import os.path as osp
import argparse
import scipy.sparse as sp
import numpy as np
import random
import math
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.utils as U
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from utils import load_dataset, load_split, mkdir_, plot_TSNE, create_csv

from torch.utils.tensorboard import SummaryWriter
import csv
from save_data import Record, cal_norm

import torch_geometric.transforms as T
from Contrast_Reg import Contrast_Reg
from Contrast_loss import ML, CO, LC
from model import Encoder
from classifier import Node_Class
import json


EPS = 1e-15

def set_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    # General
    parser.add_argument('--dataset', type=str, default = 'cora')
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument('--pre-norm', action="store_true") # dataset dependent
    # Optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--activate", type=str, default='relu', choices=["relu", "prelu"])
    parser.add_argument("--num-layers-to", type=int, default=1)
    # For LC model
    parser.add_argument('--use-curri', action="store_true")
    parser.add_argument("--curri-round", type=int, default=1)
    parser.add_argument("--lr-deduct", type=float, default=0.5)
    # Contrast loss selection
    parser.add_argument("--contrast-model", type=str, default="ML", choices=["CO", "ML", "LC"])
    # Reg
    parser.add_argument('--reg', action="store_true")
    parser.add_argument("--ratio", type=float, default=1.)
    # Analyze, visualization and saving data
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--mu", type=float, default=1.)
    parser.add_argument('--bilinear', action="store_true", help="only support ML and LC for now")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument("--write-tb", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--save-norm", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--save-embed", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--path-split", type=str, default="splits")
    parser.add_argument("--save-path", type=str, default="")
    # For baseline
    parser.add_argument("--final-norm", action="store_true")# directly normalize representation in test stage
    parser.add_argument("--normalization", action="store_true")#have effect on optimization, corresponding to l2 norm
    parser.add_argument('--dgi', action="store_true")#baseline: dgi
    parser.add_argument('--dgi-contrast', action="store_true")#baseline: dgi+contrast, useful when dgi set to true
    parser.add_argument('--norm-penalty', action="store_true")#baseline: l2norm penalty
    parser.add_argument('--sig-contrast', action="store_true")#baseline: sigmoid for contrast
    parser.add_argument('--norm-contrast', action="store_true")#baseline: before contrast, do the norm
    parser.add_argument('--mu-contrast', type=float, default=1.)#baseline: before contrast, do the norm
    parser.add_argument("--penalty-ratio", type=float, default=0.1)#baseline: l2norm penalty ratio. 0.1 for LC, 0.01 for ML

    # For graph structure noise
    parser.add_argument('--noise-struct', action="store_true")#add noise to the graph struct
    parser.add_argument("--modify-ratio", type=float, default=0.25)#noise ratio for edges

    parser.add_argument("--run-name", type=str, default="run", help='give a run name')
    args = parser.parse_args()
    #c-m : contrast-model, hd : hidden
    #eps : epochs, w-d : weight_decay
    #pat : patience, u-cu : use_curri
    #cu-r : curri_round, lr-de : lr_deduct
    #p-nm : pre_norm, f-nm: final_norm
    #norm: normalization, act: activation 
    #dgi: dgi_baseline, nm-pn: norm_penalty_baseline
    #pn-rt: penalty_ratio, #n-l: num_layers_to
    #ns: noise_struct, mr:modify_ratio
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    args.path_name = "{}_{}_{}".format(args.dataset.lower(), args.contrast_model, args.run_name)
    # args.path_name = args.dataset.lower()+"_c-m_"+args.contrast_model+"_hd_"+str(args.hidden)\
    #                  +"_n-sds_"+str(args.num_seeds)+'_eps_'+str(args.epochs)\
    #                  +'_lr_'+str(args.lr)+"_wd_"+str(args.weight_decay)+"_pat_"+str(args.patience)\
    #                  +'_u-cu_'+str(args.use_curri)+'_cu-r_'+str(args.curri_round)+'_lr-de_'+str(args.lr_deduct)\
    #                  +"_reg_"+str(args.reg)+'_ratio_'+str(args.ratio)\
    #                  +'_p-nm_'+str(args.pre_norm)+'_f-nm_'+str(args.final_norm)+"_norm_"+str(args.normalization)+"_mu_"+str(args.mu)+"_act_"+args.activate\
    #                  +'_dgi_'+str(args.dgi)+'_dgicontrast_'+str(args.dgi_contrast)+'_sig_contrast_'+str(args.sig_contrast)+'_nm-pn_'+str(args.norm_penalty)+'_pn-rt_'+str(args.penalty_ratio)+'_n-l_'+str(args.num_layers_to)\
    #                 +'_ns_'+str(args.noise_struct)+'_mr_'+str(args.modify_ratio)
    args.save_name = os.path.join("csv", args.save_path, args.dataset.lower(), args.contrast_model, "node_class", args.path_name)
    args.csv_path = os.path.join(args.save_name, args.path_name +'.csv')
    # args.csv_path = os.path.join("csv", args.save_path, args.path_name +'.csv')
    mkdir_(args.save_name)
    with open(os.path.join(args.save_name, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)
    if args.save_csv:
        create_csv(args.csv_path)
    if args.write_tb:
        args.tb_path = os.path.join("tb", args.save_path, args.path_name)

    if args.device == -1:
        args.device = 'cpu'

    if args.pre_norm:
        dataset = load_dataset(args.dataset, transform = T.NormalizeFeatures())
    else:
        dataset = load_dataset(args.dataset)

    args.num_classes = dataset.num_classes

    if args.contrast_model == "CO":
        args.prop = False
    elif args.contrast_model == "ML":
        args.prop = True
    elif args.contrast_model == "LC":
        args.prop = False

    return args, dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def train(model, data, data_partial, optimizer, args, epoch, recode):
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, reg_vec = model(data, data_partial)

    # Plotting TSNE figure for every 50 epochs
    if args.tsne and (epoch % 50==1):
        save_name = os.path.join("tsne", args.save_path, args.path_name)
        mkdir_(save_name)
        z, _ = pos_z
        for name, param in model.named_parameters():
            if name == "weight":
                weight = param.data.detach()
        plot_TSNE(save_name, pos_z, neg_z, data.y, num_class=dataset.num_classes, weight=weight)

    if recode is not None:
        z, _ = pos_z
        recode.update_norm(z)
        for name, param in model.named_parameters():
            if name == "weight":
                weight = param.data.detach()
                recode.update_logits(z, weight, reg_vec.to(args.device))

    # Passing different parameters to different model
    if args.contrast_model == "CO":
        pos_edge_index = data_partial.train_pos_edge_index
        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index, num_nodes=data.x.size(0))
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=data.x.size(0),
            num_neg_samples=pos_edge_index.size(1))
        kwargs = {"pos_edge_index": pos_edge_index, "neg_edge_index": neg_edge_index}
    elif args.contrast_model == "LC":
        kwargs = {"epoch": epoch}
    elif args.contrast_model == "ML":
        kwargs = {}
    else:
        raise NotImplementedError("Contrast model wrong select")

    loss = model.loss(pos_z, neg_z, reg_vec, args, **kwargs)
    loss.backward()

    # for name, parms in model.named_parameters():	
    #     print("-->name: {}, -->grad_requirs: {}, -->grad_value: {}".format(name, parms.requires_grad, parms.grad))
    optimizer.step()
    return loss.item()

def test(model, data, splits):
    model.eval()
    z = model.inference(data)
    if args.final_norm:
        z = F.normalize(z, p=2., dim=-1)
    classifier = Node_Class(max_iter=150)
    test_acc, val_acc = classifier(z.cpu().detach().numpy(), data, splits)
    return test_acc, val_acc

if __name__ == "__main__":
    args, dataset = set_args()
    data = dataset[0].to(args.device)

    if args.plot or args.save_norm:
        recode = Record(args, len(data.x))
    else:
        recode = None

    #Loading standard split or 0 split in random splits just for debugging. We won't use any mask in all training procedure
    if args.dataset in ["cora", "citeseer", "pubmed", "yelp", "flickr"]:
        splits = data.train_mask, data.val_mask, data.test_mask
    elif args.dataset in ["chameleon", "squirrel"]:
        splits = data.train_mask[:,0], data.val_mask[:,0], data.test_mask[:,0]
    elif args.dataset == "wikics":
        splits = data.train_mask[:,0], data.val_mask[:,0], data.test_mask
    else:
        splits = torch.load(osp.join(args.path_split, args.dataset+str(0)+'.pt'))

    for seed in range(args.num_seeds):
        setup_seed(seed)

        if args.noise_struct:
            data.edge_index = torch.load(os.path.join("edge_noise", str(args.dataset.lower()), str(args.dataset.lower())+"_transform_None"+"_ratio_"+str(args.modify_ratio)+"_seed_"+str(seed)+".pt")).to(args.device)
        # Loading training positive and negative edges used in CO model
        if args.contrast_model == "CO":
            data_partial = torch.load("sage_data/"+args.dataset+str(seed)+'.pt').to(args.device)
        else:
            data_partial = None

        if args.write_tb:
            TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
            tb_path = args.tb_path+"_sed_{}/{}".format(seed, TIMESTAMP)
            mkdir_(tb_path)
            sw = SummaryWriter(tb_path)

        if args.dgi:
            func = lambda z: torch.sigmoid(z.mean(dim=0))
            if args.dgi_contrast:
                contrast_cls = globals()[args.contrast_model]
                contrast_loss = contrast_cls(args)
            else:
                contrast_loss = None
            args.reg = True
        else:
            func = lambda z: args.scale*torch.rand(z.size(1))
            contrast_cls = globals()[args.contrast_model]
            contrast_loss = contrast_cls(args)

        model = Contrast_Reg(encoder=Encoder(dataset.num_features, args.hidden, args.num_layers_to, 
                                            args.normalization, args.activate),
                            reg_vec=func, corruption=corruption, contrast_loss=contrast_loss, args=args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

        best_loss = 10000.
        best_loss_epoch = 0
        cnt_wait = 0
        best_loss_val_acc = best_loss_test_acc = best_loss_epoch = 0
        patience = args.patience
        cut_round = math.ceil(args.epochs/(args.curri_round + 0.))

        for epoch in range(1, args.epochs+1):
            if args.use_curri and epoch - 1 != 0 and (epoch - 1)%cut_round == 0:
                lr = args.lr*args.lr_deduct
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = args.weight_decay)

            loss = train(model, data, data_partial, optimizer, args, epoch, recode)

            # The grid search is based on the performance of node classification
            test_acc, val_acc = test(model, data, splits)
            print('Epoch: {:03d}, Loss: {:.4f}, Test_acc: {:.4f}, Val_acc: {:.4f}'.format(
                                                              epoch, loss, test_acc, val_acc))

            if args.write_tb:
                sw.add_scalar("Accuracy/test", test_acc, epoch)
                sw.add_scalar("Accuracy/validate", val_acc, epoch)
                sw.add_scalar("Loss/train", loss, epoch)
                norm_z = cal_norm(model, data)
                sw.add_scalar("Norm/average", norm_z, epoch)

            # Using training loss for early stopping
            if args.save_norm or args.plot:
                recode.update_train(loss, test_acc, val_acc)

            if loss < best_loss:
                cnt_wait = 0
                best_loss = loss

            if loss <= best_loss:
                best_loss_val_acc = val_acc
                best_loss_test_acc = test_acc
                best_loss_epoch = epoch

                # Saving model for downstream tasks
                if args.save_model:
                    weight_checkpoint = model.weight if args.reg else None
                    state = {
                        "args": args,
                        "model": model.state_dict(),
                        "encoder": model.encoder.state_dict(),
                        "gnn": model.encoder.m1.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "weight": weight_checkpoint
                    }
                    save_model_path = os.path.join("saved_model", args.save_path,  args.dataset.lower(), args.contrast_model, "node_class", args.path_name)
                    path = os.path.join(save_model_path, args.path_name+"_seed_{}.pt".format(seed))
                    mkdir_(save_model_path)
                    torch.save(state, path)
                    with open(os.path.join(save_model_path, "args.json"), 'w') as f:
                        json.dump(vars(args), f, indent=True)

                # Saving embedding for dowanstream tasks
                if args.save_embed:
                    pos_z, _, _ = model(data.x, data.edge_index)
                    z, _ = pos_z
                    z = z.to('cpu').detach().numpy()
                    path = os.path.join("saved_embed", args.save_path, args.path_name+"_seed_{}.npy".format(seed))
                    mkdir_(os.path.join("saved_embed", args.save_path))
                    np.save(path, z)

            else:
                cnt_wait += 1

            if cnt_wait > patience:
                break

        print('best_loss_test_acc: {:.4f}, best_loss_val_acc: {:.4f}, best_loss_epoch: {:03d}'.format(best_loss_test_acc, best_loss_val_acc, best_loss_epoch))

        # Saving data or plotting
        if args.save_csv:
            with open(args.csv_path,'a+') as f:
                csv_write = csv.writer(f)
                data_row = [seed, test_acc, val_acc, best_loss_test_acc, best_loss_val_acc, best_loss_epoch]
                csv_write.writerow(data_row)

        if args.save_norm:
            recode.save()

        if args.plot:
            recode.plot()

        if recode is not None:
            recode.reset()
