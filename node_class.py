from sklearn.linear_model import LogisticRegression
import torch
from main import test
from utils import load_dataset, load_split, create_csv, mkdir_
import numpy as np
import argparse
import os
from model import Encoder
import csv
import torch_geometric.transforms as T
import json
import time
from classifier import Node_Class

# class Node_Class(torch.nn.Module):
#     r"""Node classification task."""
#     def __init__(self, solver="lbfgs", multi_class="auto", max_iter=150):
#         super().__init__()
#         self.solver = solver
#         self.multi_class = multi_class
#         self.max_iter = max_iter

#     def forward(self, z, data, splits):
#         (train_mask, val_mask, test_mask) = splits
#         train_mask = train_mask.cpu().numpy()
#         val_mask   = val_mask.cpu().numpy()
#         test_mask  = test_mask.cpu().numpy()

#         data.y = data.y
#         train_z = z[train_mask]
#         train_y = data.y[train_mask]

#         train_y = train_y.cpu()
#         clf = LogisticRegression(solver=self.solver, multi_class=self.multi_class, max_iter=self.max_iter).fit(train_z, train_y)

#         val_z = z[val_mask]
#         val_y = data.y[val_mask]
#         val_y = val_y.cpu()

#         test_z = z[test_mask]
#         test_y = data.y[test_mask]
#         test_y = test_y.cpu()
        
#         test_acc = clf.score(test_z, test_y)
#         val_acc = clf.score(val_z, val_y)

#         return test_acc, val_acc

#     def load_model(self, model, data, path):
#         checkpoint = torch.load(path)
#         model.load_state_dict(checkpoint["encoder"])
#         model.eval()
#         # inference
#         z = model(data.x, data.edge_index)
#         z = z.detach().cpu().numpy()
#         return z

#     def load_embed(self, path):
#         embed = np.load(path)
#         return embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--num-splits", type=int, help="number of splits for node classification training", required=True)

    parser.add_argument('--dataset', type=str, default = 'cora')
    parser.add_argument("--num-seeds", type=int, required=True)
    # parser.add_argument("--hidden", type=int, default=512)
    # parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument('--pre-norm', action="store_true") # dataset dependent
    # # Optimization
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--weight-decay", type=float, default=0)
    # parser.add_argument("--patience", type=int, default=40)
    # parser.add_argument("--activate", type=str, default='relu', choices=["relu", "prelu"])
    # parser.add_argument("--num-layers-to", type=int, default=1)
    # # For LC model
    # parser.add_argument('--use-curri', action="store_true")
    # parser.add_argument("--curri-round", type=int, default=1)
    # parser.add_argument("--lr-deduct", type=float, default=0.5)
    # # Contrast loss selection
    # parser.add_argument("--contrast-model", type=str, default="ML", choices=["CO", "ML", "LC"])
    # # Reg
    # parser.add_argument('--reg', action="store_true")
    # parser.add_argument("--ratio", type=float, default=1.)
    # # Analyze, visualization and saving data
    # parser.add_argument("--scale", type=float, default=1.)
    # parser.add_argument("--mu", type=float, default=1.)
    # parser.add_argument('--plot', action="store_true")
    # parser.add_argument("--path-split", type=str, default="splits")
    # parser.add_argument("--save-path", type=str, default="")
    # # For baseline
    # parser.add_argument("--final-norm", action="store_true")# directly normalize representation in test stage
    # parser.add_argument("--normalization", action="store_true")#have effect on optimization, corresponding to l2 norm
    # parser.add_argument('--dgi', action="store_true")#baseline: dgi
    # parser.add_argument('--dgi-contrast', action="store_true")#baseline: dgi+contrast, useful when dgi set to true
    # parser.add_argument('--sig-contrast', action="store_true")#baseline: sigmoid for contrast
    # parser.add_argument('--norm-contrast', action="store_true")#baseline: before contrast, do the norm
    # parser.add_argument('--norm-penalty', action="store_true")#baseline: l2norm penalty
    # parser.add_argument("--penalty-ratio", type=float, default=0.1)#baseline: l2norm penalty ratio. 0.1 for LC, 0.01 for ML

    # # For graph structure noise
    # parser.add_argument('--noise-struct', action="store_true")#add noise to the graph struct
    # parser.add_argument("--modify-ratio", type=float, default=0.25)#noise ratio for edges
    parser.add_argument("--path-split", type=str, default="splits")
    parser.add_argument("--task-folder", type=str, default=None, help="must give a folder to run", required=True)
    parser.add_argument("--csv-name", type=str, default='node_class_test_csv', help="csv file prefix name")

    args = parser.parse_args()
    #c-m : contrast-model, hd : hidden
    #eps : epochs, w-d : weight_decay
    #pat : patience, u-cu : use_curri
    #cu-r : curri_round, lr-de : lr_deduct
    #p-nm : pre_norm, f-nm: final_norm
    #norm: normalization, act: activation
    #dgi: dgi_baseline, nm-pn: norm_penalty_baseline
    #pn-rt: penalty_ratio, #n-l: num_layers_to

    assert (args.task_folder is not None), "task folder can not be not given"
    assert (args.dataset is not None), "dataset can not be not given"
    # assert (args.load_model is True), "not supported if not load model"

    model_files = os.listdir(args.task_folder)
    args.csv_path = os.path.join(args.task_folder, args.dataset + args.csv_name + '.csv')


    # args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # args.path_name = "{}_{}_{}".format(args.dataset.lower(), args.contrast_model, args.run_name)
    # # args.path_name = args.dataset.lower()+"_c-m_"+args.contrast_model+"_hd_"+str(args.hidden)\
    # #                  +"_n-sds_"+str(args.num_seeds)+'_eps_'+str(args.epochs)\
    # #                  +'_lr_'+str(args.lr)+"_wd_"+str(args.weight_decay)+"_pat_"+str(args.patience)\
    # #                  +'_u-cu_'+str(args.use_curri)+'_cu-r_'+str(args.curri_round)+'_lr-de_'+str(args.lr_deduct)\
    # #                  +"_reg_"+str(args.reg)+'_ratio_'+str(args.ratio)\
    # #                  +'_p-nm_'+str(args.pre_norm)+'_f-nm_'+str(args.final_norm)+"_norm_"+str(args.normalization)+"_mu_"+str(args.mu)+"_act_"+args.activate\
    # #                  +'_dgi_'+str(args.dgi)+'_nm-pn_'+str(args.norm_penalty)+'_pn-rt_'+str(args.penalty_ratio)+'_n-l_'+str(args.num_layers_to)\
    # #                  +'_ns_'+str(args.noise_struct)+'_mr_'+str(args.modify_ratio)
    # args.save_name = os.path.join("csv", args.save_path, args.dataset.lower(), args.contrast_model, "node_class/test")
    # args.csv_path = os.path.join(args.save_name, args.path_name +'.csv')
    # mkdir_(args.save_name)
    create_csv(args.csv_path, whole=True)

    for model_folder in model_files:
        if os.path.splitext(model_folder)[1] == '.csv':
            print(model_folder)
            continue
        single_csv_path = os.path.join(args.task_folder, model_folder, model_folder + '.csv')
        # create_csv(single_csv_path, whole=False)
        args_file = os.path.join(args.task_folder, model_folder, "args.json")
        if not os.path.isfile(args_file):
            continue
        with open(args_file, 'r') as f:
            init_model_args = json.load(f)

        #preserve the parameters from loaded model
        assert args.dataset == init_model_args['dataset'], "dataset not same error"
        args.lr = init_model_args['lr']
        args.mu = init_model_args['mu']
        args.hidden = init_model_args['hidden']
        args.normalization = init_model_args['normalization']
        args.epochs = init_model_args['epochs']
        args.patience = init_model_args['patience']
        args.activate = init_model_args['activate']
        args.dgi = init_model_args['dgi']
        if args.dgi:
            args.dgi_contrast = init_model_args['dgi_contrast']
        else:
            args.dgi_contrast = False
        if 'sig_contrast' in init_model_args:
            args.sig_contrast = init_model_args['sig_contrast']
        if 'norm_contrast' in init_model_args:
            args.norm_contrast = init_model_args['norm_contrast']
        
        args.pre_norm = init_model_args['pre_norm']
        args.final_norm = init_model_args['final_norm']
        args.norm_penalty = init_model_args['norm_penalty']
        args.penalty_ratio = init_model_args['penalty_ratio']
        args.num_layers_to = init_model_args['num_layers_to']
        args.contrast_model = init_model_args['contrast_model']
        args.weight_decay = init_model_args['weight_decay']
        args.mu_contrast = init_model_args['mu_contrast']

        args.reg = init_model_args['reg']
        args.ratio = init_model_args['ratio']

        args.noise_struct = False 
        args.modify_ratio = False
        # with open(args.csv_path,'a+') as f:
        #     csv_write = csv.writer(f)
        #     data_row = [args.contrast_model, args.hidden, args.epochs, args.patience, args.activate, args.lr, args.weight_decay, args.reg, args.ratio, \
        #                 args.pre_norm, args.final_norm, args.normalization, args.mu, args.norm_penalty, args.penalty_ratio, args.dgi, args.dgi_contrast, args.sig_contrast, args.norm_contrast, \
        #                 args.num_layers_to, args.noise_struct, args.modify_ratio]
        #     csv_write.writerow(data_row)

        if args.pre_norm:
            dataset = load_dataset(args.dataset, transform = T.NormalizeFeatures())
        else:
            dataset = load_dataset(args.dataset)
        data = dataset[0]

        classifier = Node_Class(max_iter=150)

        test_accs = []
        val_accs = []
        if args.load_model:
            create_csv(single_csv_path, whole=False)
            for seed in range(args.num_seeds):
                if args.load_model:
                    # path = os.path.join("saved_model", args.save_path, args.dataset.lower(), args.contrast_model, "node_class", args.path_name, args.path_name+"_seed_{}.pt".format(seed))
                    path = os.path.join(args.task_folder, model_folder, model_folder+"_seed_{}.pt".format(seed))
                    model = Encoder(dataset.num_features, args.hidden, args.num_layers_to, args.normalization, args.activate)
                    z = classifier.load_model(model, data, path)
                else:
                    print("not supported")
                    exit()
                    # path = os.path.join("saved_embed", args.save_path, args.path_name+"_seed_{}.npy".format(seed))
                    # z = cluster.load_embed(path)
                
                if args.noise_struct:
                    data.edge_index = torch.load(os.path.join("edge_noise", str(args.dataset.lower()), str(args.dataset.lower())+"_transform_None"+"_ratio_"+str(args.modify_ratio)+"_seed_"+str(seed)+".pt"))

                for split in range(args.num_splits):
                    if args.num_splits == 1:
                        if args.dataset in ["cora", "citeseer", "pubmed", "yelp", "flickr"]:
                            splits = data.train_mask, data.val_mask, data.test_mask
                        elif args.dataset in ["chameleon", "squirrel"]:
                            splits = data.train_mask[:,0], data.val_mask[:,0], data.test_mask[:,0]
                        elif args.dataset == "wikics":
                            splits = data.train_mask[:,0], data.val_mask[:,0], data.test_mask
                        else:
                            splits = torch.load(osp.join(args.path_split, args.dataset+str(0)+'.pt'))
                    else:
                        splits = load_split(args.path_split, args.dataset, data, split)

                    test_acc, val_acc = classifier(z, data, splits)

                    test_accs.append(test_acc)
                    val_accs.append(val_acc)

                    with open(single_csv_path,'a+') as f:
                        csv_write = csv.writer(f)
                        data_row = [seed, split, test_acc, val_acc]
                        csv_write.writerow(data_row)

            print("test_acc: ", test_accs)
            print("val_acc: ", val_accs)
            print("test acc mean: {:.4f}, std: {:.4f}".format(np.mean(test_accs), np.std(test_accs)))
            print("val acc mean: {:.4f}, std: {:.4f}".format(np.mean(val_accs), np.std(val_accs)))
            test_acc_mean = np.mean(test_accs)
            test_acc_std = np.std(test_accs)
            val_acc_mean = np.mean(val_accs)
            val_acc_std = np.std(val_accs)
            with open(args.csv_path,'a+') as f:
                csv_write = csv.writer(f)
                data_row = [test_acc_mean, test_acc_std, val_acc_mean, val_acc_std, \
                            args.contrast_model, args.hidden, args.epochs, args.patience, args.activate, args.lr, args.weight_decay, args.reg, args.ratio, \
                            args.pre_norm, args.final_norm, args.normalization, args.mu, args.norm_penalty, args.penalty_ratio, args.dgi, args.dgi_contrast, args.sig_contrast, args.norm_contrast, \
                            args.mu_contrast, args.num_layers_to, args.noise_struct, args.modify_ratio, model_folder]
                csv_write.writerow(data_row)
        else:
            # directly get from csv
            with open(single_csv_path,'r') as f:
                reader = csv.reader(f)
                test_acc = [row[1] for row in reader]
                print(test_acc)
            with open(single_csv_path,'r') as f:
                reader = csv.reader(f)
                val_acc = [row[2] for row in reader]
                print(val_acc)
            if len(test_acc) == 6:
                for i in range (1,6):
                    test_accs.append(float(test_acc[i]))
                    val_accs.append(float(val_acc[i]))
                    # print(test_accs)
                    # exit()
                # print(test_accs)
                # print(val_accs)
                test_acc_mean = np.mean(test_accs)
                test_acc_std = np.std(test_accs)
                val_acc_mean = np.mean(val_accs)
                val_acc_std = np.std(val_accs)
                with open(args.csv_path,'a+') as f:
                    csv_write = csv.writer(f)
                    data_row = [test_acc_mean, test_acc_std, val_acc_mean, val_acc_std, \
                                args.contrast_model, args.hidden, args.epochs, args.patience, args.activate, args.lr, args.weight_decay, args.reg, args.ratio, \
                                args.pre_norm, args.final_norm, args.normalization, args.mu, args.norm_penalty, args.penalty_ratio, args.dgi, args.dgi_contrast, args.sig_contrast, args.norm_contrast, args.mu_contrast,\
                                args.num_layers_to, args.noise_struct, args.modify_ratio, model_folder]
                    csv_write.writerow(data_row)


