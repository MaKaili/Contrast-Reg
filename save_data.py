import torch
import torch.nn as nn
import numpy as np
from utils import mkdir_
import os

def _cal_norm(z):
    norm_z = torch.norm(z, dim=1).mean().detach().to('cpu').numpy()
    return norm_z

def cal_norm(model, data):
    model.eval()
    z = model.inference(data)
    norm_z = _cal_norm(z)
    return norm_z

class Record(object):
    def __init__(self, args, num_nodes):
        self.args = args
        self.cluster = np.load("cluster/cluster_"+args.dataset+".npy")
        self.num_pos = np.sum(self.cluster)
        self.num_neg = num_nodes*num_nodes-self.num_pos
        self.norm_z_mean = []
        self.norm_z_variance = []
        self.norm_pos = []
        self.norm_neg = []
        self.norm_ratio = []
        self.loss = []
        self.test_acc = []
        self.val_acc = []
        self.logits = []
        self.reset()

        self.norm_path = os.path.join("norm", args.save_path, args.path_name)

    def reset(self):
        self.norm_z_mean = []
        self.norm_z_variance = []
        self.norm_pos = []
        self.norm_neg = []
        self.norm_ratio = []
        self.norm_ratio = []
        self.loss = []
        self.test_acc = []

    def update_norm(self, z):
        p_mat = torch.matmul(z, torch.transpose(z, 0, 1)).to('cpu').detach().numpy()
        inner_p = p_mat*self.cluster
        inter_p = p_mat*(1-self.cluster)
        p_pos=np.linalg.norm(inner_p, 'fro')**2/self.num_pos
        p_neg=np.linalg.norm(inter_p, 'fro')**2/self.num_neg

        self.norm_z_mean.append(_cal_norm(z))
        self.norm_variance = (torch.norm(z, dim=1).std())**2
        self.norm_z_variance.append(self.norm_variance.detach().to('cpu').numpy())
        self.norm_pos.append(p_pos)
        self.norm_neg.append(p_neg)

    def update_train(self, loss, test_acc, val_acc):
        self.loss.append(loss)
        self.test_acc.append(test_acc)
        self.val_acc.append(val_acc)

    def update_logits(self, z, weight, reg_vec):
        r = torch.matmul(weight, reg_vec.detach())
        r = r.view(1, z.size(1))
        logits = (z*r).sum(-1).sum()
        self.logits.append(logits)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        ax.plot(np.arange(1, args.epochs+1), np.array(self.norm_pos[:-1]), label='Inner class')
        ax.plot(np.arange(1, args.epochs+1), np.array(self.norm_neg[:-1]), label='Inter class')
        plt.legend()
        ax = fig.add_subplot(2,2,2)
        ax.plot(np.arange(1, args.epochs+1), np.array(self.norm_ratio[:-1]), label='norm_ratio')
        plt.legend()
        ax = fig.add_subplot(2,2,3)
        ax.plot(np.arange(1, args.epochs+1), np.array(self.loss), label='loss')
        ax = fig.add_subplot(2,2,4)
        ax.plot(np.arange(1, args.epochs+1), np.array(self.test_acc), label="test_acc")
        ax.plot(np.arange(1, args.epochs+1), np.array(self.val_acc), label="val_acc")
        fig_name = os.path.join("fig", args.save_path, args.dataset.lower()+"contrast-model_"+args.contrast_model+"_hidden_"+str(args.hidden)
                                 +'_num_splits_'+str(args.num_splits)+"_num_seeds_"+str(args.num_seeds)+'_epochs_'+str(args.epochs)#general
                                 +'_lr_'+str(args.lr)+"_weight_decay_"+str(args.weight_decay)+"_patience_"+str(args.patience)#optimization
                                 +'_use_curri_'+str(args.use_curri)+'_curri_round_'+str(args.curri_round)+'_lr_deduct_'+str(args.lr_deduct)#LC model
                                 +'_pre_norm_'+str(args.pre_norm)+'_final_norm_'+str(args.final_norm)+"_normalization_"+str(args.normalization)
                                 +'_ratio_'+str(args.ratio)+".pdf")
        plt.savefig(fig_name)



    def save(self):
        mkdir_(os.path.join("norm", self.args.save_path))
        np.save(self.norm_path+"_norm_pos.npy", np.array(self.norm_pos[:-1]))
        np.save(self.norm_path+"_norm_neg.npy", np.array(self.norm_neg[:-1]))
        np.save(self.norm_path+"_norm_ratio.npy", np.array(self.norm_ratio[:-1]))
        np.save(self.norm_path+"_loss_list.npy", np.array(self.loss))
        np.save(self.norm_path+"_test_acc.npy", np.array(self.test_acc))
        np.save(self.norm_path+"_val_acc.npy", np.array(self.val_acc))
        np.save(self.norm_path+"_norm_z_mean.npy", np.array(self.norm_z_mean))
        np.save(self.norm_path+"_norm_z_variance.npy", np.array(self.norm_z_variance))
        np.save(self.norm_path+"_logits.npy", np.array(self.logits))



