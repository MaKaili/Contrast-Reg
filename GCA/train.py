import os
import numpy
import numpy as np
import argparse
import os.path as osp
import random
import nni

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from contrast_reg import Contrast_Reg
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
import csv
from pGRACE.model import Encoder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1, _, _ = model(x_1.to(device), edge_index_1.to(device), return_neg=False)
    z2, _, _ = model(x_2.to(device), edge_index_2.to(device), return_neg=False)

    loss_contrast = model.loss(z1, z2, batch_size=None)
    # loss_contrast = 0.
    if args.reg:
        z, z_neg, reg_vec = model(data.x.to(device), data.edge_index.to(device))
        loss_reg = model.loss_reg(z, z_neg, reg_vec)
        loss = loss_contrast + args.reg_ratio * loss_reg
    else:
        loss = loss_contrast
    loss.backward()
    optimizer.step()
    return loss.item()


def test(final=False):
    model.eval()
    z, _, _ = model(data.x.to(device), data.edge_index.to(device))

    if args.final_norm:
        z = F.normalize(z, p=2, dim=-1)
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--reg-ratio', type=float, default=1.)
    parser.add_argument('--reg_scale', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--save_csv', action='store_true')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--final_norm', action='store_true')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 1,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    # torch_seed = args.seed
    # torch.manual_seed(torch_seed)
    # random.seed(12345)

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    # data = data.to(device)

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)


    csv_path = f"{args.result_path}/reg_{args.reg}.csv"
    if args.save_csv:
        with open(csv_path, 'w') as f:
            csv_write = csv.writer(f)
            headers = ["seed", "nni_acc", "last_acc", "largest_acc"]
            csv_write.writerow(headers)

    acc_list = []
    for seed in range(args.num_seeds):
        seed_everything(seed)
        encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation'], param['num_hidden']),
                         base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
        func = lambda z: torch.sigmoid(z.mean(dim=0))
        model = Contrast_Reg(model=GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device),
                             reg_vec=func,
                             corruption=corruption,
                             hidden=param['num_hidden'],
                             reg=args.reg).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )

        for name, param_ in model.named_parameters():
            print(f"{name}: {param_.size()}")
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None

        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        log = args.verbose.split(',')

        for epoch in range(1, param['num_epochs'] + 1):
            loss = train()
            if 'train' in log:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.5f}')

            if epoch % 100 == 0:
                acc = test()

                if 'eval' in log:
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

                acc_list.append(acc)

        nni_acc = test(final=True)

        if args.save_csv:
            with open(csv_path, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [seed, nni_acc, acc_list[-1], max(acc_list)]
                csv_write.writerow(data_row)

        if 'final' in log:
            print(f'{acc}')


