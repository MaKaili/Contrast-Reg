# Code for Calibrating and Improving Graph Contrastive Learning

## Environment

The running environment of the repo is listed in requirements.txt.

## Reproducing results
### GCA
Please first enter the GCA folder,
```
cd GCA
```
Then run the following command
```python
python train.py --dataset WikiCS --param local:wikics.json --device cuda:0 --reg --reg-ratio 0.0001 --num_seeds 10
python train.py --dataset Amazon-Photo --param local:amazon_photo.json --device cuda:0 --reg --reg-ratio 0.0001 --num_seeds 10
python train.py --dataset Amazon-Computers --param local:amazon_computers.json --device cuda:0 --reg --reg-ratio 0.01 --num_seeds 10
python train.py --dataset Coauthor-CS --param local:coauthor_cs.json --device cuda:0 --reg --reg-ratio 0.1 --num_seeds 10
python train.py --dataset Cora --param local:cora.json --device cuda:0 --reg --reg-ratio 8.0 --num_seeds 10
python train.py --dataset CiteSeer --param local:citeseer.json --device cuda:0 --reg --reg-ratio 8.0 --num_seeds 10
python train.py --dataset PubMed --param local:pubmed.json --device cuda:0 --reg --reg-ratio 8.0 --num_seeds 10
```

### ML/LC

To replicate the ML/LC results presented in our paper, we apply the GCN encoder on Cora, Citeseer, Pubmed, Wiki, Computers, and Photo, setting the number of layer to 1 and implementing full-batch training. The Contrast-Reg ratio is set to 1.

For Cora (LC), we set a learning rate of 3e-3 and enable curriculum learning. To reproduce the experiments, execute the following command:

```
python main.py --dataset cora --contrast-model LC --lr 3e-3 --use-curri --curri-round 5 --reg --ratio 1 --bilinear --num-layers-to 1
```

For Cora (ML), we set the learning rate to 3e-3. To reproduce the experiments, execute the following command:
```
python main.py --dataset cora --contrast-model ML --lr 3e-3 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000
```

For Citeseer, Pubmed, Computers, Photo, Wiki dataset, run the following commands:

```
python main.py --dataset citeseer --contrast-model LC --lr 1e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 500 --use-curri --curri-round 3 --pre-norm --final-norm
python main.py --dataset citeseer --contrast-model ML --lr 5e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 300 --pre-norm --final-norm
python main.py --dataset pubmed --contrast-model LC --lr 1e-2 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 300 --final-norm
python main.py --dataset pubmed --contrast-model ML --lr 1e-2 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000 --final-norm
python main.py --dataset computers --contrast-model LC --lr 3e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000
python main.py --dataset computers --contrast-model ML --lr 1e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000 
python main.py --dataset photo --contrast-model LC --lr 3e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000
python main.py --dataset photo --contrast-model ML --lr 1e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000 
python main.py --dataset wiki --contrast-model LC --lr 1e-3 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000
python main.py --dataset wiki --contrast-model ML --lr 1e-4 --reg --ratio 1 --bilinear --num-layers-to 1 --epochs 1000 
```

For the link prediction task, the necessary codes are readily available in the 'link-prediction' folder. To replicate the LC outcomes for the Cora, Citeseer, Pubmed, and Wiki datasets, execute the commands outlined below:

```
python link_exete.py --dataset cora --reg --lr 0.003 --epochs 300 --use_curri --curri_round 5 --lr_deduct 0.5 --same_layer
python link_exete.py --dataset citeseer --reg --lr 0.0001 --epochs 500 --use_curri --curri_round 3 --pre_norm --final_norm --lr_deduct 0.5 --same_layer
python link_exete.py --dataset pubmed --reg --lr 0.01 --epochs 300 --final_norm --same_layer
python link_exete.py --dataset wiki --reg --lr 0.001 --epochs 1000 --same_layer
```

The hyperparameters for link prediction tasks adhere rigorously to the settings derived from node classification, with no additional fine-tuning applied.
