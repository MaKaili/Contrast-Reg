if [ $# != 3 ]; then
    echo "Usage: bash grid_search.sh <device> <model> <dataset>";
    exit
else
    device=$1
    model=$2
    dataset=$3
fi
# for dataset in "citeseer pubmed"; do
#     echo $dataset

# For Normalize and WD baseline (except Cora and Computers)
SAVE_PATH="2021-Oct/standard-split/grid_search_CO_2021_DEC"
for lr in 3e-3 1e-3 5e-4 1e-4; do
    for epoch in 300 500 800 1000 1500; do
        echo $epoch
        for wd in 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5; do
            echo $wd
            OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu --weight-decay $wd  --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        done
        for mu in 1 0.7 0.5 0.3 0.1 0.05 0.01; do
            echo $mu
            OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu --normalization --mu $mu           --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        done
        for pn in 1 0.5 0.1 0.05 0.01; do
            echo $pn
            OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu --norm-penalty --penalty-ratio $pn --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model 
        done
        for ratio in 0.1 0.3 0.5 0.7 1 2 4 8; do
            OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu --reg --ratio $ratio --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        done
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu  --norm-contrast       --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu  --dgi --dgi-contrast  --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu  --sig-contrast        --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu                        --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
    done
done