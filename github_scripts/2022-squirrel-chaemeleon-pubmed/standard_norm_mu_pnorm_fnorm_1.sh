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
SAVE_PATH="2022-jan/linear/standard-split/pnorm+fnorm+reg"
for lr in 3e-3 1e-3 5e-4 1e-4; do
    for epoch in 300 500 800; do
        echo $epoch
        for mu_contrast in 0.01 0.05 0.1 0.3 0.5 0.7; do
            echo $mu_contrast
            for ratio in 0 0.1 0.5 1 4 8; do
                OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 10 --lr $lr --activate prelu  --norm-contrast     --reg --ratio $ratio --pre-norm --final-norm --mu-contrast $mu_contrast --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
            done
        # OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 10 --lr $lr --activate prelu  --dgi --dgi-contrast  --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        # OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --contrast-model $model --dataset $dataset --num-seeds 10 --lr $lr --activate prelu  --sig-contrast        --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
        done
    done
done