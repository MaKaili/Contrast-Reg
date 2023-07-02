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
for lr in 1e-3 5e-4; do
    for epoch in 500 800 1500; do
        echo $epoch
        for mu_contrast in 0.05 0.1 0.2; do
            echo $mu_contrast
            for ratio in 0 0.5 2 4; do
                echo $ratio 
                if [ $model == "CO" ]; then
                echo $dataset
                    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --pre-norm --final-norm --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu  --reg --ratio $ratio --norm-contrast     --mu-contrast $mu_contrast --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
                else
                    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python main.py --num-layers-to 1 --pre-norm --final-norm --contrast-model $model --dataset $dataset --num-seeds 5 --lr $lr --activate prelu  --reg --ratio $ratio --norm-contrast     --mu-contrast $mu_contrast --epochs $epoch --save-path "$SAVE_PATH" --save-csv --save-model
                fi
            done
        done
    done
done