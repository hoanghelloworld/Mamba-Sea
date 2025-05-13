#!/bin/bash

file=$1
dataset=$2
model=$3
device=$4
ratio=$5
config=$6
extra=${@:7}


# check n args
if [ $# -lt 6 ]; then
    echo "Usage: $0 <file> <dataset> <model> <device> <ratio> <config_name> [extra]"
    exit
fi

method=$(basename $file .py)

seed=1339
cfg="configs/dataset/$dataset.yaml"  # dataset config
tcfg="configs/$config.yaml"  # training config

if [[ $seed == 1339 ]]; then
    dir="outputs/$dataset/$ratio/$model/$method-$config"
else
    dir="outputs-$seed/$dataset/$ratio/$model/$method-$config"
fi

echo $dir

case $dataset in
    "prostate")
        domain_ids=(3 4 5)
        domain_names=(D E F)
        ;;
    "prostate2")
        domain_ids=(3 4 5)
        domain_names=(D E F)
        ;;
    "fundus")
        domain_ids=(2 3)
        domain_names=(C D)
        ;;
    "mnms")
        domain_ids=(2 3)
        domain_names=(C D)
        ;;
    "scgm")
        domain_ids=(2 3)
        domain_names=(C D)
        ;;
    *)
        echo "Dataset not supported!"
        exit
        ;;
esac

mkdir -p $dir

for i in "${!domain_ids[@]}"; do
    domain_id=${domain_ids[$i]}
    domain_name=${domain_names[$i]}
    CUDA_VISIBLE_DEVICES=$device python $file \
        --config $cfg \
        --train-config $tcfg \
        --save-path $dir/$domain_name \
        --model $model \
        --seed $seed \
        --domain $domain_id \
        --ratio $ratio \
        $extra \
        >> $dir/$domain_name.log 2>&1 &
    # run one by one
    wait $!
done
