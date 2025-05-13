#!/bin/bash

par=$1
file=$2
dataset=$3
model=$4
device=$5
ratio=$6
config=$7
extra=${@:8}


# check n args
if [ $# -lt 7 ]; then
    echo "Usage: $0 <parallel> <file> <dataset> <model> <device> <ratio> <config_name> [extra]"
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
        domain_ids=(0 1 2 3 4 5)
        domain_names=(A B C D E F)
        ;;
    "fundus")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    "mnms")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    "scgm")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    "covid")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    "skin")
        domain_ids=(0 1)
        domain_names=(A B)
        ;;
    *)
        echo "Dataset not supported!"
        exit
        ;;
esac

mkdir -p $dir

if [[ $par == "parallel" ]]; then
    echo "Running in parallel"
    devices=(${device//,/ })
    echo "Devices: ${devices[@]}"

    for i in "${!domain_ids[@]}"; do
        domain_id=${domain_ids[$i]}
        domain_name=${domain_names[$i]}
        # device=${devices[$i]}
        # circular device assignment
        k=$((i % ${#devices[@]}))
        device=${devices[$k]}
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
    done
    wait
else
    echo "Running in serial"
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
            >> $dir/$domain_name.log 2>&1
        wait $!
    done
fi
