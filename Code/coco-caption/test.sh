#!/bin/bash
echo "test"
path=$(pwd)
echo ${path}
cd ..
for file in $(ls ${path}); do
    echo ${file}
    echo ${path}/${file}
    #../bin/traffic_compressor /data/zyh/janus-v5/klotski_experiments/${file}/ /data/zyh/janus-v5/klotski_experiments/${file}/traffic
    python predictions_runner.py  --checkpoint U_0_sqrt_0.1/${file}  --dataset_mode 0 --cuda 0
done