#!/bin/bash
echo "evalueate"
path=$(pwd)
echo ${path}
for file in $(ls ${path}/results); do
    echo ${file}
    #echo ${path}/${file}
    #../bin/traffic_compressor /data/zyh/janus-v5/klotski_experiments/${file}/ /data/zyh/janus-v5/klotski_experiments/${file}/traffic
    outfile=${file%.json}.txt
    echo ${outfile}
    python evaluation.py  --res ./results/${file}  --outpath data_res/${outfile}
done