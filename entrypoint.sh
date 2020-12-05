#!/bin/bash
cd /root/vision_transformer

args=("$@")
for i in `seq 1 ${#}`
do
    case ${args[i]} in
        --model)
	   MODEL=${args[(i+1)]};;
    esac
done

echo "model: ${MODEL}"
wget https://storage.googleapis.com/vit_models/imagenet21k/${MODEL}.npz
python -m vit_jax.train "$@"
