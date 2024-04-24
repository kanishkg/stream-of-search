#!/bin/bash

conda activate sos

cd src

data_file="b4_3_random/train1_b4_t100_n500000_random.json"
python eval_neo.py --ckpt "checkpoint-45500" -n 220000 -o 215000 -d "$data_file" --temperature 0.8 --batch_size 32
