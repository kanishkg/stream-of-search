#!/bin/bash

conda activate sos

cd src

python countdown_generate.py --seed 4 --data_dir data/b4_3_random/ --min_range 4 --start_range 4 --num_samples 500000
