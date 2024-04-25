#!/bin/bash

conda activate sos 

cd src

accelerate launch --config_file ../configs/accelerate.yaml train.py --config ../configs/sft-mix-4-cd5e5.conf
