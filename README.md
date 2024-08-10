# Stream of Search
Repository for the paper Stream of Search: Learning to Search in Language(https://arxiv.org/abs/2404.03683)

See APA code here: https://github.com/kanishkg/RLHF-APA

## Installation

1. Install conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Create a conda environment
```bash
conda create -n sos python=3.11
conda activate sos
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

## Running the code
Please update the scripts in the `scripts/` directory to reflect the correct paths to the data and model checkpoints. The following steps outline the process of running the code:
1. Generate the countdown dataset
```bash
sh scripts/gen_task.sh
```
2. Train the model
```bash
sh scripts/train.sh
```
3. Generating data for STaR
```bash
sh scripts/gen_star.sh
```
4. Train the model with STaR
```bash
sh scripts/star.sh
```
5. Evaluate the model
```bash
sh scripts/eval.sh
```

## Repository Structure Overview

This repository is structured to support efficient development, training, and evaluation of models. Below is an organized breakdown of each directory:

### `analysis/`
**Purpose**: Contains scripts and tools for analyzing experimental results and generating plots.

### `configs/`
**Purpose**: Houses configuration files for various training settings.
- `gpt-neo-s.json`: For the GPT-Neo transformer model.
- `oft-mix-4-cd.conf`: For the Optimal Solution (OT) model.
- `sft-mix-4-cd.conf`: For the Stream of Search (SoS) model.
- `star1-mix-4-cd.conf`: For Star iteration 1 model.
- `star2-mix-4-cd.conf`: For Star iteration 2 model.
- `star3-mix-4-cd.conf`: For Star iteration 3 model.

### `scripts/`
**Purpose**: Contains scripts for data generation and model training.
- `gen_task.sh`: Generates the initial countdown dataset.
- `train.sh`: Trains models under OT or SoS settings.
- `gen_star.sh`: Generates data for Star iterations.
- `star.sh`: Trains models in Star setting.
- `eval.sh`: Evaluates the performance of the models.

### `src/`
**Purpose**: Includes all source code for model training, data generation, and evaluation.
- `data.py`: Generates the countdown dataset.
- `model.py`: Main file for model definitions.
- `train.py`: Executes model training processes.
- `countdown.py`: Generates countdown problem scenarios.
- `countdown_bfs.py`: Utilizes BFS for generating search streams.
- `countdown_dfs.py`: Utilizes DFS for generating search streams.
- `countdown_utils.py`: Provides utility functions for countdown scenarios.
- `countdown_generate.py`: Generates countdown dataset.
- `countdown_optimal.py`: Adds optimal paths to the countdown dataset.
- `eval_neo.py`: Script for model evaluation.
