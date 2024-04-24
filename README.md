# Stream of Search
Repository for the paper Stream of Search: Learning to Search in Language(https://arxiv.org/abs/2404.03683)

See APA code here: https://github.com/kanishkg/RLHF-APA

The repository is organized as follows:
- `analysis/`: Contains files for analyzing the results of the experiments and to generate plots
- `configs/`: Has the configuration files for training the language models
  - `gpt-neo-s.json`: Configuration file for the transformer model
  - `oft-mix-4-cd.json`: Configuration file for training the OT (Optimal Solution) model
  - `sft-mix-4-cd.json`: Configuration file for training the SoS (Stream of Search) model
  - `star1-mix-4-cd.json`: Configuration file for training the Star iteration 1 model
  - `star2-mix-4-cd.json`: Configuration file for training the Star iteration 2 model
  - `star3-mix-4-cd.json`: Configuration file for training the Star iteration 3 model
- `scripts/`: Contains the scripts for training the models
  - `gen_task.sh`: Script for generating the initial countdown dataset
  - `train.sh`: Script for training the model in the OT or SoS setting
  - `gen_star.sh`: Script for generating the Star dataset for the next improvement iteration
  - `star.sh`: Script for training the model in the Star setting
- `src/`: Contains the source code for training the models, generating the dataset, and evaluating the model
  - `data.py`: Contains the code for generating the countdown dataset
  - `model.py`: Contains the code for the model
  - `train.py`: Contains the code for training the  model
  - `countdown.py`: Generating the countdown problems
  - `countdown_bfs.py`: Generate search streams with BFS
  - `countdown_dfs.py`: Generate search streams with DFS
  - `countdown_utils.py`: Utility functions for countdown, for heuristics, analysis, metrics etc.
  - `countdown_generate.py`: Generate the countdown dataset
  - `countdown_optimal.py`: Add optimal paths to the countdown dataset
  - `train.py`: Script for training the model in the OT or SoS or STaR setting
  - `eval_neo.py`: Script for evaluating the model

