# Program Directory
This directory contains all implementations of the processes described throughout the thesis. 

## Folder explanation
- `data_classes/`
  - Holds the class and data type definitions that are utilized in the skripts.
- `evaluation/`
  - Contains the evaluation skript that produces the trajectories of the models. A skript for the conversion of trajectories stored as .pckl to .json format is also included (`evaluation/load_pickles.py`)
  - To run fine-tuned models, LoRA adapters (`data/models/`) have to be merged with the corresponding base models. Skript for merging is found at `merge_adapters_with_base_model.py` 
  - `evaluation/inference/` includes classes which allow multi-turn tool calling inference.
- `fine_tuning_protocol`
  - Holds the fine-tuning protocols for the ReAct and non-ReAct fine-tuning of the chosen compact language models and the model fitting protocol for the SVC for the node search.
- `node_search/`
  - Includes the skript for generating the supervised dataset used for training the SVC.
- `preprocessing/`
  -  Contains the preprocessing skripts.
- `prompts/`
  - Collects all prompts that were used throughout the thesis.


## Folder structure
```text
.
├───data_classes
├───evaluation
│   └───inference
├───fine_tuning_protocol
├───node_search
├───preprocessing
├───prompts
└───react_dataset
