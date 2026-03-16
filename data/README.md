# Data Directory

This directory contains all data used and created in the course of this thesis.


## Folder explanation
- `evaluation/`
  - Folder containing all results of the model inferences. Trajectories are given in json format.
    `d1/` and `d2/` contain the results of the respective evaluation datasets, 
    each divided into the different training configurations (base_model, fine_tuned, react_fine_tuned).
- `fine_tuning/`
  - Datasets used for fine-tuning that were generated through the described process with programs `programs/node_search/create_dataset.py` and `programs/react_dataset/generate_dataset.py`.
    Contains training dataset for ReAct fine-tuning and SVC embedding.
- `JIPipe/`
  - `JIPipe/JIPipe_nodes`: Structured data. Derived from JIPipe development tools and ImageJ manual.
  - `JIPipe/JIPipe_projects`: base projects used as a basis for the dataset creation.
  - `JIPipe/short_pipelines`: Short pipelines that are obtained through the preprocessing process from the base projects.
- `models/`
  - `models/pipeline_building`: Contains adapters for ReAct fine-tuned models and non-ReAct fine-tuned models. 
  - `models/text_classification`: Contains the SVC models used for the node search.

## Folder structure
```text
.
├───chat_templates
├───evaluation
│   ├───dataset
│   ├───eval_pipelines
│   ├───results_json
│   │   ├───d1
│   │   │   ├───base_model
│   │   │   ├───fine_tuned
│   │   │   └───react_tuned
│   │   └───d2
│   │       ├───base_model
│   │       ├───fine_tuned
│   │       └───react_tuned
│   └───templates
├───fine_tuning
│   ├───node_search
│   └───react
├───JIPipe
│   ├───JIPipe_nodes
│   ├───JIPipe_projects
│   └───short_pipelines
│       │
│       ...
└───models
    ├───pipeline_building
    │   ├───BitAgent_Bounty_8B_adapter
    │   ├───Nanbeige4_3B_Thinking_2511_adapter
    │   ├───Nanbeige4_3B_Thinking_2511_no_react_adapter
    │   ├───Qwen3_0.6B_adapter
    │   ├───Qwen3_0.6B_no_react_adapter
    │   ├───Qwen3_8B_adapter
    │   └───Qwen3_8B_no_react_adapter
    └───text_classification



