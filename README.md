# Master Thesis - Large Language Model-based Distillation of JIPipe Workflows into Compact Language Models

The repository contains the skripts and data that were used throughout the thesis.

Files are separated in implementation skripts (`/programs`) and data. The directory `/data` contains all data, on which basis dataset generation was carried out and all generated datasets. The directory `/envs` holds information regarding library versions that were used for the skripts.

## Folder structure
.
├───data
│   ├───chat_templates
│   ├───evaluation
│   │   ├───dataset
│   │   ├───eval_pipelines
│   │   ├───results_json
│   │   │   ├───d1
│   │   │   │   ├───base_model
│   │   │   │   ├───fine_tuned
│   │   │   │   └───react_tuned
│   │   │   └───d2
│   │   │       ├───base_model
│   │   │       ├───fine_tuned
│   │   │       └───react_tuned
│   │   └───templates
│   ├───fine_tuning
│   │   ├───node_search
│   │   └───react
│   ├───JIPipe
│   │   ├───JIPipe_nodes
│   │   ├───JIPipe_projects
│   │   └───short_pipelines
│   └───models
│       ├───pipeline_building
│       │   ├───BitAgent_Bounty_8B_adapter
│       │   ├───Nanbeige4_3B_Thinking_2511_adapter
│       │   ├───Nanbeige4_3B_Thinking_2511_no_react_adapter
│       │   ├───Qwen3_0.6B_adapter
│       │   ├───Qwen3_0.6B_no_react_adapter
│       │   ├───Qwen3_8B_adapter
│       │   └───Qwen3_8B_no_react_adapter
│       └───text_classification
├───envs
└───programs
    ├───data_classes
    ├───evaluation
    │   └───inference
    ├───fine_tuning_protocol
    ├───node_search
    ├───preprocessing
    ├───prompts
    └───react_dataset