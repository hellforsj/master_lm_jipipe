# Fine-tuning datasets

## File description ReAct dataset
The .csv files contain the generated fine-tuning dataset in the message formatting, that is expected through the chat templates. Message formatting is of json and defines roles and content.

- `/react/whole_ds.csv`
    - Contains 4 columns (question, reasoning, answer, pipeline). Question contains the user input in plain text, reasoning contains the complete multi-turn trajectory in message formatting, answer contains the final output message of the model, and pipeline has the final JIPipe pipeline.
- `/react/fine_tuning_dataset_qwen.csv`
    - Contains consecutive trajectory steps as input/output pairs. Derived from `/whole_ds.csv`. Input/output are given in message formatting that is expected by the qwen chat template. As the chat template formatting of qwen is similar to nanbeige, this dataset is also utilized for the nanbeige model.
- `/react/fine_tuning_dataset_bitagent.csv`
    - Similar to the just described dataset. Formatting of messages is adapted to the bitagent chat template.

## File description node search dataset
The .csv files contain the produced node search prompts.

## Folder structure
```text
.
├───node_search
└───react