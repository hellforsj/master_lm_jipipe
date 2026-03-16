#fine tuning protocol for react fine tuning and non react fine tuning

import unsloth
from unsloth import FastModel
from datasets import load_dataset
from transformers import Conv1D
from trl import SFTConfig, SFTTrainer
import torch
from peft import LoraConfig, get_peft_model
import json
import evaluate
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path
import json
from prompts import SYSTEM_PROMPT_MODEL


#load metrics and models used for calculating metrics
rouge = evaluate.load("rouge")
nltk.download("punkt_tab", quiet=True)
model_emb = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def get_target_modules(model):
    """
        Function that defines which layers of the architecutre should be represented by lora adapters
    """
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    return list(set(list(filter(None, layer_names))))

def fine_tuning_unsloth(model_name:str, epochs:int, dataset_file:str, template:str=None):
    """
        Fine tuning protocol for react fine tuning on the generated dataset
    """
    run_name=model_name.split("/")[1].replace("-","_")
    save_dir= "data/models/"+run_name

    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = False,
        load_in_8bit = False, 
        device_map="auto"
    )

    #set custom template, else fall back on given template by the base model
    if template:
        tokenizer.chat_template = template

    eos=tokenizer.eos_token

    #load base model
    model = FastModel.get_peft_model(
        model,
        r = 8,
        target_modules = get_target_modules(model),
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    def compute_metrics(eval_preds):
        """
            gives custom metrics that give more information of training process
            does not interfere with training process
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        mask = labels != -100

        preds_for_decode  = np.where(mask, preds,  tokenizer.pad_token_id)
        labels_for_decode = np.where(mask, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds_for_decode,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

        decoded_preds  = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        pred_emb  = model_emb.encode(decoded_preds, convert_to_tensor=True)
        label_emb = model_emb.encode(decoded_labels, convert_to_tensor=True)
        cosine_mean = util.pytorch_cos_sim(pred_emb, label_emb).diag().mean().item()

        return {
            "rougeL": float(rouge_score["rougeL"]),
            "embedding_cosine": float(cosine_mean),
        }

    def tokenize_function(examples):
        """
            Tokenizer that applies the chat template formatting and masks labels to enable training on completion only
        """
        inputs = [json.loads(x) for x in examples["input"]]
        outputs = [json.loads(x) for x in examples["output"]]

        full_texts=[]
        prompt_texts=[]
        # Full prompt + completion text
        for i,o in zip(inputs, outputs):
            # Prompt-only text
            prompt_texts.append(
                tokenizer.apply_chat_template(
                    i,
                    tokenize=False,
                    add_generation_prompt=True #adds the instruction to the model to generate an output
                )
            )
            # full text with eos token at the end -> stopping signal to be trained by the model
            full_texts.append(
                tokenizer.apply_chat_template(
                    i+[o],
                    tokenize=False,
                    add_generation_prompt=False
                )+eos
            )

        # Tokenize both
        full_encodings = tokenizer(
            full_texts,
            truncation=True,
            padding=False,
        )

        prompt_encodings = tokenizer(
            prompt_texts,
            truncation=True,
            padding=False,
        )

        labels = []

        for full_ids, prompt_ids in zip(
            full_encodings["input_ids"],
            prompt_encodings["input_ids"],
        ):
            prompt_len = len(prompt_ids)

            # Mask prompt tokens with -100
            label = [-100] * prompt_len + full_ids[prompt_len:]
            labels.append(label)

        full_encodings["labels"] = labels
        return full_encodings

    #loading and tokenizing fine tuning dataset
    dataset=load_dataset('csv', data_files=dataset_file, delimiter=";", column_names=["input","output"], skiprows=1, split="train")
    split=dataset.train_test_split(test_size=0.2)
    eval_dataset=split["test"]
    train_dataset=split["train"]
    eval_dataset=eval_dataset.map(tokenize_function, batched=True)
    train_dataset=train_dataset.map(tokenize_function, batched= True)

    #define trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, # Can set up evaluation!
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = epochs, # Set this for 1 full training run.
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            eval_strategy="steps",
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
        ),
    )


    #training
    trainer_stats = trainer.train()
    
    #saving model adapters
    model.save_pretrained(save_dir+"_adapter", tokenizer)

    #clean cache
    del model, trainer
    torch.cuda.empty_cache()
    
    return trainer_stats

def fine_tuning_unsloth_no_react(model_name:str, epochs:int, dataset_file:str, template:str=None):
    run_name=model_name.split("/")[1].replace("-","_")+"_no_react"
    save_dir= "data/models/"+run_name

    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = False,
        load_in_8bit = False, 
        device_map="auto"
    )

    if template:
        tokenizer.chat_template = template

    model = FastModel.get_peft_model(
        model,
        r = 8,
        target_modules = get_target_modules(model),
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    def compute_metrics(eval_preds):
        """
            gives custom metrics that give more information of training process
            does not interfere with training process
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        mask = labels != -100

        preds_for_decode  = np.where(mask, preds,  tokenizer.pad_token_id)
        labels_for_decode = np.where(mask, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds_for_decode,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

        decoded_preds  = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        pred_emb  = model_emb.encode(decoded_preds, convert_to_tensor=True)
        label_emb = model_emb.encode(decoded_labels, convert_to_tensor=True)
        cosine_mean = util.pytorch_cos_sim(pred_emb, label_emb).diag().mean().item()

        return {
            "rougeL": float(rouge_score["rougeL"]),
            "embedding_cosine": float(cosine_mean),
        }

    def tokenize_function(examples):
        """
            Applies chat template formatting to the dataset
        """
        inputs = [x for x in examples["question"]]
        outputs = [x for x in examples["answer"]]
        pipelines = [x for x in examples["pipeline"]]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role":"system", "content": SYSTEM_PROMPT_MODEL},
                    {"role":"user", "content": i},
                    {"role":"assistant", "content": o+"\nJIPipe pipeline: \n"+json.dumps(p, indent=3)+""}
                ],
                tokenize = False,
            ) 
            for i, o, p in zip(inputs, outputs, pipelines)
        ]
        return tokenizer(texts, truncation = True)

    #load and format dataset
    dataset=load_dataset('csv', data_files=dataset_file, delimiter=";", column_names=["question", "reasoning", "answer", "pipeline"], skiprows=1, split="train")
    split=dataset.train_test_split(test_size=0.2)
    eval_dataset=split["test"]
    train_dataset=split["train"]
    eval_dataset=eval_dataset.map(tokenize_function, batched=True)
    train_dataset=train_dataset.map(tokenize_function, batched= True)

    #define trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, # Can set up evaluation!
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = epochs, # Set this for 1 full training run.
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            eval_strategy="steps",
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
        ),
    )


    #training
    trainer_stats = trainer.train()

    #saving model adapters
    model.save_pretrained(save_dir+"_adapter", tokenizer)

    #clear cache
    del model, trainer
    torch.cuda.empty_cache()

    return trainer_stats


def finetune_model_class(model_names, dataset, template):
    """

    """
    for model_name in model_names:
        stats=fine_tuning_unsloth(model_name, 1, dataset, template)
        print(stats)


#============START=============
#dataset paths
qwen_style_dataset="data/fine_tuning/fine_tuning_dataset_qwen.csv" #dataset in input/output text format, contains system prompt as expected by qwen and formatting in the qwen chat template
bitagent_style_dataset="data/fine_tuning/fine_tuning_dataset_bitagent.csv" #same as above but for bitagent model
dataset_path="data/fine_tuning/whole_ds.csv" #generated dataset in message format

#qwen model ft
qwen_models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]
qwen_ft_template=Path("data/chat_templates/qwen_template_ft.jinja").read_text()
for q_model in qwen_models:
    stats_q_react=fine_tuning_unsloth(q_model, 1, qwen_style_dataset)
    print(stats_q_react)
    stats_q_no_react=fine_tuning_unsloth_no_react(q_model, 1, dataset_path)
    print(stats_q_no_react)

#bitagent model ft
bitagent_models=["BitAgent/BitAgent-Bounty-8B"]
bitagent_ft_template=Path("data/chat_templates/bitagent_template_ft.jinja").read_text()
for b_model in bitagent_models:
    stats_b_react=fine_tuning_unsloth(b_model, 1, bitagent_style_dataset, bitagent_ft_template)
    print(stats_b_react)

#nanbeige model ft
nanbeige_models=["Nanbeige/Nanbeige4-3B-Thinking-2511"]
nanbeige_ft_template=Path("data/chat_templates/nanbeige_template_ft.jinja").read_text()
for n_model in nanbeige_models:
    stats_n_react=fine_tuning_unsloth(n_model, 1, qwen_style_dataset, nanbeige_ft_template)
    print(stats_n_react)
    stats_n_no_react=fine_tuning_unsloth_no_react(n_model, 1, dataset_path)
    print(stats_q_no_react)
