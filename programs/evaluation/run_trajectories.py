#skript for running trajectories on evaluation dataset. Trajectories run on merged models (have to be created from lora adapters)

import json
import uuid
import time
from typing import List, Dict, Any, Callable
import datetime
import pickle
import joblib
import torch
import gc
from pathlib import Path

from inference import *
from data_classes import *
from prompts import SYSTEM_PROMPT_MODEL, TOOL_DESCRIPTION, TOOL_DESCRIPTION_NATIVE_REASONING, TOOL_DESCRIPTION_NO_REASONING

#Initialize model for classification of nodes
embedding_model, classifier=joblib.load("data/models/text_classification/text_phrase_classifier_all-mpnet-base-v2_logistic_reg.joblib")
embedding_model = embedding_model.to("cuda")

#initialize label dict and node dict
label2id=json.load(open("asbdata/hellfors/eval/label2id.json"))
node_description=json.load(open("data/JIPipe/JIPipe_nodes/node_id_description.json"))
nodes_datatypes=json.load(open("data/JIPipe/JIPipe_nodes/nodes_data_types.json"))
data_types_conversion=json.load(open("data/JIPipe/JIPipe_nodes/conversions_data_types.json"))

def load_questions(path:str) -> list[str]:
    """
        Loads the question for evaluation from a json file.
        Args:
            path (str): path to the file
        Returns:
            questions (list[str]): evaluation questions
    """
    return json.load(open(path))

#Tools
#----------------------------------------------------------------------------------------------------------
def get_node_name(prompt: str) -> tuple[str, str]:
    """
    Based on a short prompt stating the desired function, the best node is found with its description of functionality.

    Args:
        prompt: search prompt for the node

    Returns:
        tuple[str, str]: A tuple containing:
            - node_name: Node name of the best match
            - node_description: Description of the node's functionality
    """
    encoding=embedding_model.encode(prompt, device="cuda")
    classification=classifier.predict(encoding.reshape(1,-1))
    node_name = label2id[str(int(classification[0]))]
    description=node_description[node_name]
    return node_name, description

def check_connection(node_a:str, node_b:str) -> bool:
    """
        Returns True, if a connection/edge from node_a to node_b is possible, otherwise returns False.
        Args:
            node_a: Name of the source node.
            node_b: Name of the target node.
        Returns:
            can_connect: Decision on whether a connection from node_a to node_b is possible.
    """
    a=nodes_datatypes[node_a]["output"]
    b=nodes_datatypes[node_b]["input"]
    for input in a:
        if input in b:
            return True
        else:
            conversion=data_types_conversion[input]["to (trivial)"]
            for c in conversion:
                if c in b:
                    return True
    return False


#Session on Single Evaluation Question
#----------------------------------------------------------------------------------------------------------
def run_session(prompt: str, model: Model, max_turns: int = 25) -> Session:
    """
    Multi-turn session of a model given a user request
    
    :param prompt: user request
    :type prompt: str
    :param model: model used for multi-turn generation
    :type model: Model
    :param max_turns: max. steps the model might take
    :type max_turns: int
    :return: Multi-turn session of the user request
    :rtype: Session
    """
    #initialize session and model
    session = Session(session_id=str(uuid.uuid4()), prompt=prompt, functions=model.available_functions)
    turns=0
    model.init_chat(prompt)

    #initial model response
    model_response=model.single_turn_generation()
    
    while turns<max_turns:
        timestamp=time.time()
        model_message=Message(
            content=model_response.content, 
            thinking=model_response.thinking, 
            tool_calls=model_response.tool_calls
            )

        #get tool calls from response
        tool_calls = model.extract_tool_call(model_response)
        
        #model reached final answer if there are no more tool calls
        if tool_calls is None:
            session.final_output = model_response.content
            return session

        #append model response
        session.turns.append(Turn(timestamp=timestamp, model_message=model_message))
        
        #append tool responses
        tool_responses=model.tool_router(tool_calls=tool_calls)
        session.turns.append(Turn(timestamp=time.time(), tool_response=tool_responses)) 

        #update tool response and generate new model output
        model.update_tool_response(tool_responses)
        model_response= model.single_turn_generation()
        turns+=1

    #max turns reached
    session.final_output = "(max turns reached)"
    return session

#Sessions on All Questions 
#----------------------------------------------------------------------------------------------------------
def complete_sessions(questions: List[str], model: Model, log) -> Dict[str, Any]:
    sessions = []
    log.write(f"Run trajectories of {len(questions)} questions\n")
    i=0
    print(len(questions))
    for q in questions:
        print(i)
        try:
            log.write(f"Question {str(i)}\n")
            log.flush()
            #run each questions as sessions to get trajectory
            s = run_session(q, model)
            sessions.append(s)
        except Exception as e:
            print(f"Failed Question: {q}")
            print(e)
        i+=1

    return {
        "sessions": sessions
    }

#Saving etc.
#----------------------------------------------------------------------------------------------------------
def save_session(results, path):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()

def run(model: Model, run_id: str, log):
    #load questions for eval
    questions=load_questions("data/evaluation/dataset/d1.json")

    #automated evaluation of model
    results = complete_sessions(questions, model, log)
    log.write(f"Evaluations: {len(results)}")
    log.flush()

    save_session(results, f"data/evaluation/results/results_{model.model_path.split("/")[-1]}_{run_id}.pckl")
    del model
    gc.collect()
    torch.cuda.empty_cache()

def run_list_of_models(models:List[str], system_prompt:str, tools:List[Callable], chat_template:str=None, tool_description:str=None, special_tokens: Dict=None):
    for name in models:
        model=Model(name, tools, tool_description=tool_description, system_prompt=system_prompt, chat_template=chat_template, special_tokens=special_tokens)
        model_name=name.split("/")[-1]
        run_id=datetime.datetime.now().strftime("%d-%m_%H_%M_%S")
        log=open(f"data/evaluation/results/{model_name.replace("/","-")}_evaluation_{run_id}.log", "w")
        run(model, run_id, log)
        log.close()

#START OF PROGRAM
#==========================================================================================================
tools=[get_node_name, check_connection]

#replace if needed with needed version of the model. at the moment, latest version is loaded from HF hub
base_models_non_reasoning=[
             "BitAgent/BitAgent-Bounty-8B", 
            ]  

#replace if needed with needed version of the model. at the moment, latest version is loaded from HF hub
base_models_reasoning=[
             "Qwen/Qwen3-0.6B", 
             "Qwen/Qwen3-8B", 
             "Nanbeige/Nanbeige4-3B-Thinking-2511"
            ]

qwen_models=["/data/models/Qwen3_0.6B_merged", "/data/models/Qwen3_8B_merged"]
qwen_tokens=Tokens(reasoning_start="<think>",
                   reasoning_end="</think>",
                   tool_response_start="<tool_response>",
                   tool_response_end="</tool_response>",
                   tool_call_start="<tool_call>",
                   tool_call_end="</tool_call>")

bitagent_models=["/data/models/BitAgent_Bounty_8B_merged"]
bitagent_template=Path("data/chat_templates/bitagent_template_inference.jinja").read_text()
bitagent_tokens=Tokens(reasoning_start="<think>",
                   reasoning_end="</think>",
                   tool_response_start="",
                   tool_response_end="",
                   tool_call_start="",
                   tool_call_end="")

nanbeige_models=["/data/models/Nanbeige4_3B_Thinking_2511_merged"]
nanbeige_tokens=Tokens(reasoning_start="<think>",
                   reasoning_end="</think>",
                   tool_response_start="",
                   tool_response_end="",
                   tool_call_start="<tool_call>",
                   tool_call_end="</tool_call>")

run_list_of_models(qwen_models, system_prompt=SYSTEM_PROMPT_MODEL, tools=tools, tool_description=TOOL_DESCRIPTION_NATIVE_REASONING, special_tokens=qwen_tokens)
run_list_of_models(nanbeige_models, system_prompt=SYSTEM_PROMPT_MODEL, tools=tools, tool_description=TOOL_DESCRIPTION_NATIVE_REASONING, special_tokens=nanbeige_tokens)
run_list_of_models(bitagent_models, system_prompt=SYSTEM_PROMPT_MODEL, tools=tools, tool_description=TOOL_DESCRIPTION_NO_REASONING, special_tokens=bitagent_tokens, chat_template=bitagent_template)
run_list_of_models(base_models_reasoning, system_prompt=SYSTEM_PROMPT_MODEL, tools=tools, tool_description=TOOL_DESCRIPTION_NATIVE_REASONING.format(tools=json.dumps(TOOL_DESCRIPTION, indent=3)), special_tokens=qwen_tokens)
run_list_of_models(base_models_non_reasoning, system_prompt=SYSTEM_PROMPT_MODEL, tools=tools, tool_description=TOOL_DESCRIPTION_NO_REASONING, special_tokens=bitagent_tokens)