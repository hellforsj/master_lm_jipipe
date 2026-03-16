# Calculates the evaluation on the trajectories produced by the compared models given the evaluation datasets

from data_classes import *
from inference import *
import json
import re
import numpy 
import os

node_ds=list(json.load(open("/data/JIPipe/JIPipe_nodes/id2name.json")).keys())
nodes_datatypes=json.load(open("/data/JIPipe/JIPipe_nodes/nodes_data_types.json"))
data_types_conversion=json.load(open("/data/JIPipe/JIPipe_nodes/conversions_data_types.json"))

def load_json_eval(eval):
    """
        Reads in a trajectory in json formatting and returns a class formatting.

    """
    def message_from_json(message):
        """
            Returns Message class object given a json representation of the message
        """
        content=message["content"]
        thinking=message["thinking"]
        tool_calls=message["tool_calls"]
        return Message(content=content, thinking=thinking, tool_calls=tool_calls)
    
    def tool_response_from_json(tool_response):
        """
            Returns ToolResponse class object given a json representation of the tool response
        """
        try:
            request_id=tool_response["request_id"]
            tool_name=tool_response["tool_name"]
            successful=tool_response["successful"]
            response=tool_response["response"]
            latency=tool_response["latency"]
            return ToolResponse(request_id=request_id, tool_name=tool_name, successful=successful, response=response, latency=latency)
        except Exception as e:
            return None
    
    def turn_from_json(turn):
        """
            Returns Turn class object given a json representation of the turn
        """
        timestamp=turn["timestamp"]
        try:
            model_message=message_from_json(turn["model_message"])
        except:
            model_message=None
        try:
            tool_response=[tool_response_from_json(t) for t in turn["tool_response"]]
        except Exception as e:
            tool_response=None
        return Turn(timestamp=timestamp, model_message=model_message, tool_response=tool_response)
    
    def session_from_json(session_json, id):
        """
            Returns Session class object given a json representation of the session
        """
        turns=[]
        prompt=session_json["prompt"]
        final_output=session_json["final_output"]
        expected=session_json["expected"]
        for t in session_json["turns"]:
            turns.append(turn_from_json(t))
        return Session(session_id=id, prompt=prompt, functions=None, turns=turns, final_output=final_output, expected=expected)
    
    sessions=[]
    for s_id in eval["sessions"]:
        sessions.append(session_from_json(eval["sessions"][s_id], s_id))
    return sessions

def get_pipeline(text):
    """
        Loads the pipeline from a text, if present. Tests for formatting and returns dict object.
    """
    try:
        pipeline=json.loads(re.search(r"\{[\s\S]*\"nodes\"[\s\S]*\"edges\"[\s\S]*\}",text).group(0))
        for n_id in pipeline["nodes"]:
            pipeline["nodes"][n_id]["name"]
        for e in pipeline["edges"]:
            e["target-node"]
            e["source-node"]
        return pipeline
    except Exception as e:
        return None


# Functions determining statistic values over the trajectories of a model
#------------------------------------------------------------------------------------------------------------------------------
def average_tool_calls(sessions:list[Session]):
    """
        Calculates the average tool calls, that are taken through all trajectories of a model. 
        Also gives other statistical measures such as the median, standard deviation, q1, q3 quantile, and the list of all calls per trajectory
    """
    def tool_calls(session):
        calls=0
        for t in session.turns:
            if t.model_message and t.model_message.tool_calls:
                calls+=len(t.model_message.tool_calls)
        return calls
    calls=[]
    for s in sessions:
        calls.append(tool_calls(s))
    mean=numpy.mean(calls)
    median=numpy.median(calls)
    std=numpy.std(calls)
    q1=numpy.percentile(calls, 25)
    q3=numpy.percentile(calls, 25)
    return mean, median, std, q1, q3, q3-q1, calls

def connection_possible(edge, pipeline):
    """
        Checks if the given edge correlates to a valid connection within the given pipeline 
    """
    def check_connection(node_a:str, node_b:str) -> bool:
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
    return check_connection(pipeline["nodes"][edge["source-node"]]["name"], pipeline["nodes"][edge["target-node"]]["name"])

def valid_pipeline(pipeline):
    """
        Checks for the validity of the pipeline given in json formatting
    """
    for n in pipeline["nodes"]:
        if pipeline["nodes"][n]["name"] not in node_ds:
            return False, "node"
    for e in pipeline["edges"]:
        if not connection_possible(e, pipeline):
            return False, "connection"
    return True, None

def tool_call_success_rate(sessions: list[Session]):
    """
        Share of tool calls, that are sucessfully called by the model
    """
    def tool_call_success_rate_for_session(session: Session):
        successful=0
        all_calls=0
        for t in session.turns:
            if t.tool_response:
                for response in t.tool_response:
                    all_calls+=1
                    if response.successful==True:
                        successful+=1
        if all_calls!=0:
            return successful/all_calls
        else:
            return None
        
    rates=[]
    for s in sessions:
        r=tool_call_success_rate_for_session(s)
        if r:
            rates.append(r)
    return numpy.average(rates)

def pipeline_presence_rate(sessions:list[Session]):
    """
        Rate at which on average pipelines are given in the output of the model
    """
    present=0
    for s in sessions:
        if re.search("JIPipe pipeline:",s.final_output) or re.search(r"\{[\s\S]*\"nodes\"[\s\S]*\"edges\"[\s\S]*?\}", s.final_output):
            present+=1
    return present/len(sessions)

def pipeline_schema_validity_rate(sessions:list[Session]):
    """
        Rate at which the given pipeline by the model is of correct format. If the given pipeline is incorrect, the reasoning is stated.
    """
    present=0
    valid=0
    reason=[]
    for s in sessions:
        if re.search("JIPipe pipeline:",s.final_output):
            try:
                pipeline=json.loads(re.search(r"\{[\s\S]*\"nodes\"[\s\S]*\"edges\"[\s\S]*\}", s.final_output.split("JIPipe pipeline:")[-1]).group(0))
                if valid_pipeline(pipeline)[0]:
                    valid+=1
                else:
                    reason.append(valid_pipeline(pipeline)[1])
            except:
                reason.append("other")
                pass
            present+=1
        elif re.search(r"\{[\s\S]*\"nodes\"[\s\S]*\"edges\"[\s\S]*\}", s.final_output):
            try:
                pipeline=json.loads(re.search(r"\{[\s\S]*\"nodes\"[\s\S]*\"edges\"[\s\S]*\}", s.final_output).group(0))
                if valid_pipeline(pipeline)[0]:
                    valid+=1
                else:
                    reason.append(valid_pipeline(pipeline)[1])
            except:
                reason.append("other")
                pass
            present+=1
    try:
        return valid/present, reason.count("node"), reason.count("connection"), reason.count("other")
    except:
        return 0, reason.count("node"), reason.count("connection"), reason.count("other")
    
def share_of_nodes_from_expected(sessions:list[Session]):
    """
        Counts the nodes which on average are present in the given pipeline by the model compared to the expected output of the supervised dataset
    """
    shared_nodes=[]
    for s in sessions:
        generated_pipeline=get_pipeline(s.final_output)
        if generated_pipeline:
            expected_pipeline=json.loads(s.expected)
            expected_nodes=[expected_pipeline["nodes"][id]["name"] for id in expected_pipeline["nodes"]]
            generated_nodes=[generated_pipeline["nodes"][id]["name"] for id in generated_pipeline["nodes"]]
            shared_nodes.append(len(set(expected_nodes) & set(generated_nodes))/len(expected_nodes))
    return numpy.mean(shared_nodes), shared_nodes

def average_reasoning_length(sessions:list[Session]):
    """
        Measures the average reasoning length per turn and for the total trajectory
    """
    def reason_length_turns(session):
        length=[]
        for t in session.turns:
            if t.model_message and t.model_message.thinking:
                length.append(len(t.model_message.thinking.split(" ")))
        return length
    
    reason_length_total=[]
    reason_length_average=[]
    for s in sessions:
        length_turns=reason_length_turns(s)
        if len(length_turns)!=0:
            reason_length_total.append(sum(length_turns))
            reason_length_average.append(numpy.mean(length_turns))
            
    mean_total_length=numpy.mean(reason_length_total)
    median_total_length=numpy.median(reason_length_total)
    mean_average=numpy.mean(reason_length_average)
    median_average=numpy.median(reason_length_average)
    return mean_total_length, median_total_length, reason_length_total, mean_average, median_average, reason_length_average

# main
#------------------------------------------------------------------------------------------------------------------------------
def stats():
    metrics={}
    base_dir="/data/evaluation/results_json/"
    for dataset in os.listdir(base_dir):
        for model_version in os.listdir(os.path.join(base_dir, dataset)):
            for model_name in os.listdir(os.path.join(base_dir, dataset, model_version)):
                print(model_name)
                print(model_version)
                model="_".join([model_name.replace(".json",""), model_version, dataset])
                model_name.replace(".json","")
                eval=json.load(open(os.path.join(base_dir, dataset, model_version, model_name)))
                sessions=load_json_eval(eval)

                atc=average_tool_calls(sessions)
                tcsr=tool_call_success_rate(sessions)
                ppr=pipeline_presence_rate(sessions)
                psvr=pipeline_schema_validity_rate(sessions)
                arl=average_reasoning_length(sessions)
                metrics[model]={"average_tool_calls": {
                        "average":atc[0], 
                        "median": atc[1], 
                        "std": atc[2], 
                        "q1": atc[3],
                        "q3": atc[4],
                        "iqr": atc[5],
                        "calls":atc[6]
                        },
                    "tool_call_success_rate": tcsr,
                    "pipeline_presence_rate": ppr,
                    "pipeline_schema_validity_rate": {"value": psvr[0], "wrong_nodes": psvr[1], "wrong_connections": psvr[2], "other": psvr[3]},
                    "average_reasoning_length": {
                        "total_length":{
                            "mean":arl[0],
                            "median": arl[1],
                            "lengths":arl[2]
                        },
                        "turn_length":{
                            "mean":arl[3],
                            "median": arl[4],
                            "lengths":arl[5]
                        }
                    }
                    }
    return metrics

def stats_supervised():
    metrics={}
    base_dir="/data/evaluation/results_json/d2/"
    for model_version in os.listdir(base_dir):
        for model_name in os.listdir(os.path.join(base_dir, model_version)):
            model="_".join([model_name.replace(".json",""), model_version])
            model_name.replace(".json","")
            eval=json.load(open(os.path.join(base_dir, model_version, model_name)))
            sessions=load_json_eval(eval)
            sne=share_of_nodes_from_expected(sessions)
            metrics[model]={
                "mean": sne[0],
                "nodes": sne[1]
            }
    return metrics

# run
#=====================================================================================================
stats=stats()
file=open("/data/evaluation/metrics_results.json", "w")
file.write(json.dumps(stats, indent=3))
file.close()

stats_supervised=stats_supervised()
file=open("/data/evaluation/metrics_supervised.json", "w")
file.write(json.dumps(stats_supervised, indent=3))
file.close()