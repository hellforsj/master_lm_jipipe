import pickle
import json
import os

def load_eval(path):
    with open(path, "rb") as f:
        eval = pickle.load(f)
    return eval

def structure_tool_response(tr):
    return {"request_id": tr.request_id,
    "tool_name": tr.tool_name,
    "successful": tr.successful,
    "response": tr.response,
    "latency": tr.latency}

def structure_message(message):
    return {"content": message.content, "thinking": message.thinking, "tool_calls": message.tool_calls}

def structure_turns(turns):
    new_turns=[]
    for t in turns:
        new_turns.append({
            "timestamp": t.timestamp,
            "model_message": structure_message(t.model_message),
            "tool_response": structure_tool_response(t.tool_response)})
    return new_turns

def save_as_json(eval, path):
    final_structure={"sessions":{}}
    for s in eval["sessions"]:
        final_structure["sessions"][s.session_id]={
            "prompt": s.prompt, 
            "turns": structure_turns(s.turns),
            "final_output": s.final_output}
    file=open(path,"w")
    file=json.dumps(final_structure, indent=3)
    file.close()

def get_json_format_from_pckl_directory(directory, new_directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pckl"):
            filepath = os.path.join(directory, filename)
            eval=load_eval(filepath)
            save_as_json(eval, os.path.join(new_directory, filename.replace(".pckl", ".json")))