#do NOT use gemma, does not work

import ollama
import json
import os
import shutil
import re
import datetime
from prompts import DESCRIPTION_JIPIPE, FORMAT_JIPIPE_PIPELINE

current_datetime = datetime.datetime.now()
formatted_time = current_datetime.strftime("%d_%m_%Y__%H_%M")
log=open(os.getcwd()+"/log_automatic_splitting_"+formatted_time+".log","w")

class GraphException(Exception):
    pass

class DirectoryException(Exception):
    pass

class StuckLoopException(Exception):
    pass

class EndOfGeneration(Exception):
    pass

def model_generate(prompt,model):
    """
        Wrapper for using model.
        Args:
            prompt (str): input prompt for the model.
            model (str): name of the model load in ollama
        Returns:
            response (str): output text of the model. Does not include tool calls etc.
    """

    message={
        "role":"user",
        "content":prompt
    }
    response=ollama.generate(model=model, prompt=prompt)
    return response.response.strip()

def split_pipeline(short_pipelines, base_pipeline):
    """
        Splits the base graph in the stated short pipelines.
        Args:
            short_pipelines(dict): short pipeline description in json format. Format: {pipeline_1: {"description": "description of pipelnie", "start": [list of starting nodes], "end": [list of end nodes]}, ...}
            base_pipeline(dict): json format of graph from which to extract the short pipelines
        Returns:
            graphs(dict): json format of all short pipelines (with description of pipeline, nodes and edges)

    """
    if re.match("```json",short_pipelines):
        short_pipelines=short_pipelines.replace("```json","")
        short_pipelines=short_pipelines.replace("```","")
    try:
        short_pipelines=json.loads(short_pipelines)
        def extract_subgraph(graph, start_nodes, end_nodes):
            """
                Gives a subgraph of a graph given its start and end points.
                Args:
                    graph(dict): Main graph from which to extract the subgraph
                    start_nodes(list): starting nodes of the subgraph
                    end_nodes(list): end nodes of the subgraph
                Returns:
                    subgraph(dict): subgraph with all nodes and its edges in json format (similar to JIPipe graph format)

            """
            nodes = graph["nodes"]
            edges = graph["edges"]

            # Identify nodes to include in the subgraph
            reachable_nodes = set(start_nodes).union(set(end_nodes))
            queue = list(start_nodes.copy())  # Create a copy to avoid modifying the original list

            while queue:
                node_id = queue.pop(0)
                
                # Find edges connected to the current node
                for edge in edges:
                    if edge['source-node'] == node_id and edge['target-node'] not in reachable_nodes:
                        reachable_nodes.add(edge['target-node'])
                        queue.append(edge['target-node'])

            # Filter nodes and edges to create the subgraph
            subgraph_nodes = {
                node_id: nodes[node_id] for node_id in reachable_nodes if node_id in nodes
            }
            subgraph_edges = [
                edge for edge in edges
                if edge['source-node'] in reachable_nodes and edge['target-node'] in reachable_nodes
            ]

            # check if the end nodes are present in the subgraph
            if not all(end_node in reachable_nodes for end_node in end_nodes):
                raise GraphException("Not all end nodes are reachable from the start nodes.")
            
            return {
                'nodes': subgraph_nodes,
                'edges': subgraph_edges
            }
        graphs={}
        for pipeline in short_pipelines:
            graphs[pipeline]={}
            #graphs[pipeline]["description"]=short_pipelines[pipeline]["description"]
            #graphs[pipeline]=extract_subgraph(base_pipeline,short_pipelines[pipeline]["start"],short_pipelines[pipeline]["end"])
            graphs[pipeline]= {"description":short_pipelines[pipeline]["description"]} | extract_subgraph(base_pipeline,short_pipelines[pipeline]["start"],short_pipelines[pipeline]["end"])
        return graphs
    except json.decoder.JSONDecodeError as e:
        print(e)

def correct_output(output, base_pipeline):
    """
        Test if the output of the generated response satisfies the desired format.
        Args:
            output(str): Text of the generated response
        Returns:
            decision(bool): "True", if the output comlies to the format, otherwise "False"
    """
    try:
        num_pipelines=len(output)
        threshold=num_pipelines/2   #at least half of the pipelines should meet the criteria
        n_correct_pipelines=num_pipelines 
        print("length of short pipelines: ")
        for p in output:
            len_pipeline=len(output[p]["nodes"])
            print(len_pipeline)
            if len_pipeline<2 or len_pipeline>10:
                n_correct_pipelines=n_correct_pipelines-1
        if n_correct_pipelines>=threshold:
            return True
        else:
            print("Too many generated pipelines are too long/short")
            return False
    except Exception as e:
        print(e)
        return False

def get_short_pipelines(graph,model):
    """
        Recives a pipeline in json format (similar to JIPipe json format, graph{nodes:{...}, edges:{...}}) and automatically splits it into smaller pipelines, which have a simple objective/solve a simple task.
        Args:
            graph(dict): json object of the main graph from which to generate the smaller pipelines (usually a compartment)
            model(str): Name of model which to use to generate the response.
        Returns:
            short_pipelines(dict): short pipeline description in json format. Format: {pipeline_1: {"description": "description of pipelnie", "start": [list of starting nodes], "end": [list of end nodes]}, ...}
    """
    prompt=f"""Provided JIPipe pipeline:
{json.dumps({"graph":graph},indent=4)}



{DESCRIPTION_JIPIPE}
You are given a JIPipe pipeline in json format that lists all nodes and their edges. There is a description provided for each of the nodes. 
{FORMAT_JIPIPE_PIPELINE}
You are responsible for generating very short pipelines from the pipeline. These short pipelines should sovle a simple task.
Example: You are given a pipeline, that reads in a folder with files and proceeds to apply to each of the files filtering and thresholding and finally saves all the results from this to another folder. The short pipelines are a pipeline, that does the file handling (read in folder and files), a pipeline that does the image preprocessing (filtering, thresholding), and a pipeline, that does the analysis (parameters from preprocessing are saved).

The generated short pipelines should adhere to the following criteria:
- The short pipelines should be generated by "cutting up" the given pipeline. 
- The short pipelines should be given by stating the starting points and end points of the graph.
- The short pipelines should at least include 2 nodes and should not be longer than 10 nodes.
- The short pipelines should not solve complex tasks or multiple tasks. Keep the solved tasks very simple. Each pipeline should only solve a single short task.
- Each of the short pipelines should be unique. Do not generate the same short pipeline multiple times.
- Provide for each short pipeline the main objective/which task that is solved by applying that pipeline and describe the main steps which the pipeline takes to takle the task.
- Give your answer of the short pipelines in the following format. Only provide in your answer the short pipelines in the desired format. Do not include further explaination. 
{{"pipeline_1":  {{"description": "description of main objective/task of pipeline and steps as continuous text",
  "start":[node ids from nodes at the beginning of the pipeline],
  "end": [node ids from nodes at the end of the pipeline]}},
  "pipeline_2": {{...}},
  ...}}"""
    short_pipelines=model_generate(prompt, model)
    print("generated result: \n"+short_pipelines)
    return split_pipeline(short_pipelines,graph)

def correct_pipeline(pipeline,model):
    prompt=f"""{DESCRIPTION_JIPIPE}
You are a quality evaluator. You are given a short JIPipe pipeline and a description of its functionality. Decide, if the pipeline satisfies the task that is given in the description. 
Answer "yes", if the pipeline fits the description, otherwise answer "no".
"""
    response=model_generate(prompt, model)
    if re.search("yes", response,re.IGNORECASE):
        return True
    else:
        return False
    
def change_description(pipeline, model):
    prompt=f"""{json.dumps({"graph":{"nodes":pipeline["nodes"], "edges": pipeline["edges"]}}, indent=4)}


{DESCRIPTION_JIPIPE}
You are given a JIPipe pipeline in json format that lists all nodes and their edges. There is a description provided for each of the nodes. 
{FORMAT_JIPIPE_PIPELINE}
You are responsible for generating a description of the pipeline based on its structure and the description of the nodes.

The generated description should adhere to the following criteria:
- The description should give a short overwiev of what the pipelines function is.
- Do not use any names of nodes in the description.

Only reply with the new description. Do not add other sentences which are not part of the description.
"""
    return model_generate(prompt, model)

def make_directory(dir_name, overwrite):
    try:
        if overwrite and os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        log.write((f"Directory '{dir_name}' created successfully.\n"))
        log.flush()
    except FileExistsError:
        raise DirectoryException(f"Directory '{dir_name}' already exists.")
    except PermissionError:
        raise DirectoryException(f"Permission denied: Unable to create '{dir_name}'.")
    except Exception as e:
        raise DirectoryException(e)

def try_generation(gen_function, test_function, f_args, t_args, n_tries):
    """
        Wrapper for generating question, answer, cot, and tool calls.
        Args:
            gen_function (function): function responsible for generating desired part of dataset
            test_function (function): function for testing the quality of the generated result of gen_function
            f_args (list): arguments of the gen_function
            t_args (list): arguments of the test_function excluding the result of gen_function, which is always the first argument of test_function
            n_tries (int): number of tries before EndOfGeneration Exception is raised (prevents stuck loops)
    """
    fitting=False
    tries=0
    while not fitting and tries<n_tries:
        tries+=1
        print("try: "+str(tries))
        result=gen_function(*f_args)
        args=[result]+t_args
        fitting=test_function(*args)
    if tries<n_tries:
        return result
    else:
        if fitting:
            return result
        else:
            raise EndOfGeneration(f"Max. amount of tries ({n_tries}) reached during {gen_function.__name__}")

#base_path="dataset/structure_pipeline/automatic_short_pipelines/"
def short_pipelines_from_project(project_path, base_path, model, overwrite, tries):
    project=json.load(open(project_path))
    project_name=os.path.basename(project_path).split(".")[0]
    project_dir=base_path+project_name
    try:
        make_directory(project_dir,overwrite)
        print("project: "+project_name)
        for compartment_id in project:
            graph=project[compartment_id]["graph"]
            compartment_name=project[compartment_id]["name"]
            compartment_dir=project_dir+"/"+compartment_name
            make_directory(compartment_dir,overwrite)
            print("compartment: "+compartment_name)
            t=0
            while t<tries:
                try:
                    split_pipelines=try_generation(get_short_pipelines, correct_output, [graph, model],[graph], tries)
                    for pipeline_id in split_pipelines:
                        if not correct_pipeline(split_pipelines[pipeline_id],model):
                            split_pipelines[pipeline_id]["description"]=try_generation(change_description, correct_pipeline,[split_pipelines[pipeline_id], model],[model],tries)
                        new_file=open(compartment_dir+"/"+pipeline_id+".json","w")
                        new_file.write(json.dumps(split_pipelines[pipeline_id], indent=4))
                        new_file.close()
                    t+=1
                except GraphException as e:
                    log.write(str(e)+"\n")
                    log.write("Retrying generating short pipelines.\n")
                    log.flush()
                    t+=1
                except json.decoder.JSONDecodeError as ej:
                    log.write(str(ej)+"\n")
                    log.write("Retrying generating short pipelines.\n")
                    log.flush()
                    t+=1
                except EndOfGeneration as eg:
                    log.write("Reached end of tries ("+str(tries)+") for compartment "+compartment_name+" in file "+project_path+"\n")  
                    log.flush()
                    break
    except DirectoryException as e:
        log.write(f"Stopping due to: {str(e)}\n")
        log.flush()

""" def simplify_all_files_in_dir(directory, new_directory):
    for file in os.scandir(directory):  
        if file.is_file():
            get_simplified_graph_from_pipeline.simplify_graph_from_project(file.path, new_directory+file.name,True)
 """
def short_pipelines_from_dir(directory, new_directory, model, overwrite, tries):
    for file in os.scandir(directory):  
        if file.is_file() and (file[-4:]==".jip" or file[-4:]==".json"):
            short_pipelines_from_project(file.path, new_directory, model, overwrite, tries)

#simplify_all_files_in_dir("jipipe_pipelines", "simplified_pipelines/")
short_pipelines_from_dir("simplified_pipelines", "split_pipelines_2/","llama3.3:latest-10k", True, 15)
#short_pipelines_from_project("test.json","test/", "llama3.3:latest-10k", True, 15)

log.close()

