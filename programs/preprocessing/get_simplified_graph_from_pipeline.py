import re
import json

#get reference of every changeable parameter for the nodes
file=open("data/JIPipe/JIPipe_nodes/changeable_parameters.json")
changeable_parameters=json.load(file)
file.close()

file=open("data/JIPipe/JIPipe_nodes/node_id_description.json")
node_description=json.load(file)
file.close()

def get_slots(node, type):
    slots={}
    for slot in node["jipipe:slot-configuration"][type]:
        slots[slot]={"slot-data-type":node["jipipe:slot-configuration"][type][slot]["slot-data-type"]}
    return slots

def break_up_group_nodes(nodes, edges):
    def del_group_node(group_node,nodes_json,edges_list):
        """
            Deletes the group node and reconnects its contents with the outer graph.
            Returns nodes which also contain the group nodes and changed edges.
        """
        id=list(group_node.keys())[0]
        new_edges=[]
        #deleting group node from new nodes
        new_nodes=nodes_json.copy()
        del new_nodes[id]
        

        #get the id of the input/output slots of the group node
        for node_id in group_node[id]["contents"]["nodes"]:
            if group_node[id]["contents"]["nodes"][node_id]["jipipe:node-info-id"]=="graph-wrapper:input":
                input_id=node_id
            elif group_node[id]["contents"]["nodes"][node_id]["jipipe:node-info-id"]=="graph-wrapper:output":
                output_id=node_id
            else:
                new_nodes[node_id]=group_node[id]["contents"]["nodes"][node_id]

        #find all nodes in group, which recieve the group input/output
        input_nodes=[]
        output_nodes=[] 
        for e in group_node[id]["contents"]["edges"]:
            if e["source-node"]==input_id:
                input_nodes.append(e["target-node"])
            if e["target-node"]==output_id:
                output_nodes.append(e["source-node"])

        #redirect the input/output node edges to connec to the other graph
        for e in edges_list:
            if e["target-node"]==id:
                for n in input_nodes:
                    edge=e.copy()
                    edge["target-node"]=n
                    new_edges.append(edge)
            elif e["source-node"]==id:
                for n in output_nodes:
                    edge=e.copy()
                    edge["source-node"]=n
                    new_edges.append(edge)
            else:
                new_edges.append(e)

        #add other edges from group node
        for e in group_node[id]["contents"]["edges"]:
            if e["source-node"]!=input_id and e["target-node"]!=output_id:
                new_edges.append(e)

        return new_nodes, new_edges
    """
        Function for detecting all group nodes and breaking them up into nodes.
    """
    print(nodes.keys())
    print(type(nodes))
    for id in nodes:
        try:
            if nodes[id]["jipipe:node-info-id"]=="node-group":
                new_nodes, new_edges=del_group_node({id:nodes[id]},nodes, edges)
                nodes, edges=break_up_group_nodes(new_nodes,new_edges)
        except KeyError as e:
            pass
    return nodes, edges

def simplify_graph(nodes_json, edges_list):
    """
        Breaks up all group nodes and returns nodes and edges, which have only minimal information.
        Minimal information: only id, name and changeable parameters for each node.
    """

    #first step, break up all group nodes
    nodes_json, edges_list = break_up_group_nodes(nodes_json, edges_list)

    nodes={}
    edges=[]
    for node_id in nodes_json:
        name_id=nodes_json[node_id]["jipipe:node-info-id"]
        #check for nodes which are associated with group nodes
        if name_id!="graph-wrapper:input" and name_id!="graph-wrapper:output":
                #filter out non-node elements (e.g. text-box)
                if not re.match("graph-annotation",name_id):
                    parameters=list(changeable_parameters[name_id]["parameters"].keys())
                    values={}
                    #get changeable parameters, discard others
                    for p in parameters:
                        try:
                            values[p]=nodes_json[node_id][p]
                        except:
                            print("no parameter named "+p+" in node "+name_id)
                    input={}
                    input_list=list(nodes_json[node_id]["jipipe:slot-configuration"]["input"].keys())
                    for i in input_list:
                        input[i]={"slot-data-type":nodes_json[node_id]["jipipe:slot-configuration"]["input"][i]["slot-data-type"]}
                    output={}
                    output_list=list(nodes_json[node_id]["jipipe:slot-configuration"]["output"].keys())
                    for o in output_list:
                        output[o]={"slot-data-type":nodes_json[node_id]["jipipe:slot-configuration"]["output"][o]["slot-data-type"]}
                    nodes[node_id]={"name":name_id, "parameters": values, "input": input, "output": output, "compartment":nodes_json[node_id]["jipipe:graph-compartment"]}
        elif name_id!="jipipe:compartment-output":
            pass
        else:
            print("error, node-group still in pipeline: "+name_id)
    #read out edges
    for e in edges_list:
        try:
            del e['metadata']
        except:
            pass
        if e["source-node"] in nodes.keys() or e["target-node"] in nodes.keys():
            edges.append(e)
    return nodes,edges

def simplify_node_ids(node_ids, graph_text):
    c=0
    new_graph=graph_text
    for node_id in node_ids:
        new_graph=re.sub(node_id, "node"+str(c), new_graph)
        c+=1
    return new_graph

def add_node_description(nodes):
    new_nodes={}
    for node_id in nodes:
        new_nodes[node_id]=nodes[node_id]
        new_nodes[node_id]["description"]=node_description[nodes[node_id]["name"]]
    return new_nodes

def non_default_parameters(nodes):
    def check_default(node):
        non_default_parameters={}
        default_parameters=changeable_parameters[node["name"]]["parameters"]
        for parameter in node["parameters"]:
            if parameter=="folder-paths":
                non_default_parameters[parameter]="your_path/folder"
            elif parameter =="file-names":
                non_default_parameters[parameter]="your_filename"
            elif node["parameters"][parameter]!=default_parameters[parameter]["value"]:
                non_default_parameters[parameter]=node["parameters"][parameter]
        return non_default_parameters
    new_nodes={}
    for node_id in nodes:
        new_nodes[node_id]={"name":nodes[node_id]["name"], "parameters":check_default(nodes[node_id])}
    return new_nodes

def del_parameters(nodes):
    new_nodes={}
    for node_id in nodes:
        new_nodes[node_id]={"name":nodes[node_id]["name"], "description": nodes[node_id]["description"]}
    return new_nodes    

def simplify_graph_from_project(project_file, saving_file, adding_description):
    """
        Simplifies graph of a project. 
    """
    #read in project file
    file=open(project_file)
    project=json.load(file)
    file.close()

    graph=project["graph"]
    edges=graph["edges"]
    nodes=graph["nodes"]
    compartments_json=project["compartments"]["compartment-graph"]

    #read out compartments
    compartments={}
    for node_id in compartments_json["nodes"]:
        name=compartments_json["nodes"][node_id]["jipipe:node:name"]
        alias=compartments_json["nodes"][node_id]["jipipe:alias-id"]
        outputs=list(compartments_json["nodes"][node_id]["jipipe:slot-configuration"]["output"].keys())
        inputs=list(compartments_json["nodes"][node_id]["jipipe:slot-configuration"]["input"].keys())
        compartments[node_id]={"name": name,"alias":alias, "input":inputs, "output": outputs}

    #write pipelines down for each compartment
    compartment_graph={}
    i=0
    for c_id in compartments:
        compartment_nodes={}
        compartment_edges=[]
        for n in nodes:
            if nodes[n]["jipipe:graph-compartment"]==c_id:
                compartment_nodes[n]=nodes[n]
        for e in edges:
            if e["source-node"] in compartment_nodes.keys() or e["target-node"] in compartment_nodes.keys():
                compartment_edges.append(e)
        compartment_nodes,compartment_edges=simplify_graph(compartment_nodes,compartment_edges)
        if adding_description:
            #compartment_nodes=non_default_parameters(compartment_nodes)
            compartment_nodes=add_node_description(compartment_nodes)
            compartment_nodes=del_parameters(compartment_nodes)
        compartment_graph[c_id]=compartments[c_id] | {"graph": {"nodes": compartment_nodes, "edges": compartment_edges}}
    #saving simplified project
    file=open(saving_file,"w")
    file.write(json.dumps(compartment_graph,indent=4))
    file.close()
    print(f"pipeline {project_file} simplified and saved at {saving_file}")