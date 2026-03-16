import json

TOOL_DESCRIPTION=[
    {
        'name' : 'get_node_name',
        'description' : 'Based on a short prompt stating the desired function, the best node is found. Gives the JIPipe node id and the description of this node as a result.',
        'input' : {
            'properties' : {
                'prompt' : {
                    'type' : 'string',
                    'description' : 'Very short and to the point description of the desired functionality of the node.',
                },
            'required' : ['prompt']
            }
        },
        'output' : {
            'properties' : {
                'node_name' : {
                    'type' : 'string',
                    'description' : 'JIPipe node id of node best fitting the prompt.',
                },
                'node_description' : {
                    'type' : 'string',
                    'description' : 'Description of the functionality of the node of id node_name.',
                }
            }
        }
    },
    {
        'name' : 'check_connection',
        'description' : 'Returns True, if a connection/edge from node_a to node_b is possible, otherwise returns False.',
        'input' : {
            'properties' : {
                'node_a' : {
                    'type' : 'string',
                    'description' : 'JIPipe id of the source node.'
                },
                'node_b' : {
                    'type' : 'string',
                    'description' : 'JIpipe id of the target node.'
                },
            'required' : ['node_a, node_b']
            }
        },
        'output' : {
            'properties' : {
                'can_connect' : {
                    'type' : 'bool',
                    'description' : 'Decision on whether a connection from node_a to node_b is possible.',
                }
            }
        }
    }
]

TOOL_DESCRIPTION_NATIVE_REASONING=f"""Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>{json.dumps(TOOL_DESCRIPTION, indent=3)}</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{\"name\": <function-name>, \"arguments\": <args-json-object>}}
</tool_call><|im_end|>
"""
TOOL_DESCRIPTION_NO_REASONING=f"""You have access to a set of tools. When using tools, make calls in a single JSON array: 

[{{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}}, ... (additional parallel tool calls as needed)]

Do not interpret or respond until tool results are returned. Once they are available, process them or make additional calls if needed. For tasks that don't require tools, such as casual conversation or general advice, respond directly in plain text. The available tools are: {json.dumps(TOOL_DESCRIPTION, indent=3)}"""