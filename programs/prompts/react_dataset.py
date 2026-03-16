from .base import DESCRIPTION_JIPIPE
from .inference import TOOL_DESCRIPTION
import json

QUESTION_PROMPT=f"""{DESCRIPTION_JIPIPE}
You will be given a JIPipe pipeline in json format that lists all nodes and their edges. There is a description provided for each of the nodes as well as a short description of the pipeline.
You are responsible for generating a dataset about JIPipe pipeline creation. The dataset is made up of question/answer pairs and you are responsible for generating a fitting question/prompt.
The question should adhere to the following criteria:
- The question should reflect the scenario of a user/researcher asking a JIPipe assistant to build a pipeline.
- The given pipeline should answer the users request/problem.
- The question should be kept relatively short. Do not provide any steps that would solve the problem.
- Keep in mind, that the user does not know what the finished pipeline looks like. The user also does not know which steps to take to solve this problem. Do not include the given pipeline or parts of it in the question.
- Only include the question in your answer.

Generate a question/prompt that meets the criteria about the following pipeline. 

Pipeline: 
    {{pipeline}}"""

REASONING_PROMPT=f"""
{DESCRIPTION_JIPIPE}
You will be given a JIPipe pipeline in json format that lists all nodes and their edges. There is a description provided for each of the nodes. You will be given a question/prompt fitting the functionality of the pipeline as well.
You are simulating an assistant that incrementally builds a JIPipe pipeline. The assistant builds the pipeline according to the needs of a user. The user states the desired functionality in the question.
You are responsible for generating the individual steps of the chain-of-thought process that lead to the building of the given pipeline. To generate the steps, you are also given the previous steps already taken. 
You are responsible for generating the next step in the reasoning. The next step is adding the next node in the pipeline. You are given the node for which you are supposed to give the reasoning.
Follow these rules:
- Write a very short reasoning step of the thought process behind the JIPipe pipeline-building assistant, which builds a pipeline that fulfills the user's stated needs.
- Do not mention actual node names, only describe the intended functionality. Node names are only retrieved via tool calls. The intention to retrieve a node name via tool calls should be clearly stated in the step.
- The reasoning step should very shortly reflect upon the previous step (if provided) and base further actions on it.
- The reasoning step should consider whether the previous responses from the tool calls are appropriate for building the pipeline according to the user's stated needs.
- The reasoning step should describe very shortly whether tool calls should be taken. Actual tools should not be called. 
- All reasoning steps in total should ultimately lead to the given pipeline.
- Provide only the next reasoning step. This step should lead to the inclusion of the given node as the next node in the pipeline.
- The reasoning step should not be longer than 2 sentences.

Question/prompt: {{question}}

The current thought process and accompanying tool calls:
{{prev_steps}}

Node to include in the pipeline for which the current reasoning step is generated:
{{node}}

Pipeline:
{{pipeline}}

Tools later to be used by the assistant:
{json.dumps(TOOL_DESCRIPTION, indent=3)}
"""

TOOL_CALL_PROMPT=f"""{DESCRIPTION_JIPIPE}

You are simulating an assistant that incrementally builds a JIPipe pipeline. The assistant performs tool calls to build the pipeline.
The assistant current reasoning step is:
{{reasoning_step}}

The tools available are:
{json.dumps(TOOL_DESCRIPTION, indent=3)}

Generate the tool calls necessary based on the current reasoning step.
Give the tool calls in the following format:
[{{"name": "name of tool", "arguments": {{"name of the argument": "value of the argument", ...}}}}, ...]

Only give the tool calls in the stated format. Do not return anything else."""

TOOL_ANSWER_PROMPT=f"""{DESCRIPTION_JIPIPE}

You are simulating an assistant that incrementally builds a JIPipe pipeline. The assistant performs tool calls to build the pipeline. The finished pipeline is given in json format.
The assistant current reasoning (with tool calls) is:
{{reasoning}}

The current new reasoning step is:
{{step}}

This step lead to the following tool call(s):
{{tool_call}}

The tools available are:
{json.dumps(TOOL_DESCRIPTION, indent=3)}

Finished pipeline:
{{pipeline}}

Generate the result of the tool calls of the new reasoning step.
Give the results in the following format:
[{{"name of the return argument": "value of the return argument", ...}}, ...]

The generated results should adhere to the following criteria:
- The tool call used to retrieve node names should use the given pipeline and the reasoning to determine, which of the nodes should be the result of that tool call.
- The tool call used to determine whether a connection between two nodes is posstible should use the given pipeline as a basis. The given pipeline is correct and connections present in the given pipeline are always valid.
- Do not make up the result of the tool call, use the given pipeline to retrieve the correct result.
- All tool calls are used to build the given pipeline. This should reflect in the result the tools give.

Only give the result of the tool calls in the stated format. Do not return anything else."""

NODE_SEARCH_PROMPT=f"""{DESCRIPTION_JIPIPE}
You are given a JIPipe pipeline and a chain-of-thought process behind constructing the pipeline. Each node used in the pipeline was retrieved by calling a tool that, when given a short prompt of the desired node functionality, provides the correct node name.
You are responsible for simulating the tool calls for every node in the pipeline.
Follow the stated criteria for generating the tool calls:
- For every tool call, only generate the short prompt, with which to retrieve the node.
- Use the given descriptions of the nodes to generate the prompt.
- Keep the prompts short, they should not exceed 10 words.
- Use the chain-of-thought process to formulate the prompts.

Your answer should only include the generated prompts in the following format:
- name of the node: generated prompt
- ...

chain-of-thoughts: {{cot}}
pipeline: {{pipeline}}"""

FINAL_ANSWER_PROMPT=f"""{DESCRIPTION_JIPIPE}
You will be given a JIPipe pipeline in json format that lists all nodes and their edges. There is a description provided for each of the nodes as well as a short description of the pipeline.
You are responsible for generating a dataset about JIPipe pipeline creation. The dataset is made up of question/answer pairs and a chain-of-though process and you are responsible for generating a fitting answer.
The answer should adhere to the following criteria:
- The answer should be a suitable response to the question that was given.
- The answer should reflect the scenario of the pipeline building assistant answering a users request to build a pipeline.
- The answer should include a short description of the main steps the pipeline takes.
- The answer should not include the pipeline.
- The answer should be the result of the chain-of-thought process.

Generate a answer that meets the criteria and answers the given question. Also, base your answer on the chain-of-thought process and the following pipeline. 

Question: {{question}}

Chain-of-thought process: {{cot}}

Pipeline: 
    {{pipeline}}"""
#-------------------------JUGDE-----------------------------------------------
TEST_REASONING_PROMPT=f"""{DESCRIPTION_JIPIPE}
You are a quality evaluator. You will be given a JIPipe pipeline in json format, a question/prompt asked by the user aiming to recieve this pipeline as an answer, a node of the pipeline in json format, and a reasoning process behind adding that node to the pipeline.
You recieve the newest step in the reasoning process. This step is supposed to give the reasoning to add the given node to the pipeline. You must evaluate, if the newest step fits the following criteria.
- The step fits into the current reasoning process.
- The step is goal-oriented. The goal is to incremently build a JIPipe pipeline by retrieving the correct node names for the nodes in the pipeline and checking if a connection between the nodes is possible.
- The step describes the incremental building of a JIPipe pipeline.
- The step reflects upon the previous reasoning and tool responses (if existent).
- The reasoning process is tailored to the user's question.
- The reasoning step fits the given node.
- The reasoning step is relatively short.

Answer "yes", if the step meets all quality criteria, "no" otherwise. Only include either "yes" or "no" in your answer. Only answer "yes" if you a absolutely sure.

Question/prompt: {{question}}
Node: {{node}}
New step: {{step}}
Previous steps: {{previous_steps}}
Pipeline: {{pipeline}}"""

TEST_TOOL_CALL_PROMPT=f"""{DESCRIPTION_JIPIPE}

You are a quality evaluator. You will be given a reasoning step that aims to incrementally build a JIPipe pipeline, as well as the accompanying tool calls.
Determine if the tool calls are appropriate given the reasoning step and the description of the tools available. The tool calls should adhere to the following criteria:
- The tool calls are appropriate for the reasoning step.
- The tool calls are given in the following format: [{{"name": "name of tool", "arguments": {{"name of the argument": "value of the argument", ...}}}}, ...]
- The tool calls only include available tools.

Answer "yes", if the tool calls meets all quality criteria, "no" otherwise. Only include either "yes" or "no" in your answer. Only answer "yes" if you a absolutely sure.

Reasoning step: {{step}}
Tool calls: {{tool_call}}
Available tools: {json.dumps(TOOL_DESCRIPTION, indent=3)}"""

TEST_TOOL_ANSWER_PROMPT=f"""{DESCRIPTION_JIPIPE}

You are a quality evaluator. You will be given a reasoning step that aims to incrementally build a JIPipe pipeline with the tool call used to help build the pipeline, as well as the result of the tool call.
Determine if the result of the tool call is appropriate given the reasoning step and the description of the tools available. The result of the tool call should adhere to the following criteria:
- The tool call used to retrieve node names should use the given pipeline and the reasoning to determine, which of the nodes should be the result of that tool call.
- The tool call used to determine whether a connection between two nodes is posstible should use the given pipeline as a basis. The given pipeline is correct and connections present in the given pipeline are always valid.
- The given pipeline is used to retrieve the correct result.
- All tool calls are used to build the given pipeline. This should reflect in the result the tools give.
- The tool calls should adhere to the following format: [{{"name of the return argument": "value of the return argument", ...}}, ...]

Answer "yes", if the tool calls meets all quality criteria, "no" otherwise. Only include either "yes" or "no" in your answer. Only answer "yes" if you a absolutely sure.

Pipeline: {{pipeline}}
Reasoning step: {{step}}
Tool calls: {{tool_call}}
Available tools: {json.dumps(TOOL_DESCRIPTION, indent=3)}
Tool call result to check: {{tool_answer}}"""

TEST_QUESTION_PROMPT=f"""{DESCRIPTION_JIPIPE}
You are a quality evaluator. You will be given a JIPipe pipeline in json format including a short description of the pipeline and a question/prompt. 
The question/prompt was generated by another model to represent a real-life scenario of a JIPipe users problem to which the stated pipeline is the solution. 
You must assess, if the question meets the following criteria:
- The pipeline fits as an answer to the question.
- The question is short and to the point.
- The question is not ambiguous.
- The question reflects the scenario of a JIPipe user trying to build a pipeline and asking an assistant for help.

Answer "yes", if the question meets all quality criteria, "no" otherwise. Only include either "yes" or "no" in your answer. Only answer "yes" if you a absolutely sure.

Question/prompt: {{question}}
Pipeline: {{pipeline}}
"""

TEST_FINAL_ANSWER_PROMPT=f"""{DESCRIPTION_JIPIPE}
You are a quality evaluator. You will be given a question/answer pair representing the request of a user asking a JIPipe assistant to build a JIPipe pipeline, a chain-of-thought process describing the thought process the assistant takes to build the JIPipe pipeline, and a JIPipe pipeline in json format including a short description of the pipeline. 
You have to evaluate, if the answer correctly represents an answer to the question and respects the chain-of-thought process.
You must assess, if the answer meets the following criteria:
- The answer should be a suitable response to the question that was given.
- The answer should reflect the scenario of the pipeline building assistant answering a users request to build a pipeline.
- The answer should include a short description of the main steps the pipeline takes.
- The answer should not include the pipeline.
- The answer should be the result of the chain-of-thought process. 

Answer "yes", if all quality criteria are met, "no" otherwise. Only include either "yes" or "no" in your answer. Only answer "yes" if you a absolutely sure.

Question: {{question}}

Answer: {{answer}}

Chain-of-thought process: {{cot}}
Pipeline: {{pipeline}}
"""