
from typing import List, Dict, Any, Optional, Callable
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import uuid
import signal
from prompts import TOOL_DESCRIPTION, SYSTEM_PROMPT_JUDGE
from data_classes import *

#Time Out Handler (to stop endless generation)
#----------------------------------------------------------------------------------------------------------
def timeout_handler(signum, frame):
        raise TimeoutError()

#Model Classes
#----------------------------------------------------------------------------------------------------------
class HFModel:
    def __init__(self, model_path, system_prompt: str="You are a helpful assistant", special_tokens: Tokens=None, chat_template: str=None, n_ctx: int=16384):
        self.model_path=model_path
        self.system_prompt=system_prompt
        self.history=[{"role":"system", "content": self.system_prompt}]
        self.llm=AutoModelForCausalLM.from_pretrained(self.model_path, device_map={"": "cuda:0"}, torch_dtype="auto")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path)
        self.chat_template=chat_template
        if self.chat_template:
            self.tokenizer.chat_template= self.chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.tokenizer.eos_token_id
        self.current_response=Message()
        self.tools=None
        self.n_ctx=n_ctx  
        if special_tokens is not None:
            self.special_tokens=special_tokens
        else:
            self.special_tokens=Tokens("<think>", "</think>", "<tool_response>", "</tool_response>", "<tool_call>", "</tool_call>")
        pattern_tool_calls = r'''\s(\[\s*)?\{\s*"name"\s*:\s*"\S+"\s*,\s*"arguments"\s*:\s*\{[^\{\}]+\}\s*\}(?:\s*,\s*\{\s*"name"\s*:\s*"\S+"\s*,\s*"arguments"\s*:\s*\{[^\{\}]+\}\s*\})?(\s*\])?\s'''
        self.tool_call_regex = re.compile(
            re.escape(self.special_tokens.tool_call_start)
            + pattern_tool_calls
            + re.escape(self.special_tokens.tool_call_end),
            re.VERBOSE | re.DOTALL
        )
        pattern_reasoning=r'''.+'''
        self.reasoning_regex= re.compile(
            re.escape(self.special_tokens.reasoning_start)
            + pattern_reasoning
            + re.escape(self.special_tokens.reasoning_end),
            re.VERBOSE | re.DOTALL
        )

    def init_chat(self, prompt:str):
        """
            Initializes model chat with the user request.
            Args:
                prompt (str): user prompt
        """
        self.clear_history()
        self.history.append({'role': 'user', 'content': prompt})
    
    def update_history(self,  response: Message):
        """
            Updates history of model.
            Args: 
                message: message from the model that is added to the history
        """
        chat_response={"role": "assistant", "content": response.content, "reasoning_content": response.thinking}

        if response.tool_calls:
            #print({"tool_calls": {"function": response.tool_calls}})
            #chat_response.update({"tool_calls": {"function": json.dumps(response.tool_calls)}})
            chat_response["tool_calls"] = [
            {"function": tc}
            for i, tc in enumerate(response.tool_calls)
        ]
        self.history.append(chat_response)

    def clear_history(self):
        """
            Clears history to include only the system prompt.
        """
        self.current_response=Message()
        self.history=[{"role":"system", "content": self.system_prompt}]

    def model_response(self):
        """
            Wrapper for ChatResponse of model
            Args:
                messages: input mesages to the model
            Returns:
                response_message: response message of the model
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        try:
            input= self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
            #print(self.history)
            encoded_input=self.tokenizer(input, return_tensors="pt").to(self.llm.device)
            response_encoded= self.llm.generate(input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"], max_new_tokens=4096, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,)
            prompt_length = encoded_input["input_ids"].shape[-1]
            response= self.tokenizer.decode(response_encoded[0][prompt_length:], skip_special_tokens=False)
            self.current_response=self.format_model_output(response)
            #print(response)
            return self.current_response
        except TimeoutError as e:
            return None

    def format_model_output(self, model_response: str):
        content=model_response

        thinking_match=self.reasoning_regex.search(model_response)
        if not thinking_match:
            thinking=None
        else:
            thinking=thinking_match.group(0)
            content=content.replace(thinking,"")
            thinking=thinking.replace(self.special_tokens.reasoning_start,"").replace(self.special_tokens.reasoning_end,"").strip()
        
        tool_calls_match = self.tool_call_regex.search(model_response)
        if not tool_calls_match:
            tool_calls=None
        else:
            tool_calls_text = tool_calls_match.group(0)
            content=content.replace(tool_calls_text, "").strip()
            tool_calls=tool_calls_text.replace(self.special_tokens.tool_call_start,"").replace(self.special_tokens.tool_call_end,"").strip().replace("'",'"')
            tool_calls=json.loads(tool_calls)
            if not isinstance(tool_calls, list):
                tool_calls=[tool_calls]

        content=content.strip()
        if len(content)==0:
            content=None
        
        formatted_response=Message(content=content, thinking=thinking, tool_calls=tool_calls)
        return formatted_response

class Model(HFModel):
    """
        Wrapper around llama.cpp model. Includes functions for generating model output and updating history
    """
    def __init__(self, model_path, tools: list[Callable], system_prompt: str="You are a helpful assistant", special_tokens: Tokens=None, chat_template: str=None, n_ctx: int=16384, tool_description: Dict = TOOL_DESCRIPTION, temperature: float = 0.2):
        super().__init__(model_path,system_prompt, special_tokens, chat_template, n_ctx)
        self.tools=tools
        self.available_functions= {t.__name__:t for t in tools}
        self.temperature=temperature
        self.tool_description=tool_description
        self.system_prompt=self.system_prompt+"\n"+self.tool_description
    
    def tool_router(self, tool_calls: list[ToolCall]) -> list[ToolResponse]:
        tool_responses = []

        for call in tool_calls:
            start = datetime.datetime.now()

            if call.tool_name not in self.available_functions:
                tool_responses.append(
                    ToolResponse(
                        request_id=call.request_id,
                        tool_name=call.tool_name,
                        successful=False,
                        response={"error": "Unknown tool"},
                        latency=0.0,
                    )
                )
                continue

            try:
                func = self.available_functions[call.tool_name]
                result = func(**call.args)

                end = datetime.datetime.now()
                tool_responses.append(
                    ToolResponse(
                        request_id=call.request_id,
                        tool_name=call.tool_name,
                        successful=True,
                        response=result,
                        latency=(end - start).total_seconds(),
                    )
                )

            except Exception as e:
                end = datetime.datetime.now()
                tool_responses.append(
                    ToolResponse(
                        request_id=call.request_id,
                        tool_name=call.tool_name,
                        successful=False,
                        response={"error": str(e)},
                        latency=(end - start).total_seconds(),
                    )
                )

        return tool_responses

    def update_tool_response(self, tool_responses: List[ToolResponse]):
        """
            Updates the history by including tool responses.
            Args:
                tool_responses list[ToolResponses]: list of responses given by the tools
        """
        list_responses=[]
        for tr in tool_responses:
            if tr.successful:
                list_responses.append(tr.response)
            else:
                list_responses.append(None)
        self.history.append({"role": "tool", "content": json.dumps(list_responses)})
    
    def extract_tool_call(self, response: Message) -> Optional[ToolCall]:
        """
            Args:
                response: response message from model)
            
            Returns:
                tool_calls(Optional[ToolCall]): list of tool calls in response message, None if there are no calls

        """
        if response.tool_calls:
            tool_calls=[]
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(tc["name"], tc["arguments"], request_id=str(uuid.uuid4())))
            return tool_calls
        else:
            return None
    
    def single_turn_generation(self, new_input: Message=None):
        """
            Updates history and generates model output. Can be used for new input from user or as step in multi-turn tool calling.
            Args:
                new_input: Optional, if given, new message from user is added to history.
            Returns:
                response: response of model based on current history of model
        """
        if new_input:
            self.update_history(new_input)
        response=self.model_response()
        self.update_history(response)
        return response

class LLMJudge(HFModel):
    """
        Wrapper for LLM-as-judge, Implementaion with ollama
    """

    def __init__(self, model_path: str, system_prompt: str=SYSTEM_PROMPT_JUDGE, temperature: float= 0.2, n_ctx: int=16384):
        super().__init__(model_path, system_prompt)
        self.temperature=temperature
        self.n_ctx=n_ctx
        self.llm = AutoModelForCausalLM.from_pretrained(model_path,  device_map={"": "cuda:0"}, torch_dtype="auto")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path)



    def create_transcript(self, session: Session):
        multi_turn_transcript=""
        for turn in session.turns:
            if turn.model_message:
                if turn.model_message.thinking:
                    multi_turn_transcript+=f"<think>{turn.model_message.thinking}</think>+\n\n"
            if turn.tool_call:
                tool_str= "\n".join(str(x) for x in turn.tool_call)
                multi_turn_transcript+=f"Tool calls: {tool_str}+\n\n"
            if turn.tool_response:
                response_str= "\n".join(str(x) for x in turn.tool_response)
                multi_turn_transcript+=f"Tool responses: {response_str}+\n\n"
        multi_turn_transcript+=f"Final Answer: {session.final_output}"
    
        session_text=f"""=== User Prompt: {session.prompt}
        === Model Session ===
        {multi_turn_transcript}
        """
        return session_text

    def grade(self, session: Session) -> Dict[str, Any]:
        """
        Returns an LLM-structured grade.
        """

        self.init_chat(self.create_transcript(session))
        grading=self.model_response().content

        #parse json grading
        try:
            return json.loads(grading.replace("'",'"'))
        except:
            return {
                "score": 0.0,
                "feedback": "Judge output not parseable: " + grading
            }
