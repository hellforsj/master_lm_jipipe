from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import json

@dataclass
class Tokens:
    """_summary_

    Returns:
        _type_: _description_
    """
    reasoning_start: str
    reasoning_end: str
    tool_response_start: str
    tool_response_end: str
    tool_call_start: str
    tool_call_end: str

@dataclass
class ToolCall:
    """_summary_

    Returns:
        _type_: _description_
    """
    tool_name: str
    args: Dict[str, Any]
    request_id: str

    def __str__(self):
        if self.args==None:
            return f"Tool name: {self.tool_name}, Tool arguments: #ERROR"
        else:
            return f"Tool name: {self.tool_name}, Tool arguments: {json.dumps(self.args)}"

@dataclass
class ToolResponse:
    """_summary_

    Returns:
        _type_: _description_
    """
    request_id: str
    tool_name: str
    successful: bool
    response: Dict
    latency: float

    def __str__(self):
        if self.response==None:
            return f"Tool name: {self.tool_name}, Sucessful: {self.successful}, Tool response: #ERROR"
        else:
            return f"Tool name: {self.tool_name}, Sucessful: {self.successful}, Tool response: {json.dumps(self.response)}"

@dataclass
class Message:
    """_summary_

    Returns:
        _type_: _description_
    """
    content: str = None
    thinking: str = None
    tool_calls: List = None
    def __str__(self):
        return f"**Model**:\n\tcontent:{self.content}\n\treasoning:{self.thinking}\n\t{json.dumps(self.tool_calls)}"

@dataclass
class Turn:
    """_summary_

    Returns:
        _type_: _description_
    """
    timestamp: float
    model_message: Optional[Message] = None
    tool_response: Optional[ToolResponse] = None

@dataclass
class Session:
    """_summary_

    Returns:
        _type_: _description_
    """
    session_id: str
    prompt: str
    functions:List[Callable]
    turns: List[Turn] = field(default_factory=list)
    final_output: Optional[str] = None
    expected: Any = None
    notes: str = ""
    success: Optional[bool] = None

@dataclass
class Session2:
    """_summary_

    Returns:
        _type_: _description_
    """
    session_id: str
    prompt: str
    functions:List[Callable]
    turns: List[Turn] = field(default_factory=list)
    final_output: Optional[str] = None
    expected: Any = None
    success: Optional[bool] = None