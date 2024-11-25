from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage


def stream_response(response, return_output=False):
    """
    Print the response from the AI model, streaming each chunk as it is processed.

    This function iterates through each item in the `response` iterable. If the item is an instance of `AIMessageChunk`,
    it extracts the content of the chunk and prints it. If the item is a string, it prints the string directly.
    Optionally, the function can return a concatenated string of all response chunks.

    Parameters:
    - response (iterable): An iterable of response chunks that can be an instance of `AIMessageChunk` or a string.
    - return_output (bool, optional): If True, the function returns a concatenated string of all response chunks. The default is False.

    Returns:
    - str: If `return_output` is True, the function returns a concatenated string of all response chunks. Otherwise, it returns None.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


# Callback function that is executed when a tool is called.
def tool_callback(tool) -> None:
    print("[Tool Call]")
    print(f"Tool: {tool.get('tool')}")  # Print the name of the tool used.
    if tool_input := tool.get("tool_input"):  # If there is an input value for the tool
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # Print the key and value of the input value.
    print(f"Log: {tool.get('log')}")  # Print the log of the tool execution.


# Callback function that prints the observation.
def observation_callback(observation) -> None:
    print("[Observation]")
    print(f"Observation: {observation.get('observation')}")  # Print the observation.


# Callback function that prints the result.
def result_callback(result: str) -> None:
    print("[Result]")
    print(result)  # Print the result.


@dataclass
class AgentCallbacks:
    """
    A data class that contains the agent callback functions.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): Callback function called when a tool is used
        observation_callback (Callable[[Dict[str, Any]], None]): Callback function called when an observation is made
        result_callback (Callable[[str], None]): Callback function called when the result is returned
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    A class that parses and processes the stream output of an agent.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """
        Initialize the AgentStreamParser object.

        Args:
            callbacks (AgentCallbacks, optional): Callback functions used during the parsing process. Defaults to AgentCallbacks().
        """
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        Process the agent steps.

        Args:
            step (Dict[str, Any]): The agent step information to be processed
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """
        Process the agent actions.

        Args:
            actions (List[Any]): The list of actions to be processed
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """
        Process the tool call.

        Args:
            action (Any): The tool call action to be processed
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """
        Process the observations.

        Args:
            observations (List[Any]): The list of observations to be processed
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["action"] = getattr(observation, "action", None)
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """
        Process the result.

        Args:
            result (str): The final result to be processed
        """
        self.callbacks.result_callback(result)
        self.output = result


def pretty_print_messages(messages: list[BaseMessage]):
    for message in messages:
        message.pretty_print()


# Predefined colors for each depth level (using ANSI escape codes)
depth_colors = {
    1: "\033[96m",  # Light cyan (visually appealing first layer)
    2: "\033[93m",  # Yellow (second layer)
    3: "\033[94m",  # Light green (third layer)
    4: "\033[95m",  # Purple (fourth layer)
    5: "\033[92m",  # Light blue (fifth layer)
    "default": "\033[96m",  # Default is light cyan
    "reset": "\033[0m",  # Reset to default color
}


def is_terminal_dict(data):
    """Check if the data is a terminal dictionary."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True


def format_terminal_dict(data):
    """Format the terminal dictionary."""
    items = []
    for key, value in data.items():
        if isinstance(value, str):
            items.append(f'"{key}": "{value}"')
        else:
            items.append(f'"{key}": {value}')
    return "{" + ", ".join(items) + "}"


def _display_message_tree(data, indent=0, node=None, is_root=False):
    """
    Print the tree structure of a JSON object without type information.
    """
    spacing = " " * indent * 4
    color = depth_colors.get(indent + 1, depth_colors["default"])

    if isinstance(data, dict):
        if not is_root and node is not None:
            if is_terminal_dict(data):
                print(
                    f'{spacing}{color}{node}{depth_colors["reset"]}: {format_terminal_dict(data)}'
                )
            else:
                print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        if not is_root and node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')

        for index, item in enumerate(data):
            print(f'{spacing}    {color}index [{index}]{depth_colors["reset"]}')
            _display_message_tree(item, indent + 1)

    elif hasattr(data, "__dict__") and not is_root:
        if node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
        _display_message_tree(data.__dict__, indent)

    else:
        if node is not None:
            if isinstance(data, str):
                value_str = f'"{data}"'
            else:
                value_str = str(data)

            print(f'{spacing}{color}{node}{depth_colors["reset"]}: {value_str}')


def display_message_tree(message):
    """
    The main function to display the message tree.
    """
    if isinstance(message, BaseMessage):
        _display_message_tree(message.__dict__, is_root=True)
    else:
        _display_message_tree(message, is_root=True)
