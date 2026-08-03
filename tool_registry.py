# tool_registry.py
from typing import Callable

_TOOLS: dict = {}

def tool(name: str = None, description: str = "", params: dict = None):
    """
    Decorator — registers a function as an AI-callable tool, self-describing
    from the code itself. params is OpenAI-function-calling style, e.g.:
        {"url": {"type": "string", "description": "...", "required": True}}
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.name
        _TOOLS[tool_name] = {
            "name": tool_name,
            "description": description or (fn.doc or "").strip().split("\n")[0],
            "params": params or {},
            "fn": fn,
        }
        return fn
    return decorator


def get_available_tools() -> list:
    """What the AI sees when picking WHICH tool to call."""
    return [{"name": t["name"], "description": t["description"]} for t in _TOOLS.values()]


def get_tool_params(tool_name: str) -> dict:
    """What the AI sees when deciding HOW to call a chosen tool."""
    t = _TOOLS.get(tool_name)
    return t["params"] if t else {}


def call_tool(tool_name: str, **kwargs):
    t = _TOOLS.get(tool_name)
    if not t:
        raise ValueError(f"Unknown tool: {tool_name}")
    return t["fn"](**kwargs)