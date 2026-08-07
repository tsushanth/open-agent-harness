from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class Tool(ABC):
    """Base class for a harness tool. Subclasses define an OpenAI-style function
    schema and an execute() method. Kept minimal on purpose — this mirrors the
    shape of tool-call trajectories we'll later use for SFT, so the schema here
    IS the training data format, not just a runtime detail."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    requires_confirmation: bool = False  # gated by the agent loop before execute() runs

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
