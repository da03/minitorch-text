from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable, Protocol, Union
import openai  # We'll need this for LLM calls
import json
from pathlib import Path
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class LLMClient(Protocol):
    """Protocol for LLM clients."""
    
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Complete a prompt using the LLM."""
        ...
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Chat with the LLM using a list of messages."""
        ...

@dataclass
class LLMConfig:
    """Configuration for LLM clients."""
    client: LLMClient
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    def get_client(self) -> LLMClient:
        """Get the LLM client."""
        return self.client

class TextParameter:
    """A parameter that can be optimized through text feedback."""
    
    def __init__(self, value: str, name: Optional[str] = None):
        self.value = value
        self.name = name
        self.grad = None
        self.requires_grad = True
        
    def __repr__(self) -> str:
        return f"TextParameter(name={self.name}, value={self.value})"
        
    def backward(self, feedback: str) -> None:
        """Update the parameter based on feedback."""
        self.grad = feedback
        
    def step(self, learning_rate: float = 1.0) -> None:
        """Update the parameter value based on accumulated gradient."""
        if self.grad is not None and self.requires_grad:
            # In a real implementation, this would use the LLM to update the prompt
            # For now, we'll just store the feedback
            self.value = self.grad
            self.grad = None

class TextModule(nn.Module):
    """Base class for text-based modules."""
    
    def __init__(self):
        super().__init__()
        self._parameters: Dict[str, TextParameter] = {}
        self._modules: Dict[str, TextModule] = {}
        self.training = True
        
    def add_text_parameter(self, name: str, value: str) -> TextParameter:
        """Add a text parameter to the module."""
        param = TextParameter(value, name)
        self._parameters[name] = param
        return param
        
    def add_module(self, name: str, module: 'TextModule') -> None:
        """Add a submodule to the module."""
        self._modules[name] = module
        
    def parameters(self, recurse: bool = True) -> List[TextParameter]:
        """Get all parameters in the module and its submodules."""
        params = list(self._parameters.values())
        if recurse:
            for module in self._modules.values():
                params.extend(module.parameters())
        return params
        
    def modules(self) -> List['TextModule']:
        """Get all submodules."""
        return list(self._modules.values())
        
    def forward(
        self,
        input: Union[str, List[str]],
        history: Optional[List[ConversationTurn]] = None
    ) -> Union[str, List[str]]:
        """Forward pass through the module."""
        raise NotImplementedError
        
    def __call__(
        self,
        input: Union[str, List[str]],
        history: Optional[List[ConversationTurn]] = None
    ) -> Union[str, List[str]]:
        """Make the module callable."""
        return self.forward(input, history)
        
    def backward(
        self,
        feedback: Union[str, List[str]],
        turn_index: Optional[int] = None
    ) -> Dict[str, str]:
        """Backward pass through the module."""
        param_feedback = {}
        for name, param in self._parameters.items():
            if param.requires_grad:
                param_feedback[name] = self._generate_parameter_feedback(
                    param, feedback, history or []
                )
        return param_feedback
        
    def _generate_parameter_feedback(
        self,
        param: TextParameter,
        feedback: Union[str, List[str]],
        history: List[ConversationTurn]
    ) -> str:
        """Generate feedback for a parameter."""
        raise NotImplementedError
        
    def train(self) -> None:
        """Set the module to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()
            
    def eval(self) -> None:
        """Set the module to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()
            
    def state_dict(self) -> Dict[str, Any]:
        """Get the state of the module and its submodules."""
        state = {
            "parameters": {
                name: param.value
                for name, param in self._parameters.items()
            },
            "modules": {
                name: module.state_dict()
                for name, module in self._modules.items()
            }
        }
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the module and its submodules."""
        for name, value in state_dict["parameters"].items():
            if name in self._parameters:
                self._parameters[name].value = value
                
        for name, module_state in state_dict["modules"].items():
            if name in self._modules:
                self._modules[name].load_state_dict(module_state)
                
    def save(self, path: Union[str, Path]) -> None:
        """Save the state of the module to a file."""
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)
            
    def load(self, path: Union[str, Path]) -> None:
        """Load the state of the module from a file."""
        with open(path, "r") as f:
            state_dict = json.load(f)
        self.load_state_dict(state_dict)

def _addindent(s_: str, numSpaces: int) -> str:
    """Helper function for indentation in repr."""
    s2 = s_.split("\n")
    if len(s2) == 1:
        return s_
    first = s2.pop(0)
    s2 = [(numSpaces * " ") + line for line in s2]
    s = "\n".join(s2)
    s = first + "\n" + s
    return s 