from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from textgrad.module import TextModule, LLMConfig
from textgrad.llm_module import LLMModule

class HybridModule(nn.Module):
    """
    A module that can contain both traditional PyTorch layers and LLM modules.
    This allows for hybrid models where some parts use traditional backpropagation
    and others use LLM-based optimization.
    """
    
    def __init__(self):
        super().__init__()
        self._llm_modules: Dict[str, LLMModule] = {}
        self._torch_modules: Dict[str, nn.Module] = {}
        
    def add_llm_module(self, name: str, module: LLMModule) -> None:
        """Add an LLM module to the hybrid model."""
        self._llm_modules[name] = module
        
    def add_torch_module(self, name: str, module: nn.Module) -> None:
        """Add a PyTorch module to the hybrid model."""
        self._torch_modules[name] = module
        self.add_module(name, module)
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the hybrid model.
        For LLM modules, this performs the LLM call.
        For PyTorch modules, this performs the normal forward pass.
        """
        # Process LLM modules first
        llm_outputs = {}
        for name, module in self._llm_modules.items():
            llm_outputs[name] = module(*args, **kwargs)
            
        # Process PyTorch modules
        torch_outputs = {}
        for name, module in self._torch_modules.items():
            torch_outputs[name] = module(*args, **kwargs)
            
        return {**llm_outputs, **torch_outputs}
    
    def backward(self, feedback: str) -> Dict[str, str]:
        """
        Backward pass through the hybrid model.
        For LLM modules, this generates parameter feedback.
        For PyTorch modules, this is a no-op as they use autograd.
        """
        param_feedback = {}
        for name, module in self._llm_modules.items():
            param_feedback[name] = module.backward(feedback)
        return param_feedback
    
    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict containing both PyTorch and LLM parameters."""
        state = super().state_dict()
        llm_state = {}
        for name, module in self._llm_modules.items():
            llm_state[name] = module.state_dict()
        return {**state, **llm_state}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict containing both PyTorch and LLM parameters."""
        # Split state dict into PyTorch and LLM parts
        torch_state = {}
        llm_state = {}
        for key, value in state_dict.items():
            if key in self._llm_modules:
                llm_state[key] = value
            else:
                torch_state[key] = value
                
        # Load PyTorch state
        super().load_state_dict(torch_state)
        
        # Load LLM state
        for name, state in llm_state.items():
            self._llm_modules[name].load_state_dict(state)
            
    def parameters(self, recurse: bool = True) -> List[torch.nn.Parameter]:
        """Get all parameters, including both PyTorch and LLM parameters."""
        params = list(super().parameters(recurse))
        for module in self._llm_modules.values():
            params.extend(module.parameters())
        return params
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> List[tuple[str, torch.nn.Parameter]]:
        """Get all named parameters, including both PyTorch and LLM parameters."""
        named_params = list(super().named_parameters(prefix, recurse))
        for name, module in self._llm_modules.items():
            named_params.extend([(f"{prefix}{name}.{pname}", p) 
                               for pname, p in module.named_parameters()])
        return named_params 