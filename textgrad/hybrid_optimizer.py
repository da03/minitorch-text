from typing import List, Dict, Any, Optional
import torch
import torch.optim as optim
from textgrad.optimizer import TextSGD
from textgrad.module import LLMConfig

class HybridOptimizer:
    """
    An optimizer that can handle both PyTorch parameters and LLM parameters.
    For PyTorch parameters, it uses traditional optimization methods.
    For LLM parameters, it uses LLM-based optimization.
    """
    
    def __init__(
        self,
        torch_optimizer: optim.Optimizer,
        llm_optimizer: TextSGD,
        llm_parameters: List[Any],
        torch_parameters: List[torch.nn.Parameter]
    ):
        self.torch_optimizer = torch_optimizer
        self.llm_optimizer = llm_optimizer
        self.llm_parameters = llm_parameters
        self.torch_parameters = torch_parameters
        
    @classmethod
    def from_module(
        cls,
        module: 'HybridModule',
        torch_optimizer_class: type = optim.Adam,
        torch_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        llm_config: Optional[LLMConfig] = None
    ) -> 'HybridOptimizer':
        """Create a hybrid optimizer from a hybrid module."""
        if torch_optimizer_kwargs is None:
            torch_optimizer_kwargs = {}
            
        # Get PyTorch parameters
        torch_params = []
        for name, param in module.named_parameters():
            if name.startswith('_torch_modules'):
                torch_params.append(param)
                
        # Get LLM parameters
        llm_params = []
        for name, param in module.named_parameters():
            if name.startswith('_llm_modules'):
                llm_params.append(param)
                
        # Create PyTorch optimizer
        torch_optimizer = torch_optimizer_class(torch_params, **torch_optimizer_kwargs)
        
        # Create LLM optimizer
        if llm_config is None:
            llm_config = LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
        llm_optimizer = TextSGD(llm_params, llm_config=llm_config)
        
        return cls(torch_optimizer, llm_optimizer, llm_params, torch_params)
    
    def step(self, llm_feedback: Optional[str] = None) -> None:
        """
        Perform one optimization step.
        For PyTorch parameters, this uses the traditional optimizer.
        For LLM parameters, this uses the LLM optimizer with feedback.
        """
        # Step PyTorch optimizer
        self.torch_optimizer.step()
        
        # Step LLM optimizer if feedback is provided
        if llm_feedback is not None:
            self.llm_optimizer.step(llm_feedback)
            
    def zero_grad(self) -> None:
        """Zero the gradients of all parameters."""
        self.torch_optimizer.zero_grad()
        
    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict of both optimizers."""
        return {
            'torch_optimizer': self.torch_optimizer.state_dict(),
            'llm_optimizer': self.llm_optimizer.state_dict()
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dict of both optimizers."""
        self.torch_optimizer.load_state_dict(state_dict['torch_optimizer'])
        self.llm_optimizer.load_state_dict(state_dict['llm_optimizer']) 