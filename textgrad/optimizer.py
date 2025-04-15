from typing import Any, Dict, List, Optional, Callable, Union, Literal
from .module import TextParameter, LLMConfig
import torch
import torch.optim as optim

class TextOptimizer:
    """Base class for text optimizers."""
    
    def __init__(
        self,
        parameters: List[TextParameter],
        llm_config: Optional[LLMConfig] = None,
        learning_rate: float = 1.0,
        custom_update: Optional[Callable[[str, str], str]] = None,
        use_chat: bool = False
    ):
        self.parameters = parameters
        self.llm_config = llm_config or LLMConfig()
        self.learning_rate = learning_rate
        self.custom_update = custom_update
        self.use_chat = use_chat
        
    def step(self) -> None:
        """Update parameters based on accumulated gradients."""
        raise NotImplementedError
        
    def zero_grad(self) -> None:
        """Clear gradients of all parameters."""
        for param in self.parameters:
            param.grad = None
            
    def _update_parameter(
        self,
        param: TextParameter,
        feedback: str
    ) -> str:
        """Update a single parameter based on feedback."""
        if self.custom_update:
            return self.custom_update(param.value, feedback)
            
        # Format update prompt
        update_prompt = f"""
        Current value: {param.value}
        
        Feedback for improvement: {feedback}
        
        Please provide an improved version of the current value that addresses the feedback.
        """
        
        # Make LLM call
        client = self.llm_config.get_client()
        
        if self.use_chat:
            messages = [
                {"role": "system", "content": "You are a prompt optimization assistant."},
                {"role": "user", "content": update_prompt}
            ]
            new_value = client.chat(
                messages=messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
        else:
            new_value = client.complete(
                system_prompt="You are a prompt optimization assistant.",
                user_prompt=update_prompt,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
            
        return new_value

class TextSGD(TextOptimizer):
    """Stochastic Gradient Descent optimizer for text parameters."""
    
    def __init__(
        self,
        parameters: List[TextParameter],
        llm_config: Optional[LLMConfig] = None,
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        custom_update: Optional[Callable[[str, str], str]] = None,
        use_chat: bool = False
    ):
        super().__init__(parameters, llm_config, learning_rate, custom_update, use_chat)
        self.momentum = momentum
        self.velocity: Dict[str, str] = {}
        
    def step(self) -> None:
        """Update parameters based on accumulated gradients."""
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                # Get current velocity
                velocity = self.velocity.get(param.name, "")
                
                # Update velocity with momentum
                if self.momentum > 0 and velocity:
                    combined_feedback = f"""
                    Previous update direction: {velocity}
                    
                    New feedback: {param.grad}
                    
                    Please combine these to provide a consistent update direction.
                    """
                    velocity = self._update_parameter(param, combined_feedback)
                    self.velocity[param.name] = velocity
                    feedback = velocity
                else:
                    feedback = param.grad
                    
                # Update parameter
                new_value = self._update_parameter(param, feedback)
                param.value = new_value
                param.grad = None

class TextAdam(TextOptimizer):
    """Adam optimizer for text parameters."""
    
    def __init__(
        self,
        parameters: List[TextParameter],
        llm_config: Optional[LLMConfig] = None,
        learning_rate: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        custom_update: Optional[Callable[[str, str], str]] = None,
        use_chat: bool = False
    ):
        super().__init__(parameters, llm_config, learning_rate, custom_update, use_chat)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, str] = {}  # First moment
        self.v: Dict[str, str] = {}  # Second moment
        self.t = 0  # Time step
        
    def step(self) -> None:
        """Update parameters based on accumulated gradients."""
        self.t += 1
        
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                # Get moments
                m = self.m.get(param.name, "")
                v = self.v.get(param.name, "")
                
                # Update moments
                if m and v:
                    # Update first moment
                    m_feedback = f"""
                    Previous first moment: {m}
                    
                    New gradient: {param.grad}
                    
                    Please combine these with beta1={self.beta1} to update the first moment.
                    """
                    m = self._update_parameter(param, m_feedback)
                    self.m[param.name] = m
                    
                    # Update second moment
                    v_feedback = f"""
                    Previous second moment: {v}
                    
                    New gradient: {param.grad}
                    
                    Please combine these with beta2={self.beta2} to update the second moment.
                    """
                    v = self._update_parameter(param, v_feedback)
                    self.v[param.name] = v
                    
                    # Compute bias-corrected moments
                    m_hat = m  # In practice, this would be adjusted by 1/(1-beta1^t)
                    v_hat = v  # In practice, this would be adjusted by 1/(1-beta2^t)
                    
                    # Update parameter
                    update_feedback = f"""
                    Current value: {param.value}
                    
                    First moment: {m_hat}
                    Second moment: {v_hat}
                    
                    Please update the value using these moments with learning_rate={self.learning_rate} and epsilon={self.epsilon}.
                    """
                    new_value = self._update_parameter(param, update_feedback)
                else:
                    # Initialize moments
                    self.m[param.name] = param.grad
                    self.v[param.name] = param.grad
                    new_value = self._update_parameter(param, param.grad)
                    
                param.value = new_value
                param.grad = None

class TextSGD(optim.Optimizer):
    """
    An optimizer that can handle both PyTorch parameters and text parameters.
    For PyTorch parameters, it uses traditional SGD.
    For text parameters, it uses LLM-based optimization.
    """
    
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        llm_config: Optional[LLMConfig] = None,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        super().__init__(params, defaults)
        
        self.llm_config = llm_config or LLMConfig()
        self._text_params = [p for p in params if isinstance(p, TextParameter)]
        self._torch_params = [p for p in params if not isinstance(p, TextParameter)]
        
    def step(self, feedback: Optional[str] = None) -> None:
        """Perform one optimization step."""
        # Step PyTorch parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                    
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])
                        
                    if group['nesterov']:
                        d_p = d_p.add(buf, alpha=group['momentum'])
                    else:
                        d_p = buf
                        
                p.add_(d_p, alpha=-group['lr'])
                
        # Step text parameters if feedback is provided
        if feedback is not None:
            for param in self._text_params:
                if isinstance(param, TextParameter):
                    # Use LLM to generate parameter update
                    client = self.llm_config.get_client()
                    update = client.complete(
                        system_prompt="You are a parameter optimizer.",
                        user_prompt=f"""
                        Given the following feedback about the model's performance:
                        
                        {feedback}
                        
                        And the current parameter value:
                        
                        {param.text_value}
                        
                        Please suggest an improved value for this parameter.
                        """,
                        model=self.llm_config.model,
                        temperature=self.llm_config.temperature,
                        max_tokens=self.llm_config.max_tokens
                    )
                    
                    # Update parameter
                    param.text_value = update.strip()
                    
    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict of the optimizer."""
        state = super().state_dict()
        state['llm_config'] = self.llm_config.__dict__
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dict of the optimizer."""
        llm_config_dict = state_dict.pop('llm_config', None)
        if llm_config_dict is not None:
            self.llm_config = LLMConfig(**llm_config_dict)
        super().load_state_dict(state_dict) 