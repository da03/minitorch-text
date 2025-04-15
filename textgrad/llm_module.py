from typing import Any, Dict, Optional, Union, List
import torch
from textgrad.module import TextModule, TextParameter, LLMConfig, ConversationTurn
from textgrad.llm_client import LLMClient

class LLMModule(TextModule):
    """
    A module that uses an LLM for text processing.
    Can be used in a PyTorch model alongside traditional layers.
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_chat: bool = False
    ):
        super().__init__()
        self.llm_config = llm_config or LLMConfig()
        self.use_chat = use_chat
        
        # Add text parameters
        self.system_prompt = self.add_text_parameter(
            "system_prompt",
            system_prompt or "You are a helpful assistant."
        )
        self.user_prompt = self.add_text_parameter(
            "user_prompt",
            user_prompt or "{input}"
        )
        
    def forward(
        self,
        input: Union[str, List[str]],
        history: Optional[List[ConversationTurn]] = None
    ) -> Union[str, List[str]]:
        """Forward pass through the module."""
        if isinstance(input, str):
            return self._forward_single(input, history)
        else:
            return [self._forward_single(i, history) for i in input]
            
    def _forward_single(
        self,
        input: str,
        history: Optional[List[ConversationTurn]] = None
    ) -> str:
        """Forward pass for a single input."""
        # Format user prompt
        formatted_prompt = self.user_prompt.value.format(input=input)
        
        # Add to conversation history
        if history is None:
            history = []
            
        # Make LLM call
        client = self.llm_config.get_client()
        
        if self.use_chat:
            # Prepare messages for chat completion
            messages = [{"role": "system", "content": self.system_prompt.value}]
            for turn in history:
                messages.append({"role": turn.role, "content": turn.content})
            messages.append({"role": "user", "content": formatted_prompt})
            
            response = client.chat(
                messages=messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
        else:
            # Use completion API
            response = client.complete(
                system_prompt=self.system_prompt.value,
                user_prompt=formatted_prompt,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
        
        return response
    
    def _generate_parameter_feedback(
        self,
        param: TextParameter,
        feedback: Union[str, List[str]],
        history: List[ConversationTurn]
    ) -> str:
        """Generate feedback for a parameter."""
        if isinstance(feedback, list):
            feedback = "\n".join(feedback)
            
        if param.name == "system_prompt":
            return f"""
            Based on the following conversation history and feedback:
            
            Conversation History:
            {self._format_history(history)}
            
            Final Feedback:
            {feedback}
            
            How should the system prompt be improved to better guide the assistant's responses?
            """
        elif param.name == "user_prompt":
            return f"""
            Based on the following conversation history and feedback:
            
            Conversation History:
            {self._format_history(history)}
            
            Final Feedback:
            {feedback}
            
            How should the user prompt template be improved to better format the input?
            """
        else:
            return feedback
            
    def _format_history(self, history: List[ConversationTurn]) -> str:
        """Format conversation history for feedback generation."""
        formatted = []
        for turn in history:
            formatted.append(f"{turn.role}: {turn.content}")
        return "\n".join(formatted)

class LeaderModule(LLMModule):
    """A module that coordinates multiple worker modules."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_chat: bool = True
    ):
        super().__init__(
            system_prompt or """
            You are a leader coordinating a team of workers.
            Your task is to:
            1. Analyze worker responses
            2. Identify areas for improvement
            3. Provide specific feedback to workers
            4. Synthesize final results
            """,
            user_prompt or """
            Worker Responses:
            {input}
            
            Please analyze these responses and provide:
            1. A summary of key findings
            2. Specific feedback for each worker
            3. A final synthesized result
            """,
            llm_config,
            use_chat
        )
        self.workers = []
        
    def add_worker(self, worker: 'WorkerModule') -> None:
        """Add a worker module to be coordinated."""
        self.workers.append(worker)
        
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """Forward pass through leader and workers."""
        # Get leader's response
        leader_response = super().forward(*args, **kwargs)
        
        # Get responses from workers
        worker_responses = {}
        for i, worker in enumerate(self.workers):
            worker_response = worker(*args, **kwargs)
            worker_responses[f"worker_{i}"] = worker_response
            
        return {
            "leader": leader_response,
            "workers": worker_responses
        }
        
    def backward(self, feedback: str, turn_index: Optional[int] = None) -> Dict[str, str]:
        """Backward pass through leader and workers."""
        # Get parameter feedback for leader
        param_feedback = super().backward(feedback, turn_index)
        
        # Get parameter feedback for workers
        for i, worker in enumerate(self.workers):
            worker_feedback = worker.backward(feedback, turn_index)
            for name, fb in worker_feedback.items():
                param_feedback[f"worker_{i}.{name}"] = fb
                
        return param_feedback

class WorkerModule(LLMModule):
    """A module that performs a specific task under leader coordination."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_chat: bool = True
    ):
        super().__init__(
            system_prompt or """
            You are a worker in a team.
            Your task is to:
            1. Follow instructions carefully
            2. Provide detailed responses
            3. Ask for clarification when needed
            4. Collaborate with other workers
            """,
            user_prompt or """
            Task:
            {input}
            
            Please provide a detailed response to this task.
            """,
            llm_config,
            use_chat
        ) 