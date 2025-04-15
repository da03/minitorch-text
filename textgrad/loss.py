from typing import Any, Dict, Optional, Callable, Union, List, Literal
import torch
import torch.nn as nn
from textgrad.module import LLMConfig, TextModule, ConversationTurn, TextParameter

class TextLoss(TextModule):
    """
    A loss module that can handle both PyTorch and text outputs.
    For PyTorch outputs, it uses traditional loss functions.
    For text outputs, it uses LLM-based comparison.
    """
    
    def __init__(
        self,
        torch_loss_fn: Optional[Callable] = None,
        text_comparison_fn: Optional[Callable] = None,
        llm_config: Optional[LLMConfig] = None,
        comparison_prompt: Optional[str] = None,
        custom_comparison: Optional[Callable[[str, str], str]] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        use_chat: bool = False
    ):
        super().__init__()
        self.torch_loss_fn = torch_loss_fn or nn.MSELoss()
        self.text_comparison_fn = text_comparison_fn
        self.llm_config = llm_config or LLMConfig()
        self.custom_comparison = custom_comparison
        self.reduction = reduction
        self.use_chat = use_chat
        
        # Add text parameter for comparison prompt
        self.comparison_prompt = self.add_text_parameter(
            "comparison_prompt",
            comparison_prompt or """
            Compare the system output with the target output and provide detailed feedback on:
            1. What aspects of the system output match the target output
            2. What aspects need improvement
            3. Specific suggestions for how to improve the system output to better match the target output
            4. Any patterns or strategies that could help improve future outputs
            """
        )
        
    def forward(
        self,
        system_output: Union[str, List[str]],
        target_output: Union[str, List[str]],
        history: Optional[List[ConversationTurn]] = None
    ) -> Union[str, List[str]]:
        """Compute loss between system output and target output."""
        if isinstance(system_output, str):
            return self._forward_single(system_output, target_output, history)
        else:
            feedbacks = [
                self._forward_single(sys_out, tgt_out, history)
                for sys_out, tgt_out in zip(system_output, target_output)
            ]
            
            if self.reduction == "mean":
                return "\n".join(feedbacks)
            elif self.reduction == "sum":
                return "\n".join(feedbacks)
            else:  # "none"
                return feedbacks
                
    def _forward_single(
        self,
        system_output: str,
        target_output: str,
        history: Optional[List[ConversationTurn]] = None
    ) -> str:
        """Compute loss for a single output pair."""
        if self.custom_comparison:
            feedback = self.custom_comparison(system_output, target_output)
        else:
            # Format comparison prompt
            formatted_prompt = self.comparison_prompt.value.format(
                system_output=system_output,
                target_output=target_output
            )
            
            # Add to conversation history
            if history is None:
                history = []
            history.append(ConversationTurn("system", self.comparison_prompt.value))
            history.append(ConversationTurn("user", formatted_prompt))
            
            # Make LLM call
            client = self.llm_config.get_client()
            
            if self.use_chat:
                # Prepare messages for chat completion
                messages = [{"role": "system", "content": self.comparison_prompt.value}]
                for turn in history:
                    messages.append({"role": turn.role, "content": turn.content})
                messages.append({"role": "user", "content": formatted_prompt})
                
                feedback = client.chat(
                    messages=messages,
                    model=self.llm_config.model,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
            else:
                # Use completion API
                feedback = client.complete(
                    system_prompt=self.comparison_prompt.value,
                    user_prompt=formatted_prompt,
                    model=self.llm_config.model,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
            
            # Add response to history
            history.append(ConversationTurn("assistant", feedback))
            
        return feedback
        
    def backward(
        self,
        feedback: Union[str, List[str]],
        turn_index: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate feedback for the comparison prompt."""
        return super().backward(feedback, turn_index)
        
    def _generate_parameter_feedback(
        self,
        param: TextParameter,
        feedback: Union[str, List[str]],
        history: List[ConversationTurn]
    ) -> str:
        """Generate feedback for the comparison prompt parameter."""
        if isinstance(feedback, list):
            feedback = "\n".join(feedback)
            
        if param.name == "comparison_prompt":
            return f"""
            Based on the following conversation history and feedback:
            
            Conversation History:
            {self._format_history(history)}
            
            Final Feedback:
            {feedback}
            
            How should the comparison prompt be improved to better evaluate the system output against the target output?
            """
        else:
            return feedback
            
    def _format_history(self, history: List[ConversationTurn]) -> str:
        """Format conversation history for feedback generation."""
        formatted = []
        for turn in history:
            formatted.append(f"{turn.role}: {turn.content}")
        return "\n".join(formatted)
        
    def _handle_dict_output(self, output: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str]]:
        """Handle dictionary outputs (e.g., from hybrid models)"""
        losses = {}
        for key, out in output.items():
            tgt = target.get(key) if isinstance(target, dict) else target
            losses[key] = self.forward(out, tgt)
        return losses
        
    def _handle_pytorch_output(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Handle PyTorch tensor outputs"""
        if not isinstance(target, torch.Tensor):
            raise ValueError("Target must be a tensor when output is a tensor")
        return self.torch_loss_fn(output, target)
        
    def _handle_text_output(self, output: str, target: str) -> str:
        """Handle text outputs"""
        if self.text_comparison_fn is not None:
            return self.text_comparison_fn(output, target)
            
        # Default text comparison using LLM
        client = self.llm_config.get_client()
        feedback = client.complete(
            system_prompt="You are a text comparison expert.",
            user_prompt=f"""
            Compare the following two texts:
            
            Output:
            {output}
            
            Target:
            {target}
            
            Provide detailed feedback on how the output could be improved to better match the target.
            Focus on specific aspects that need improvement.
            """,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        return feedback
        
    def _handle_output(self, output: Union[torch.Tensor, str, Dict[str, Any]], target: Union[torch.Tensor, str, Dict[str, Any]]) -> Union[torch.Tensor, str]:
        """Handle different types of outputs"""
        if isinstance(output, dict):
            return self._handle_dict_output(output, target)
        elif isinstance(output, torch.Tensor):
            return self._handle_pytorch_output(output, target)
        elif isinstance(output, str):
            return self._handle_text_output(output, target)
        else:
            raise ValueError(f"Unsupported output type: {type(output)}")
        
    def backward(self, loss: Union[torch.Tensor, str]) -> str:
        """
        Generate feedback for optimization.
        For PyTorch losses, converts to text feedback.
        For text feedback, returns as is.
        """
        if isinstance(loss, torch.Tensor):
            # Convert tensor loss to text feedback
            client = self.llm_config.get_client()
            feedback = client.complete(
                system_prompt="You are a loss interpreter.",
                user_prompt=f"""
                The model's loss value is {loss.item():.4f}.
                Please provide feedback on how to improve the model's performance.
                Focus on specific aspects that need improvement.
                """,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
            return feedback
            
        return loss 