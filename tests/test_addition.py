import pytest
import random
from textgrad.module import LLMConfig
from textgrad.llm_module import LLMModule
from textgrad.loss import TextLoss
from textgrad.optimizer import TextSGD

def generate_addition_data(num_examples: int, min_numbers: int = 2, max_numbers: int = 5, min_value: int = 1, max_value: int = 100):
    """Generate random addition problems."""
    data = []
    for _ in range(num_examples):
        # Random number of numbers to add
        n = random.randint(min_numbers, max_numbers)
        # Generate random numbers
        numbers = [random.randint(min_value, max_value) for _ in range(n)]
        # Calculate sum
        target_sum = sum(numbers)
        data.append((numbers, target_sum))
    return data

def test_addition_learning():
    """
    Test if the system can discover the addition task through optimization.
    The system is not told what operation to perform - it must discover it
    through the feedback on its outputs.
    """
    # Create a module with a neutral system prompt that doesn't hint at the task
    module = LLMModule(
        system_prompt="You are a helpful assistant that processes numbers.",
        user_prompt_template="{instruction}\n\nNumbers: {numbers}",
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
    )
    
    # Create a loss function that only provides feedback on correctness
    def addition_comparison(system_output: str, target_output: str) -> str:
        try:
            # Try to extract a number from the system output
            words = system_output.strip().split()
            system_sum = None
            for word in reversed(words):
                try:
                    system_sum = float(word)
                    break
                except ValueError:
                    continue
            
            if system_sum is None:
                return "I couldn't find a number in your response. Please provide a numerical answer."
            
            target_sum = float(target_output)
            
            if abs(system_sum - target_sum) < 0.001:
                return "Correct! That's the right answer."
            else:
                return f"That's not correct. The right answer is {target_sum}."
        except (ValueError, IndexError):
            return "I couldn't understand your response. Please provide a numerical answer."
    
    loss = TextLoss(custom_comparison=addition_comparison)
    
    # Create optimizer
    optimizer = TextSGD(
        parameters=list(module.parameters()),
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
    )
    
    # Initial instruction (completely neutral)
    initial_instruction = "Please process these numbers."
    module.user_prompt.value = initial_instruction
    
    # Track accuracy and instructions over time
    accuracies = []
    instructions = []
    
    # Training parameters
    num_epochs = 10
    batch_size = 3
    train_size = 15  # Total training examples per epoch
    test_size = 5    # Test examples per evaluation
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Current instruction: {module.user_prompt.value}")
        instructions.append(module.user_prompt.value)
        
        # Generate new training data for this epoch
        training_data = generate_addition_data(train_size)
        
        correct = 0
        total = len(training_data)
        
        # Process in batches
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            batch_feedback = []
            
            # Forward pass for each example in batch
            for numbers, target_sum in batch:
                numbers_str = ", ".join(map(str, numbers))
                output = module(numbers=numbers_str)
                
                # Compute loss and get feedback
                feedback = loss(output, str(target_sum))
                
                # Check if correct
                if "Correct!" in feedback:
                    correct += 1
                
                batch_feedback.append(feedback)
            
            # Backward pass with batched feedback
            param_feedback = module.backward("\n".join(batch_feedback))
            
            # Update parameters
            optimizer.step(param_feedback)
            
            # Print updated parameters
            print(f"\nAfter batch {i//batch_size + 1}:")
            for name, param in module.named_parameters():
                print(f"{name}: {param.value}")
        
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"\nEpoch accuracy: {accuracy:.2%}")
        
        # Save state after each epoch
        module.save(f"addition_module_epoch_{epoch}.json")
        
        # Evaluate on new test data
        test_data = generate_addition_data(test_size)
        test_correct = 0
        for numbers, target_sum in test_data:
            numbers_str = ", ".join(map(str, numbers))
            output = module(numbers=numbers_str)
            feedback = loss(output, str(target_sum))
            if "Correct!" in feedback:
                test_correct += 1
        
        test_accuracy = test_correct / len(test_data)
        print(f"Test accuracy: {test_accuracy:.2%}")
    
    # Verify that accuracy improved
    assert accuracies[-1] >= accuracies[0], "Accuracy should improve over time"
    assert accuracies[-1] > 0.5, "Final accuracy should be reasonable"
    
    # Verify that the learned instruction discovered the task
    final_instruction = module.user_prompt.value
    print(f"\nFinal learned instruction: {final_instruction}")
    
    # Check if the instruction evolved to mention addition
    print("\nInstruction evolution:")
    for i, instr in enumerate(instructions):
        print(f"Epoch {i+1}: {instr}")
    
    # The final instruction should have discovered the addition task
    assert any(word in final_instruction.lower() for word in ["add", "sum", "total", "+", "plus"]), \
        "Learned instruction should have discovered the addition task"

if __name__ == "__main__":
    test_addition_learning() 