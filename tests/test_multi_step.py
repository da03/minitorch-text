import pytest
from textgrad.module import LLMConfig
from textgrad.llm_module import LLMModule
from textgrad.loss import TextLoss
from textgrad.optimizer import TextSGD

def test_multi_step_learning():
    """
    Test learning a prompt for a multi-step task: given a list of numbers,
    find the sum of even numbers and the product of odd numbers.
    """
    # Create a module for the multi-step task
    module = LLMModule(
        system_prompt="You are a helpful assistant that performs mathematical operations on lists of numbers.",
        user_prompt_template="{instruction}\n\nNumbers: {numbers}",
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
    )
    
    # Create a loss function that checks both the sum of evens and product of odds
    def multi_step_comparison(system_output: str, target_output: str) -> str:
        try:
            # Parse target values
            target_even_sum, target_odd_product = map(float, target_output.split(","))
            
            # Try to extract values from system output
            lines = system_output.strip().split("\n")
            system_even_sum = float(lines[0].split(":")[1].strip())
            system_odd_product = float(lines[1].split(":")[1].strip())
            
            feedback = []
            if abs(system_even_sum - target_even_sum) < 0.001:
                feedback.append("Even sum is correct.")
            else:
                feedback.append(f"Even sum is incorrect. Expected {target_even_sum}, got {system_even_sum}.")
                
            if abs(system_odd_product - target_odd_product) < 0.001:
                feedback.append("Odd product is correct.")
            else:
                feedback.append(f"Odd product is incorrect. Expected {target_odd_product}, got {system_odd_product}.")
                
            return " ".join(feedback)
        except (ValueError, IndexError):
            return "Invalid output format. Please output two lines: 'Sum of evens: X' and 'Product of odds: Y'"
    
    loss = TextLoss(custom_comparison=multi_step_comparison)
    
    # Create optimizer
    optimizer = TextSGD(
        parameters=list(module.parameters()),
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
    )
    
    # Training data: (numbers, (even_sum, odd_product))
    training_data = [
        ([1, 2, 3, 4], (6, 3)),      # evens: 2+4=6, odds: 1*3=3
        ([5, 6, 7, 8], (14, 35)),    # evens: 6+8=14, odds: 5*7=35
        ([9, 10, 11, 12], (22, 99)), # evens: 10+12=22, odds: 9*11=99
        ([13, 14, 15, 16], (30, 195)) # evens: 14+16=30, odds: 13*15=195
    ]
    
    # Initial instruction
    initial_instruction = "Please process these numbers."
    module.user_prompt.value = initial_instruction
    
    # Track accuracy over time
    accuracies = []
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Current instruction: {module.user_prompt.value}")
        
        correct = 0
        total = len(training_data)
        
        for numbers, (even_sum, odd_product) in training_data:
            # Forward pass
            numbers_str = ", ".join(map(str, numbers))
            output = module(numbers=numbers_str)
            
            # Compute loss and get feedback
            target_str = f"{even_sum},{odd_product}"
            feedback = loss(output, target_str)
            
            # Check if both parts are correct
            if "Even sum is correct" in feedback and "Odd product is correct" in feedback:
                correct += 1
            
            # Backward pass
            param_feedback = module.backward(feedback)
            
            # Update parameters
            optimizer.step(param_feedback)
        
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.2%}")
        
        # Save state after each epoch
        module.save(f"multi_step_module_epoch_{epoch}.json")
    
    # Verify that accuracy improved
    assert accuracies[-1] >= accuracies[0], "Accuracy should improve over time"
    assert accuracies[-1] > 0.5, "Final accuracy should be reasonable"
    
    # Verify that the learned instruction makes sense
    final_instruction = module.user_prompt.value
    print(f"\nFinal learned instruction: {final_instruction}")
    assert any(word in final_instruction.lower() for word in ["even", "odd", "sum", "product"]), \
        "Learned instruction should mention the task components"
    
    # Test the final model on new examples
    test_data = [
        ([17, 18, 19, 20], (38, 323)),  # evens: 18+20=38, odds: 17*19=323
        ([21, 22, 23, 24], (46, 483))   # evens: 22+24=46, odds: 21*23=483
    ]
    
    correct = 0
    for numbers, (even_sum, odd_product) in test_data:
        numbers_str = ", ".join(map(str, numbers))
        output = module(numbers=numbers_str)
        target_str = f"{even_sum},{odd_product}"
        feedback = loss(output, target_str)
        if "Even sum is correct" in feedback and "Odd product is correct" in feedback:
            correct += 1
    
    test_accuracy = correct / len(test_data)
    print(f"\nTest accuracy: {test_accuracy:.2%}")
    assert test_accuracy > 0.5, "Test accuracy should be reasonable"

if __name__ == "__main__":
    test_multi_step_learning() 