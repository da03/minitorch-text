import pytest
import torch
import torch.nn as nn
from textgrad.hybrid_module import HybridModule
from textgrad.hybrid_optimizer import HybridOptimizer
from textgrad.llm_module import LLMModule
from textgrad.module import LLMConfig

class ImageFeatureExtractor(nn.Module):
    """A simple CNN for image feature extraction."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 128)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

def test_hybrid_image_captioning():
    """
    Test a hybrid model that combines:
    1. A CNN for image feature extraction (PyTorch)
    2. An LLM for generating captions (TextGrad)
    """
    # Create hybrid model
    model = HybridModule()
    
    # Add PyTorch module for image processing
    image_processor = ImageFeatureExtractor()
    model.add_torch_module('image_processor', image_processor)
    
    # Add LLM module for caption generation
    caption_generator = LLMModule(
        system_prompt="You are a helpful assistant that generates image captions.",
        user_prompt_template="{instruction}\n\nImage features: {features}",
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.0)
    )
    model.add_llm_module('caption_generator', caption_generator)
    
    # Create hybrid optimizer
    optimizer = HybridOptimizer.from_module(
        model,
        torch_optimizer_class=torch.optim.Adam,
        torch_optimizer_kwargs={'lr': 0.001}
    )
    
    # Training loop
    num_epochs = 5
    batch_size = 4
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Generate random image batch (32x32 RGB images)
        images = torch.randn(batch_size, 3, 32, 32)
        target_captions = [
            "A beautiful sunset over mountains",
            "A cat playing with a ball",
            "A busy city street at night",
            "A peaceful forest scene"
        ]
        
        # Forward pass
        # 1. Extract image features using CNN
        features = model.image_processor(images)
        
        # 2. Generate captions using LLM
        outputs = model.caption_generator(features=features.tolist())
        
        # Compute loss and get feedback
        # For CNN: Use traditional loss
        # For LLM: Use text-based feedback
        feedback = "The captions should be more descriptive and include specific details about colors, actions, and emotions."
        
        # Backward pass
        # 1. CNN backward pass (automatic through autograd)
        # 2. LLM backward pass
        param_feedback = model.backward(feedback)
        
        # Optimizer step
        optimizer.step(llm_feedback=feedback)
        
        # Print current state
        print("\nCurrent captions:")
        for i, caption in enumerate(outputs):
            print(f"Image {i+1}: {caption}")
            
        print("\nCurrent parameters:")
        for name, param in model.named_parameters():
            if isinstance(param, torch.Tensor):
                print(f"{name}: shape={param.shape}")
            else:
                print(f"{name}: {param.value}")
    
    # Save model state
    model.save("hybrid_model_state.json")
    
    # Test on new images
    test_images = torch.randn(2, 3, 32, 32)
    test_features = model.image_processor(test_images)
    test_outputs = model.caption_generator(features=test_features.tolist())
    
    print("\nTest captions:")
    for i, caption in enumerate(test_outputs):
        print(f"Test image {i+1}: {caption}")

if __name__ == "__main__":
    test_hybrid_image_captioning() 