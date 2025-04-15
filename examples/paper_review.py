import openai
from textgrad.module import TextModule
from textgrad.llm_module import LeaderModule, WorkerModule
from textgrad.loss import TextLoss
from textgrad.optimizer import TextSGD

def main():
    # Initialize the OpenAI client
    client = openai.OpenAI()
    
    # Create the multi-agent system
    leader = LeaderModule(
        system_prompt="You are a leader agent coordinating a team of workers to write a comprehensive paper review. Your role is to synthesize the workers' analyses into a coherent review that covers all important aspects of the paper.",
        user_prompt_template="""Please synthesize the following worker analyses into a comprehensive paper review:
        
        {worker_outputs}
        
        Make sure to:
        1. Maintain a logical flow between sections
        2. Highlight the paper's key contributions
        3. Provide constructive criticism where appropriate
        4. Suggest potential improvements
        """,
        llm_client=client
    )
    
    # Create worker modules for different sections of the paper
    workers = [
        WorkerModule(
            system_prompt="You are a worker agent analyzing the introduction and motivation section of a paper. Focus on understanding the problem statement, motivation, and research goals.",
            user_prompt_template="Please analyze the following introduction section:\n\n{paper_section}",
            llm_client=client
        ),
        WorkerModule(
            system_prompt="You are a worker agent analyzing the methodology section of a paper. Focus on understanding the technical approach, algorithms, and experimental setup.",
            user_prompt_template="Please analyze the following methodology section:\n\n{paper_section}",
            llm_client=client
        ),
        WorkerModule(
            system_prompt="You are a worker agent analyzing the results and discussion section of a paper. Focus on understanding the findings, their significance, and limitations.",
            user_prompt_template="Please analyze the following results section:\n\n{paper_section}",
            llm_client=client
        )
    ]
    
    # Create the loss module
    loss = TextLoss(llm_client=client)
    
    # Create the optimizer
    optimizer = TextSGD(
        parameters=list(leader.parameters()) + [p for w in workers for p in w.parameters()],
        llm_client=client
    )
    
    # Example paper sections (in a real scenario, these would be the actual paper sections)
    paper_sections = [
        "Introduction section...",
        "Methodology section...",
        "Results section..."
    ]
    
    # Example human review (in a real scenario, this would be the actual human review)
    human_review = "This is an example human review of the paper..."
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Forward pass
        worker_outputs = []
        for worker, section in zip(workers, paper_sections):
            output = worker(paper_section=section)
            worker_outputs.append(output)
            
        # Combine worker outputs
        system_output = leader(worker_outputs="\n\n".join(worker_outputs))
        
        # Compute loss and get feedback
        feedback = loss(system_output, human_review)
        print("\nFeedback from loss module:")
        print(feedback)
        
        # Backward pass
        leader_feedback = leader.backward(feedback)
        worker_feedbacks = [w.backward(feedback) for w in workers]
        
        # Combine feedbacks
        all_feedback = {**leader_feedback}
        for i, w_feedback in enumerate(worker_feedbacks):
            for k, v in w_feedback.items():
                all_feedback[f"worker_{i}_{k}"] = v
                
        # Update parameters
        optimizer.step(all_feedback)
        
        print("\nUpdated prompts:")
        print(f"Leader system prompt: {leader.system_prompt.value}")
        print(f"Leader user prompt: {leader.user_prompt.value}")
        for i, worker in enumerate(workers):
            print(f"\nWorker {i} system prompt: {worker.system_prompt.value}")
            print(f"Worker {i} user prompt: {worker.user_prompt.value}")

if __name__ == "__main__":
    main() 