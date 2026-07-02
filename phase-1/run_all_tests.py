import subprocess
import os

scripts = [
    "direct_prompt_agent.py",
    "augmented_prompt_agent.py",
    "knowledge_augmented_prompt_agent.py",
    "rag_knowledge_prompt_agent.py",
    "evaluation_agent.py",
    "routing_agent.py",
    "action_planning_agent.py"
]

print("==================================================")
print("RUNNING ALL PHASE 1 TEST SCRIPTS FOR VERIFICATION")
print("==================================================\n")

for script in scripts:
    print(f"--- Running {script} ---")
    if not os.path.exists(script):
        print(f"Error: {script} does not exist!\n")
        continue
    try:
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}:")
        print(e.stderr)
    print("-" * 50 + "\n")

print("All test runs completed.")
