# Test script for ActionPlanningAgent class
from workflow_agents.base_agents import ActionPlanningAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "One morning I wanted to have scrambled eggs"

# Instantiate the ActionPlanningAgent
action_agent = ActionPlanningAgent(openai_api_key=openai_api_key)

# Get the action plan steps
steps = action_agent.respond(prompt)

# Print the steps
print("=== Extracted Action Steps ===")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step}")
