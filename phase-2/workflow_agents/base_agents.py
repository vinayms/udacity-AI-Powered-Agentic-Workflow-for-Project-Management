import math
import json
import logging
from openai import OpenAI

logger = logging.getLogger("workflow_agents")

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def create_client(openai_api_key):
    base_url = "https://openai.vocareum.com/v1" if openai_api_key and openai_api_key.startswith("voc-") else None
    logger.debug(f"Creating OpenAI client (base_url: {base_url or 'default'})")
    return OpenAI(api_key=openai_api_key, base_url=base_url)


class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = create_client(self.openai_api_key)

    def respond(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Sending direct prompt to LLM...")
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.client = create_client(self.openai_api_key)

    def respond(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Sending augmented prompt with persona: {self.persona}")
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.client = create_client(self.openai_api_key)

    def respond(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Sending prompt with persona and specific knowledge context...")
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    def __init__(self, openai_api_key, persona, chunk_size=500, chunk_overlap=200):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = create_client(self.openai_api_key)
        self.chunks = []
        self.embeddings = []

    def chunk_text(self, text):
        logger.info(f"[{self.__class__.__name__}] Chunking knowledge text (size: {self.chunk_size}, overlap: {self.chunk_overlap})...")
        chunks = []
        start = 0
        if not text:
            self.chunks = []
            return []
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text) or self.chunk_size <= self.chunk_overlap:
                break
        self.chunks = chunks
        return chunks

    def calculate_embeddings(self):
        logger.info(f"[{self.__class__.__name__}] Calculating embeddings for {len(self.chunks)} text chunks...")
        embeddings = []
        for chunk in self.chunks:
            response = self.client.embeddings.create(
                input=chunk,
                model="text-embedding-3-large"
            )
            embeddings.append(response.data[0].embedding)
        self.embeddings = embeddings
        return embeddings

    def find_prompt_in_knowledge(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Querying prompt in retrieved vector embeddings...")
        # Generate prompt embedding
        prompt_resp = self.client.embeddings.create(
            input=prompt,
            model="text-embedding-3-large"
        )
        prompt_emb = prompt_resp.data[0].embedding

        # Find best matching chunk
        best_similarity = -1.0
        best_chunk = ""
        for chunk, emb in zip(self.chunks, self.embeddings):
            sim = cosine_similarity(prompt_emb, emb)
            if sim > best_similarity:
                best_similarity = sim
                best_chunk = chunk

        logger.info(f"[{self.__class__.__name__}] Selected best chunk with similarity score: {best_similarity:.4f}")

        # Query LLM with RAG context
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {best_chunk}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions=10):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.client = create_client(self.openai_api_key)

    def evaluate(self, prompt):
        current_prompt = prompt
        iterations = 0
        worker_response = ""
        evaluation_result = ""

        while iterations < self.max_interactions:
            iterations += 1
            logger.info(f"[{self.__class__.__name__}] Iteration {iterations}/{self.max_interactions}: Fetching worker response...")
            worker_response = self.worker_agent.respond(current_prompt)

            logger.info(f"[{self.__class__.__name__}] Iteration {iterations}/{self.max_interactions}: Evaluating response against criteria...")
            eval_system = f"You are {self.persona}."
            eval_user = (
                f"Evaluate the worker response against the evaluation criteria.\n"
                f"Criteria: {self.evaluation_criteria}\n"
                f"Worker Response: {worker_response}\n\n"
                f"Respond with a JSON object in this exact format:\n"
                f"{{\n"
                f"  \"status\": \"VALID\" or \"INVALID\",\n"
                f"  \"issues\": [\"list of specific reasons why it does not meet criteria, if INVALID\"]\n"
                f"}}\n"
                f"Do not include any other text, markdown formatting, or wrappers."
            )
            eval_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": eval_system},
                    {"role": "user", "content": eval_user}
                ],
                temperature=0
            )
            
            raw_eval_result = eval_response.choices[0].message.content.strip()
            logger.debug(f"[{self.__class__.__name__}] Raw evaluation response: {raw_eval_result}")

            # Try parsing JSON
            status = "INVALID"
            issues = []
            try:
                # Remove json block backticks if LLM mistakenly returned markdown json code block
                cleaned_result = raw_eval_result
                if cleaned_result.startswith("```"):
                    lines = cleaned_result.splitlines()
                    if len(lines) > 2:
                        cleaned_result = "\n".join(lines[1:-1])
                
                parsed_json = json.loads(cleaned_result)
                status = parsed_json.get("status", "INVALID").upper()
                issues = parsed_json.get("issues", [])
            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] Failed to parse JSON response: {e}. Falling back to text-based matching.")
                if "VALID" in raw_eval_result.upper():
                    status = "VALID"
                else:
                    issues = [raw_eval_result]

            if status == "VALID":
                logger.info(f"[{self.__class__.__name__}] Response evaluated as VALID after {iterations} iterations.")
                return {
                    "final_response": worker_response,
                    "evaluation": "VALID",
                    "iterations": iterations
                }

            evaluation_result = ", ".join(issues) if issues else "INVALID response format"
            logger.info(f"[{self.__class__.__name__}] Response evaluated as INVALID: {evaluation_result}")

            # Generate correction instructions
            logger.info(f"[{self.__class__.__name__}] Iteration {iterations}/{self.max_interactions}: Generating correction instructions...")
            correct_system = f"You are {self.persona}."
            correct_user = (
                f"The worker response did not meet the criteria.\n"
                f"Criteria: {self.evaluation_criteria}\n"
                f"Worker Response: {worker_response}\n"
                f"Issues identified: {evaluation_result}\n\n"
                f"Generate clear and direct instructions telling the worker agent how to correct their response to meet the criteria."
            )
            correct_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": correct_system},
                    {"role": "user", "content": correct_user}
                ],
                temperature=0
            )
            correction_instructions = correct_response.choices[0].message.content.strip()
            logger.debug(f"[{self.__class__.__name__}] Correction instructions: {correction_instructions}")

            # Update prompt for next iteration
            current_prompt = (
                f"Your previous response: {worker_response}\n"
                f"It was invalid for the following reasons: {evaluation_result}\n"
                f"Correction instructions: {correction_instructions}\n"
                f"Please provide a new response that fully addresses the prompt and these instructions."
            )

        logger.warning(f"[{self.__class__.__name__}] Reached max interactions ({self.max_interactions}) without obtaining a valid response.")
        return {
            "final_response": worker_response,
            "evaluation": evaluation_result,
            "iterations": iterations
        }


class RoutingAgent:
    def __init__(self, openai_api_key, config=None):
        self.openai_api_key = openai_api_key
        self.config = config or {}
        self.agents = []
        self.client = create_client(self.openai_api_key)

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding

    def route(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Routing prompt: {prompt}")
        prompt_embedding = self.get_embedding(prompt)
        best_similarity = -1.0
        selected_agent = None

        for agent in self.agents:
            desc_embedding = self.get_embedding(agent["description"])
            sim = cosine_similarity(prompt_embedding, desc_embedding)
            logger.debug(f"[{self.__class__.__name__}] Similarity to agent '{agent['name']}': {sim:.4f}")
            if sim > best_similarity:
                best_similarity = sim
                selected_agent = agent

        if selected_agent:
            logger.info(f"[{self.__class__.__name__}] Routing to selected agent: {selected_agent['name']} (similarity: {best_similarity:.4f})")
            return selected_agent["func"](prompt)
        
        logger.warning(f"[{self.__class__.__name__}] No suitable agent found for prompt.")
        return "No suitable agent found."


class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge=""):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = create_client(self.openai_api_key)

    def respond(self, prompt):
        logger.info(f"[{self.__class__.__name__}] Generating action plan steps for prompt...")
        system_message = (
            "You are an Action Planning Agent. Your task is to extract and list the chronological "
            "steps required to execute the task described in the user's prompt. "
            f"Use the following knowledge if relevant: {self.knowledge}. "
            "Format the output as a clear list of actions, one per line. Do not include intro or outro text."
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        
        steps = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            
            cleaned_line = line
            if cleaned_line.startswith(("-", "*", "+")):
                cleaned_line = cleaned_line[1:].strip()
            else:
                parts = cleaned_line.split(".", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    cleaned_line = parts[1].strip()
                else:
                    parts = cleaned_line.split(")", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        cleaned_line = parts[1].strip()
            
            if cleaned_line:
                steps.append(cleaned_line)
        
        logger.info(f"[{self.__class__.__name__}] Extracted {len(steps)} plan steps.")
        return steps
