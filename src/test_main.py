import os
import json
import requests
from llama_cpp import Llama

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------
# 1. Load Environment Variables (DO NOT HARDCODE)
# ---------------------------------------------------------
FIREWORKS_AVAILABLE = True
API_KEY = None
BASE_URL = None
FIREWORKS_MODEL = None

try:
    API_KEY = os.environ["FIREWORKS_API_KEY"] # [cite: 56]
    BASE_URL = os.environ["FIREWORKS_BASE_URL"] # [cite: 58]
    # ALLOWED_MODELS is a comma-separated string [cite: 59]
    ALLOWED_MODELS = os.environ["ALLOWED_MODELS"].split(",") # [cite: 59]
    
    # Pick the first allowed model for external calls (or write logic to select)
    FIREWORKS_MODEL = ALLOWED_MODELS[0] 
except KeyError as e:
    print(f"WARNING: Missing environment variable {e}. Fireworks API will be disabled.")
    FIREWORKS_AVAILABLE = False


# ---------------------------------------------------------
# 2. File I/O Paths
# ---------------------------------------------------------
if os.path.exists("/input/tasks.json"):
    INPUT_PATH = "/input/tasks.json" # [cite: 20]
    OUTPUT_PATH = "/output/results.json" # [cite: 26]
    TOKEN_USAGE_PATH = "/output/token_usage.json"
else:
    # Resolve relative to the repository root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(base_dir, "input", "tasks.json")
    OUTPUT_PATH = os.path.join(base_dir, "output", "results.json")
    TOKEN_USAGE_PATH = os.path.join(base_dir, "output", "token_usage.json")

# ---------------------------------------------------------
# 3. Initialize Local Model (Zero Cost)
# ---------------------------------------------------------
# Assuming you pre-baked 'model.gguf' into your Docker image via COPY
LOCAL_MODEL_PATH = "./model.gguf" 
if not os.path.exists(LOCAL_MODEL_PATH):
    # Try parent directory relative to this script
    possible_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.gguf")
    if os.path.exists(possible_path):
        LOCAL_MODEL_PATH = possible_path

print(f"Loading local model from {LOCAL_MODEL_PATH} into memory...")
# Keep n_ctx small to save RAM in the 4GB environment [cite: 76]
llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=1024, verbose=False) 
print("Local model loaded.")


# ---------------------------------------------------------
# 4. Helper Functions & Token Metrics
# ---------------------------------------------------------
token_metrics = {
    "local_prompt_tokens": 0,
    "local_completion_tokens": 0,
    "api_prompt_tokens": 0,
    "api_completion_tokens": 0
}

def call_local_model(prompt):
    """Processes the prompt using the local zero-cost model and records token usage."""
    system_instruction = (
        "You are a highly precise and extremely concise assistant. "
        "Answer the user query directly and briefly. Avoid conversational intro/outro and unnecessary words."
    )
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256
    )
    usage = response.get("usage", {})
    p_tokens = usage.get("prompt_tokens", 0)
    c_tokens = usage.get("completion_tokens", 0)
    total_tokens = p_tokens + c_tokens
    
    token_metrics["local_prompt_tokens"] += p_tokens
    token_metrics["local_completion_tokens"] += c_tokens
    return response['choices'][0]['message']['content'].strip(), total_tokens


def call_fireworks_api(prompt, task_type="general"):
    """Processes the prompt using the premium Fireworks API with retries, timeout, and token tracking."""
    import time
    url = f"{BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    if task_type == "code":
        system_instruction = (
            "You are a strict programming assistant. Your task is to output ONLY the requested code. "
            "Do NOT include any conversational text, introductory thoughts, explanations, comments, or markdown explanations. "
            "Output only raw or formatted code. Start your response directly with the code."
        )
    elif task_type == "math_logic":
        system_instruction = (
            "You are a precise math and logic assistant. Provide ONLY the final answer and a short, direct step-by-step logic proof. "
            "Do NOT include any conversational filler."
        )
    else:
        system_instruction = (
            "You are a highly precise assistant. Provide correct, complete answers but keep your response direct and as concise as possible. "
            "Avoid conversational intro/outro, friendly filler, or excessively long explanations."
        )
    
    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024
    }
    
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            res_json = response.json()
            usage = res_json.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            total_tokens = p_tokens + c_tokens
            
            token_metrics["api_prompt_tokens"] += p_tokens
            token_metrics["api_completion_tokens"] += c_tokens
            return res_json["choices"][0]["message"]["content"].strip(), total_tokens
        except Exception as e:
            if attempt == max_attempts:
                print(f"Fireworks API call failed after {max_attempts} attempts: {e}")
                raise e
            wait_time = attempt * 2
            print(f"Warning: Connection attempt {attempt} failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)


def is_simple_arithmetic(prompt_lower):
    import re
    # Map word numbers and operators
    words_map = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "multiplied": "*",
        "divided by": "/", "divided": "/", "percent of": "/100*", "percent": "/100"
    }
    cleaned = prompt_lower
    for word, replacement in words_map.items():
        cleaned = re.sub(r'\b' + re.escape(word) + r'\b', replacement, cleaned)
    
    cleaned = re.sub(r'(what is|calculate|solve|evaluate|\?|\s|\=)', '', cleaned)
    return bool(re.match(r'^[\d\+\-\*\/\(\)\.]+$', cleaned)) and len(cleaned) > 0


def classify_locally(prompt):
    classification_prompt = (
        "You are a task routing classifier. Decide if a prompt is SIMPLE (NLP tasks relying purely on text in the prompt, basic arithmetic, greetings) or COMPLEX (general knowledge, logic puzzles, coding, debugging, guides, reasoning).\n\n"
        "Examples:\n"
        "Prompt: \"Summarise the following: The cat sat on the mat.\"\n"
        "Category: SIMPLE\n\n"
        "Prompt: \"What is the capital of France?\"\n"
        "Category: COMPLEX\n\n"
        "Prompt: \"Write a python script to reverse a string\"\n"
        "Category: COMPLEX\n\n"
        "Prompt: \"Solve 2 + 2\"\n"
        "Category: SIMPLE\n\n"
        "Prompt: \"A container has 10L. 2L leaks out. How much is left?\"\n"
        "Category: COMPLEX\n\n"
        "Prompt: \"Extract all companies from this article: Apple announced new phones.\"\n"
        "Category: SIMPLE\n\n"
        f"Prompt: \"{prompt}\"\n"
        "Category:"
    )
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": classification_prompt}
        ],
        max_tokens=3,
        temperature=0.0
    )
    content = response['choices'][0]['message']['content'].strip().upper()
    return "SIMPLE" in content or "COMPLEX" not in content


def route_task(prompt):
    """
    Decides whether to use the local model or the premium Fireworks API.
    Routes straightforward NLP tasks (summarization, sentiment analysis, extraction, simple translation)
    and simple arithmetic to the zero-cost local model.
    Routes complex tasks (advanced math, logic puzzles, coding, instructions, factual)
    to the Fireworks API to maximize accuracy while minimizing cost.
    """
    global FIREWORKS_AVAILABLE
    
    if not FIREWORKS_AVAILABLE:
        print("Routing to Local Model (Fireworks API is unavailable)")
        return call_local_model(prompt)
        
    prompt_lower = prompt.lower().strip()
    code_indicators = ["def ", "class ", "function", "bug", "python", "javascript", "c++", "regex"]
    
    # 1. Quick length limit (long tasks must be routed to the premium API)
    if len(prompt) > 800:
        print("Routing to Fireworks API (Long prompt context)")
        try:
            return call_fireworks_api(prompt, task_type="general")
        except Exception as e:
            print(f"Error calling Fireworks API: {e}. Falling back to local model.")
            FIREWORKS_AVAILABLE = False
            return call_local_model(prompt)
            
    # 2. Heuristics: Simple math check
    if is_simple_arithmetic(prompt_lower):
        print("Routing to Local Model (Simple Math Task)")
        return call_local_model(prompt)
        
    # 3. Heuristics: Coding/development keywords
    if any(ind in prompt_lower for ind in code_indicators):
        print("Routing to Fireworks API (Coding task keyword)")
        try:
            return call_fireworks_api(prompt, task_type="code")
        except Exception as e:
            print(f"Error calling Fireworks API: {e}. Falling back to local model.")
            FIREWORKS_AVAILABLE = False
            return call_local_model(prompt)
            
    # 4. Heuristics: Light NLP keywords
    light_keywords = [
        "summarize", "summarise", "summary", 
        "sentiment", "classify the sentiment", 
        "extract all named entities", "extract entities", "named entity extraction",
        "translate", "translation"
    ]
    if any(kw in prompt_lower for kw in light_keywords):
        print("Routing to Local Model (Light NLP Task)")
        return call_local_model(prompt)
        
    # 5. LLM Classification fallback for ambiguous queries
    print("Evaluating prompt complexity locally...")
    is_simple = classify_locally(prompt)
    if is_simple:
        print("Routing to Local Model (LLM Classified Simple)")
        return call_local_model(prompt)
    else:
        print("Routing to Fireworks API (LLM Classified Complex)")
        # Classify task type for optimized system prompting
        if any(ind in prompt_lower for ind in code_indicators):
            t_type = "code"
        elif any(ind in prompt_lower for ind in ["solve", "calculate", "math", "logic", "puzzle", "riddle"]):
            t_type = "math_logic"
        else:
            t_type = "general"
            
        try:
            return call_fireworks_api(prompt, task_type=t_type)
        except Exception as e:
            print(f"Error calling Fireworks API: {e}. Falling back to local model.")
            FIREWORKS_AVAILABLE = False
            return call_local_model(prompt)



# ---------------------------------------------------------
# 5. Main Execution Loop
# ---------------------------------------------------------
def main():
    # Read the tasks [cite: 20]
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found at {INPUT_PATH}")
        exit(1)
        
    import re
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # Clean up trailing commas so that standard json parser won't fail
    content_clean = re.sub(r',\s*([\]}])', r'\1', content)
    tasks = json.loads(content_clean)
        
    results = []

    
    # Process each task
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"Processing task: {task_id}...")
        
        try:
            answer, tokens_used = route_task(prompt)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            answer = "Error generating response."
            tokens_used = 0
            
        results.append({
            "task_id": task_id,
            "answer": answer,
            "tokens_used": tokens_used
        })
        
    # Sort results by tokens_used descending
    results.sort(key=lambda x: x["tokens_used"], reverse=True)
        
    # Write the results [cite: 26]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        # Remove 'tokens_used' to strictly match the expected evaluation schema
        final_results = [{"task_id": r["task_id"], "answer": r["answer"]} for r in results]
        json.dump(final_results, f, indent=2)

    # Write token tracking to a separate file
    token_report_data = {
        "summary": {
            "local_prompt_tokens": token_metrics['local_prompt_tokens'],
            "local_completion_tokens": token_metrics['local_completion_tokens'],
            "api_prompt_tokens": token_metrics['api_prompt_tokens'],
            "api_completion_tokens": token_metrics['api_completion_tokens'],
            "total_local_tokens": token_metrics['local_prompt_tokens'] + token_metrics['local_completion_tokens'],
            "total_api_tokens": token_metrics['api_prompt_tokens'] + token_metrics['api_completion_tokens'],
            "total_tokens": token_metrics['local_prompt_tokens'] + token_metrics['local_completion_tokens'] + token_metrics['api_prompt_tokens'] + token_metrics['api_completion_tokens']
        },
        "tasks": {r["task_id"]: r["tokens_used"] for r in results}
    }
    with open(TOKEN_USAGE_PATH, "w") as f:
        json.dump(token_report_data, f, indent=2)
        
    print(f"Successfully processed {len(tasks)} tasks. Exiting cleanly.")
    
    print("\n" + "="*40)
    print("TOKEN USAGE REPORT:")
    print(f"  Local Model Prompt Tokens:      {token_metrics['local_prompt_tokens']}")
    print(f"  Local Model Completion Tokens:  {token_metrics['local_completion_tokens']}")
    print(f"  Fireworks API Prompt Tokens:    {token_metrics['api_prompt_tokens']}")
    print(f"  Fireworks API Completion Tokens: {token_metrics['api_completion_tokens']}")
    print(f"  Total Local Tokens:             {token_metrics['local_prompt_tokens'] + token_metrics['local_completion_tokens']}")
    print(f"  Total Fireworks API Tokens:     {token_metrics['api_prompt_tokens'] + token_metrics['api_completion_tokens']}")
    print("="*40)
    print("TOKEN USAGE BY TASK (Sorted Descending):")
    for r in results:
        print(f"  Task {r['task_id']}: {r['tokens_used']} tokens")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()