import os
import json
import requests
# pyrefly: ignore [missing-import]
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
    API_KEY = os.environ["FIREWORKS_API_KEY"]
    BASE_URL = os.environ["FIREWORKS_BASE_URL"]
    # ALLOWED_MODELS is a comma-separated string
    ALLOWED_MODELS = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",")]
    
    # Pick the first allowed model for external calls (or write logic to select)
    FIREWORKS_MODEL = ALLOWED_MODELS[0] 
except KeyError as e:
    print(f"WARNING: Missing environment variable {e}. Fireworks API will be disabled.")
    FIREWORKS_AVAILABLE = False

track_1_routing_matrix = {
    "sentiment_classification": [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/glm-5p1"
    ],
    "text_summarisation": [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/glm-5p1"
    ],
    "named_entity_recognition": [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/glm-5p1"
    ],
    "router_intent_classification": [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/glm-5p1"
    ],
    "code_generation": [
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/minimax-m2p7"
    ],
    "code_debugging": [
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/minimax-m2p7"
    ],
    "multi_step_math": [
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/glm-5p1"
    ]
}

# ---------------------------------------------------------
# 2. File I/O Paths
# ---------------------------------------------------------
if os.path.exists("/input/tasks.json"):
    INPUT_PATH = "/input/tasks.json"
    OUTPUT_PATH = "/output/results.json"
    TOKEN_USAGE_PATH = None
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
# Keep n_ctx small to save RAM in the 4GB environment
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
    
    # Choose model based on track_1_routing_matrix and ALLOWED_MODELS
    task_category = None
    prompt_lower = prompt.lower().strip()
    
    if task_type == "code":
        if "bug" in prompt_lower or "fix" in prompt_lower or "correct" in prompt_lower:
            task_category = "code_debugging"
        else:
            task_category = "code_generation"
    elif task_type == "math_logic":
        task_category = "multi_step_math"
    else:
        # Heuristics based on prompt content
        if any(kw in prompt_lower for kw in ["summarize", "summarise", "summary"]):
            task_category = "text_summarisation"
        elif any(kw in prompt_lower for kw in ["sentiment", "classify the sentiment"]):
            task_category = "sentiment_classification"
        elif any(kw in prompt_lower for kw in ["extract all named entities", "extract entities", "named entity extraction", "ner"]):
            task_category = "named_entity_recognition"
        elif any(kw in prompt_lower for kw in ["routing", "router", "intent"]):
            task_category = "router_intent_classification"
        elif any(ind in prompt_lower for ind in ["def ", "class ", "function", "python", "javascript", "c++", "regex"]):
            if "bug" in prompt_lower or "fix" in prompt_lower:
                task_category = "code_debugging"
            else:
                task_category = "code_generation"
        elif any(ind in prompt_lower for ind in ["solve", "calculate", "math", "logic", "puzzle", "riddle", "percent", "remain"]):
            task_category = "multi_step_math"
        else:
            task_category = "sentiment_classification"
            
    chosen_model = None
    for model_name in track_1_routing_matrix.get(task_category, []):
        if model_name in ALLOWED_MODELS:
            chosen_model = model_name
            break
            
    if not chosen_model:
        chosen_model = FIREWORKS_MODEL
        
    print(f"Routing task category '{task_category}' to model: {chosen_model}")
    
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
        "model": chosen_model,
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
    to the zero-cost local model, and routes complex tasks (math, logic puzzles, coding, instructions, factual)
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
    # Read the tasks
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

    
    task_token_usage = {}
    
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
            "answer": answer
        })
        task_token_usage[task_id] = tokens_used
        
    # Write the results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Write token tracking to a separate file only if set
    if TOKEN_USAGE_PATH is not None:
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
            "tasks": task_token_usage
        }
        with open(TOKEN_USAGE_PATH, "w") as f:
            json.dump(token_report_data, f, indent=2)
        
    print(f"Successfully processed {len(tasks)} tasks. Exiting cleanly.")

if __name__ == "__main__":
    main()
