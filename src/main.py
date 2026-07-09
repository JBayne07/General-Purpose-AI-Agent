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
    API_KEY = os.environ["FIREWORKS_API_KEY"]
    BASE_URL = os.environ["FIREWORKS_BASE_URL"]
    # ALLOWED_MODELS is a comma-separated string
    ALLOWED_MODELS = os.environ["ALLOWED_MODELS"].split(",")
    
    # Pick the first allowed model for external calls (or write logic to select)
    FIREWORKS_MODEL = ALLOWED_MODELS[0] 
except KeyError as e:
    print(f"WARNING: Missing environment variable {e}. Fireworks API will be disabled.")
    FIREWORKS_AVAILABLE = False


# ---------------------------------------------------------
# 2. File I/O Paths
# ---------------------------------------------------------
if os.path.exists("/input/tasks.json"):
    INPUT_PATH = "/input/tasks.json"
    OUTPUT_PATH = "/output/results.json"
else:
    # Resolve relative to the repository root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(base_dir, "input", "tasks.json")
    OUTPUT_PATH = os.path.join(base_dir, "output", "results.json")

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
# 4. Helper Functions
# ---------------------------------------------------------
def call_local_model(prompt):
    """Processes the prompt using the local zero-cost model."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=256
    )
    return response['choices'][0]['message']['content'].strip()


def call_fireworks_api(prompt):
    """Processes the prompt using the premium Fireworks API with retries and timeout."""
    import time
    url = f"{BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512
    }
    
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_attempts:
                print(f"Fireworks API call failed after {max_attempts} attempts: {e}")
                raise e
            wait_time = attempt * 2
            print(f"Warning: Connection attempt {attempt} failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)


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
        
    prompt_lower = prompt.lower()
    
    # If the prompt is very long, it requires larger context and better comprehension
    if len(prompt) > 800:
        print("Routing to Fireworks API (Long prompt context)")
        try:
            return call_fireworks_api(prompt)
        except Exception as e:
            print(f"Error calling Fireworks API: {e}. Falling back to local model.")
            FIREWORKS_AVAILABLE = False
            return call_local_model(prompt)
        
    # Light task keywords (NLP tasks where Qwen 0.5B excels)
    light_keywords = [
        "summarize", "summarise", "summary", 
        "sentiment", "classify the sentiment", 
        "extract all named entities", "extract entities", "named entity extraction",
        "translate", "translation"
    ]
    
    # Check if prompt contains any of the light keywords
    is_light = any(kw in prompt_lower for kw in light_keywords)
    
    # Heavy indicators (even if a light keyword is present, if it looks like coding/math it should go to Fireworks)
    heavy_indicators = ["python", "javascript", "c++", "code", "bug", "function", "solve", "calculate"]
    has_heavy = any(ind in prompt_lower for ind in heavy_indicators)
    
    if is_light and not has_heavy:
        print("Routing to Local Model (Light NLP Task)")
        return call_local_model(prompt)
    else:
        print("Routing to Fireworks API (Heavy/Reasoning/Factual Task)")
        try:
            return call_fireworks_api(prompt)
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

    
    # Process each task
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"Processing task: {task_id}...")
        
        try:
            answer = route_task(prompt)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            answer = "Error generating response."
            
        results.append({
            "task_id": task_id,
            "answer": answer
        })
        
    # Write the results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully processed {len(tasks)} tasks. Exiting cleanly.")

if __name__ == "__main__":
    main()
