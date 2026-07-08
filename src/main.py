from dotenv import load_dotenv
from fireworks.client import Fireworks
import os
import utils

# Load environment variables from .env
load_dotenv()

api_key = os.environ.get("FIREWORKS_API_KEY")  # provided by harness — do not use your own
base_url = os.environ.get(
    "FIREWORKS_BASE_URL"
)  # route ALL Fireworks calls through this URL
models = os.environ.get("ALLOWED_MODELS", "").split(
    ","
)  # exact model IDs published on launch day


def main():
    if not api_key:
        print("Error: FIREWORKS_API_KEY is not set.")
        return

    client = (
        Fireworks(api_key=api_key, base_url=base_url)
        if base_url
        else Fireworks(api_key=api_key)
    )

    tasks = utils.getInputJson()
    print(f"Tasks: {tasks}")
    
    # Use the first allowed model or deepseek-v4-pro as fallback
    model_to_use = models[0] if models and models[0] else "accounts/fireworks/models/gemma-4-26b-a4b-it"
    print(f"Using model: {model_to_use}")
    
    import time
    max_attempts = 3
    for task in tasks:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "user",
                            "content": task['prompt'],
                        }
                    ],
                )
                print(f"Response: {response.choices[0].message.content}")
                result = utils.formatOutput(task["task_id"], response.choices[0].message.content)
                utils.addToOutputJson(result)
                break
            except Exception as e:
                if attempt == max_attempts:
                    print(f"Error: Failed after {max_attempts} attempts.")
                    raise e
                print(f"Warning: Connection attempt {attempt} failed ({e}). Retrying in {attempt * 2}s...")
                time.sleep(attempt * 2)


if __name__ == "__main__":
    main()
