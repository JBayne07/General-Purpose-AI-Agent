from dotenv import load_dotenv
from fireworks import Fireworks
import os

# # Load environment variables from .env
load_dotenv()

# api_key = os.environ["FIREWORKS_API_KEY"]  # provided by harness — do not use your own
# base_url = os.environ[
#     "FIREWORKS_BASE_URL"
# ]  # route ALL Fireworks calls through this URL
# models = os.environ["ALLOWED_MODELS"].split(
#     ","
# )  # exact model IDs published on launch day


def main():
    print("Hello, World!")
    client = (
        Fireworks(api_key=api_key, base_url=base_url)
        if base_url
        else Fireworks(api_key=api_key)
    )
    response = client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v3p1",
        messages=[
            {
                "role": "user",
                "content": "Say hello in Spanish",
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
