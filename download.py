import os
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

def download_model():
    repos = [
        "mradermacher/gemma-4-E2B-it-GGUF",
        "unsloth/gemma-4-E2B-it-GGUF",
        "bartowski/google_gemma-4-E2B-it-GGUF"
    ]
    
    for repo_id in repos:
        print(f"Checking repo: {repo_id}")
        try:
            files = [f.rfilename for f in api.model_info(repo_id).siblings if f.rfilename.endswith('.gguf')]
            # Look for Q4_K_M first
            target_file = next((f for f in files if 'Q4_K_M' in f), None)
            if not target_file and files:
                target_file = files[0]
                
            if target_file:
                print(f"Found target file: {target_file}. Downloading...")
                local_path = hf_hub_download(repo_id=repo_id, filename=target_file, local_dir=".")
                print(f"Downloaded to {local_path}")
                
                # Rename to model.gguf
                if os.path.exists("model.gguf"):
                    os.remove("model.gguf")
                os.rename(local_path, "model.gguf")
                print("Successfully renamed to model.gguf")
                return True
        except Exception as e:
            print(f"Error checking/downloading from {repo_id}: {e}")
            continue
            
    print("Failed to find and download a suitable model.")
    return False

if __name__ == "__main__":
    download_model()
