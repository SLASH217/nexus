import os
from huggingface_hub import snapshot_download

# This is where the REAL files will land
LOCAL_DIR = os.path.expanduser("~/hf_cache_temp")

models = [
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
]


def download_flat():
    print(f"🚀 Starting FLAT download to: {LOCAL_DIR}")

    for model_id in models:
        # Create a clean folder name for each model
        # e.g., ~/hf_cache_temp/gemma-2-9b-it-bnb-4bit
        folder_name = model_id.split("/")[-1]
        target_path = os.path.join(LOCAL_DIR, folder_name)

        print(f"\n📥 Securing {model_id}...")
        try:
            # local_dir: Forces the files into this specific folder
            # local_dir_use_symlinks=False: FORCES real data, not pointers
            snapshot_download(
                repo_id=model_id,
                local_dir=target_path,
                local_dir_use_symlinks=False,
                token=True,  # Uses your saved HF login if needed
            )
            print(f"✅ {model_id} is now a physical resident of your SSD.")
        except Exception as e:
            print(f"❌ Failed {model_id}: {e}")


if __name__ == "__main__":
    download_flat()
