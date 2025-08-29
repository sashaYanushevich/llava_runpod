import os
from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("MODEL_ID", "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384")
cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

print(f"[download_models] Downloading {MODEL_ID} -> cache_dir={cache_dir!r}")
snapshot_download(repo_id=MODEL_ID, cache_dir=cache_dir, local_files_only=False)
print("[download_models] Done.")
