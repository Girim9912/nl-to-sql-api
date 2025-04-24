from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/codet5-base",
    local_dir="./models/codet5",
    local_dir_use_symlinks=False
)
