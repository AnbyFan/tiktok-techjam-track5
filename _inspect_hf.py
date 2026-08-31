from huggingface_hub import HfApi

api = HfApi()
repo_id = "Shamima/sd3-medium-scm-corpus"
exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

files = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
print("total entries:", len(files))

imgs = [f for f in files if hasattr(f, "size") and f.path.lower().endswith(exts)]
print("image files:", len(imgs))
for f in imgs[:25]:
    print(f"  {f.path}  ({f.size} bytes)")

# Also show non-image top-level files for context
non_img = [f.path for f in files if hasattr(f, "size") and not f.path.lower().endswith(exts)]
print("\nnon-image files (first 20):")
for p in non_img[:20]:
    print("  ", p)
