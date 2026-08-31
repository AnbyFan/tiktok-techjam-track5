import sys
print("STEP1: starting", flush=True)
import torch
print("STEP2: torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), flush=True)
import open_clip
print("STEP3: open_clip imported", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device="cuda")
model.eval()
print("STEP4: model loaded on cuda", flush=True)
import numpy as np
x = torch.randn(2,3,224,224).cuda()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    f = model.encode_image(x)
print("STEP5: forward pass OK, feat shape", tuple(f.shape), flush=True)
print("ALL DONE", flush=True)
