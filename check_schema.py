import torch
import sys

def check_model(path):
    try:
        model = torch.jit.load(path)
        print(f"Model: {path}")
        # For JIT models, we can look at the graph or the forward method's schema
        forward = getattr(model, 'forward', None)
        if forward:
             print(f"  Schema: {forward.schema}")
        else:
             print("  No forward method found.")
    except Exception as e:
        print(f"  Error loading {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            check_model(p)
    else:
        print("Usage: python check_schema.py <model_path>")
