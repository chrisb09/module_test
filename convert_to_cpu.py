import sys
import torch
import os

if len(sys.argv) < 3:
    print("Usage: python convert_to_cpu.py <src> <dst>")
    sys.exit(1)

src = sys.argv[1]
dst = sys.argv[2]

if os.path.exists(dst):
    # Check if dst is newer than src
    if os.path.getmtime(dst) > os.path.getmtime(src):
        print(f"Model {dst} is already up to date.")
        sys.exit(0)

print(f"Converting {src} to CPU model {dst}...")
try:
    model = torch.jit.load(src, map_location="cpu")
    model = model.eval()
    torch.jit.save(model, dst)
    print("Done.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
