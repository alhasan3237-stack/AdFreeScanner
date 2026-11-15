import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen2
N = int(sys.argv[1]); part = int(sys.argv[2]); parts = int(sys.argv[3]); base = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
X = np.lib.format.open_memmap(f"/root/gen/data_{part}.npy", mode="w+", dtype=np.uint8, shape=(N // parts, 256, 256, 3))
C = np.zeros((N // parts, 4, 2), np.float32)
t = time.time()
for k in range(N // parts):
    i = base + k * parts + part
    im, c = gen2.scene(i)
    X[k], C[k] = gen2.letterbox256(im, c)
    if k % 500 == 0: print(part, k, f"{time.time() - t:.0f}s", flush=True)
X.flush(); np.save(f"/root/gen/corners_{part}.npy", C); print("done", part)
