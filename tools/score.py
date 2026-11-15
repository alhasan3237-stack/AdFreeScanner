import cv2, numpy as np, sys, importlib, time
from synth2 import sample

def metrics(out, d):
    ideal = d["ideal"]
    og = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(float) if out.ndim == 3 else out.astype(float)
    ig = cv2.cvtColor(ideal, cv2.COLOR_BGR2GRAY).astype(float)
    special = d["big"] | d["col"] | d["faint"]
    paper = (ig > 250) & ~special
    paper = cv2.erode(paper.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0       # away from ink
    ink = (ig < 120) & ~d["big"] & ~d["col"]
    m = {}
    m["paper_white%"] = 100 * np.mean(og[paper] >= 240)                             # clean white paper
    blur = cv2.GaussianBlur(og, (0, 0), 12)
    m["shadow_left"] = np.mean(np.abs(255 - blur[paper]))                            # leftover shadow / blotches
    m["text_err"] = np.mean(np.abs(og[ink] - ig[ink]))                               # text darkness vs ideal
    m["dark_err"] = np.mean(np.abs(og[d["big"]] - ig[d["big"]])) if d["big"].any() else np.nan  # photos/boxes kept
    m["faint_seen%"] = 100 * np.mean(og[d["faint"]] < 215) if d["faint"].any() else np.nan     # pencil/thermal visible
    if out.ndim == 3 and d["col"].any():
        lo = cv2.cvtColor(out, cv2.COLOR_BGR2Lab).astype(float); li = cv2.cvtColor(ideal, cv2.COLOR_BGR2Lab).astype(float)
        m["color_dE"] = np.mean(np.linalg.norm(lo[d["col"]] - li[d["col"]], axis=1))
    else: m["color_dE"] = np.nan
    hp = og - cv2.GaussianBlur(og, (0, 0), 2)
    m["noise"] = np.std(hp[paper])                                                    # speckle on paper
    return m

def overall(r):
    # 0-100, higher better. Weights reflect what people notice first.
    s = 0.30 * r["paper_white%"] + 0.20 * max(0, 100 - r["shadow_left"] * 8) + 0.20 * max(0, 100 - r["text_err"] * 1.5)
    s += 0.10 * max(0, 100 - np.nan_to_num(r["dark_err"]) * 1.5) + 0.08 * np.nan_to_num(r["faint_seen%"], nan=100)
    s += 0.07 * max(0, 100 - np.nan_to_num(r["color_dE"]) * 2) + 0.05 * max(0, 100 - r["noise"] * 10)
    return s

def run(fn, seeds, verbose=True):
    rows = []; t = 0
    for s in seeds:
        d = sample(s); t0 = time.time(); out = fn(d["photo"]); t += time.time() - t0
        rows.append(metrics(out, d) | {"kind": d["kind"]})
    keys = ["paper_white%", "shadow_left", "text_err", "dark_err", "faint_seen%", "color_dE", "noise"]
    avg = {k: np.nanmean([r[k] for r in rows]) for k in keys}
    avg["score"] = np.mean([overall(r) for r in rows]); avg["ms"] = t / len(seeds) * 1000
    if verbose:
        print("  ".join(f"{k}={v:.1f}" for k, v in avg.items()))
        worst = sorted(zip(seeds, [overall(r) for r in rows], [r["kind"] for r in rows]), key=lambda x: x[1])[:5]
        print("  worst:", [(s, round(v, 1), k) for s, v, k in worst])
    return avg

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1]); fn = getattr(mod, sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60; base = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(f"{sys.argv[1]}.{sys.argv[2]}:"); run(fn, range(base, base + n))
