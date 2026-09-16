"""Extra checks added during the 2026-09-16 review. Run from this directory: python extra_checks.py"""
from collections import Counter
import subprocess
import sys
import numpy as np
from scipy.spatial.distance import cdist
from main import load, nearest, sample

if len(sys.argv) > 1:  # fresh-process probe: peak working set before and after the search step (Windows psutil)
    import psutil
    peak = lambda: psutil.Process().memory_info().peak_wset / 2**20
    q, r, y, labels = sample()
    after_load, rss_after_load = peak(), psutil.Process().memory_info().rss / 2**20
    ids = cdist(q, r).argmin(1) if sys.argv[1] == "full" else nearest(q, r)[0]
    print(f"{sys.argv[1]:5s}: peak after sample() {after_load:6.2f} MiB (rss {rss_after_load:6.2f}) | peak after search {peak():6.2f} MiB | search added {peak() - after_load:+7.2f} MiB")
    sys.exit()

q, r, y, labels = sample()
ids, dist = nearest(q, r)

# Second oracle, different algorithm from SciPy's per-pair subtraction: ||q||^2 + ||r||^2 - 2 q.r
gram = (q ** 2).sum(1)[:, None] + (r ** 2).sum(1)[None, :] - 2 * q @ r.T
print("expansion-formula oracle: indices agree", int((gram.argmin(1) == ids).sum()), "/", len(ids),
      "| max |distance diff|", f"{np.abs(np.sqrt(np.maximum(gram.min(1), 0)) - dist).max():.1e}")

# Odd tile sizes on the real data: partial tiles, cross-tile ties, tile larger than both inputs
for tile in [37, 1000, 4095, 5000, 100000]:
    i2, d2 = nearest(q, r, tile)
    print(f"tile={tile:>6}: indices identical {np.array_equal(i2, ids)}, distances identical {np.array_equal(d2, dist)}")

# Baselines for the condition-label accuracy, per-class recall, and split hygiene
classes = np.unique(labels)
majority = Counter(labels.astype(int)).most_common(1)[0][0]
centroids = np.stack([r[labels == c].mean(0) for c in classes])
print(f"majority-class baseline (predict reference class {majority} for every query): {(y == majority).mean():.2%}")
print(f"nearest-centroid baseline: {(classes[cdist(q, centroids).argmin(1)] == y).mean():.2%}")
print(f"1-NN condition accuracy: {(labels[ids] == y).mean():.2%}")
print("1-NN per-class recall:", {int(c): f"{(labels[ids][y == c] == c).mean():.1%}" for c in classes})
print("rows shared between query and reference:", len(set(map(tuple, q)) & set(map(tuple, r))))

# Seed spread: same split sizes, different shuffles
data = np.unique(load(), axis=0)
spread = []
for seed in [42, 0, 1, 2, 3, 4]:
    d = data[np.random.default_rng(seed).permutation(len(data))]
    ref, qu = d[:8192], d[8192:12288]
    mean, scale = ref[:, :-1].mean(0), ref[:, :-1].std(0)
    scale[scale == 0] = 1
    i, _ = nearest((qu[:, :-1] - mean) / scale, (ref[:, :-1] - mean) / scale)
    spread.append((ref[:, -1][i] == qu[:, -1]).mean())
print("1-NN accuracy over seeds 42,0,1,2,3,4:", [f"{a:.2%}" for a in spread], f"| mean {np.mean(spread):.2%}")

# Where the peak memory comes from, in fresh processes
for mode in ["tiled", "full", "tiled", "full"]:
    print(subprocess.check_output([sys.executable, __file__, mode], text=True).rstrip())
