"""Second-pass experiments from EXPERIMENT_PROMPT.md. Run from this directory: python experiments.py

Child mode (one fresh process per measurement): python experiments.py <full|tiled|tiled_blas> <n_ref> <tile> <split>
"""
import hashlib
import json
import statistics
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from main import load, nearest, sample

QUERIES = 4096


def scaling_split(n_ref):
    """Fixed 4,096 queries taken after the largest reference block; references nested by prefix."""
    data = np.unique(load(), axis=0)
    data = data[np.random.default_rng(42).permutation(len(data))]
    reference, query = data[:n_ref], data[32768:32768 + QUERIES]
    mean, scale = reference[:, :-1].mean(0), reference[:, :-1].std(0)
    scale[scale == 0] = 1
    return (query[:, :-1] - mean) / scale, (reference[:, :-1] - mean) / scale


def nearest_blas(query, reference, tile=256):
    """Same tile loop as main.nearest, but the kernel is ||q||^2 + ||r||^2 - 2 q.r via a matrix multiply."""
    qn, rn = (query ** 2).sum(1), (reference ** 2).sum(1)
    best, ids = np.full(len(query), np.inf), np.zeros(len(query), dtype=int)
    for start in range(0, len(query), tile):
        block, bn = query[start:start + tile], qn[start:start + tile]
        for offset in range(0, len(reference), tile):
            ref = reference[offset:offset + tile]
            d2 = bn[:, None] + rn[offset:offset + tile][None, :] - 2.0 * (block @ ref.T)
            local = d2.argmin(axis=1)
            values = d2[np.arange(len(block)), local]
            better = values < best[start:start + len(block)]
            best[start:start + len(block)][better] = values[better]
            ids[start:start + len(block)][better] = offset + local[better]
    return ids, np.sqrt(np.maximum(best, 0))


def run_child(method, n_ref, tile, split):
    import psutil
    if split == "main":
        q, r = sample()[:2]
    else:
        q, r = scaling_split(n_ref)
    start = time.perf_counter()
    if method == "full":
        matrix = cdist(q, r)
        ids = matrix.argmin(1)
        distances = matrix[np.arange(len(q)), ids]
    elif method == "tiled":
        ids, distances = nearest(q, r, tile)
    else:
        ids, distances = nearest_blas(q, r, tile)
    elapsed = time.perf_counter() - start
    print(json.dumps(dict(seconds=elapsed, peak_mib=psutil.Process().memory_info().peak_wset / 2 ** 20,
                          index_sha256=hashlib.sha256(ids.tobytes()).hexdigest()[:16],
                          distance_sha256=hashlib.sha256(distances.tobytes()).hexdigest()[:16])))


def measure(method, n_ref, tile, split, reps=3):
    runs = [json.loads(subprocess.check_output([sys.executable, __file__, method, str(n_ref), str(tile), split], text=True))
            for _ in range(reps)]
    return dict(method=method, n_ref=n_ref, tile=tile, split=split, runs=runs,
                median_seconds=statistics.median(r["seconds"] for r in runs),
                median_peak_mib=statistics.median(r["peak_mib"] for r in runs),
                index_sha256=sorted({r["index_sha256"] for r in runs}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_child(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        sys.exit()

    results = {"scaling": [], "tile_sweep": [], "kernel": {}}

    print("== 1. Reference-count sweep: 4,096 fixed queries, nested references, 3 fresh processes each ==")
    print(f"{'refs':>6} {'method':>6} {'peak MiB':>9} {'seconds':>8} {'matrix MiB':>10}  index hash")
    for n_ref in [8192, 16384, 32768]:
        row = {}
        for method in ["full", "tiled"]:
            m = measure(method, n_ref, 256, "scaling")
            results["scaling"].append(m)
            row[method] = m
            print(f"{n_ref:>6} {method:>6} {m['median_peak_mib']:>9.2f} {m['median_seconds']:>8.4f} "
                  f"{QUERIES * n_ref * 8 / 2**20 if method == 'full' else 0.5:>10.1f}  {m['index_sha256']}")
        assert row["full"]["index_sha256"] == row["tiled"]["index_sha256"], f"full/tiled disagree at n_ref={n_ref}"

    print("\n== 2. Tile-size sweep at 8,192 references (main.py split), cdist kernel and BLAS kernel, 3 fresh processes each ==")
    print(f"{'tile':>6} {'kernel':>10} {'peak MiB':>9} {'seconds':>8}  index hash")
    for tile in [32, 64, 128, 256, 512, 1024, 4096]:
        for method in ["tiled", "tiled_blas"]:
            m = measure(method, 8192, tile, "main")
            results["tile_sweep"].append(m)
            print(f"{tile:>6} {'cdist' if method == 'tiled' else 'blas':>10} {m['median_peak_mib']:>9.2f} {m['median_seconds']:>8.4f}  {m['index_sha256']}")

    print("\n== 3. Kernel check: BLAS expansion vs exact cdist on the main.py split, in-process ==")
    q, r, y, labels = sample()
    ids_exact, dist_exact = nearest(q, r)
    ids_blas, dist_blas = nearest_blas(q, r)
    agree = int((ids_exact == ids_blas).sum())
    max_diff = float(np.abs(dist_exact - dist_blas).max())
    results["kernel"] = dict(indices_agree=agree, of=len(ids_exact), max_abs_distance_diff=max_diff,
                             accuracy_exact=float((labels[ids_exact] == y).mean()),
                             accuracy_blas=float((labels[ids_blas] == y).mean()))
    print(f"indices agree {agree} / {len(ids_exact)} | max |distance diff| {max_diff:.2e} | "
          f"condition accuracy exact {results['kernel']['accuracy_exact']:.2%} vs blas {results['kernel']['accuracy_blas']:.2%}")
    if agree != len(ids_exact):
        diff = np.flatnonzero(ids_exact != ids_blas)
        print(f"differing queries: {diff.tolist()[:10]}{'...' if len(diff) > 10 else ''}; "
              f"exact-distance gap at those queries: {np.abs(dist_exact[diff] - cdist(q[diff], r)[np.arange(len(diff)), ids_blas[diff]]).max():.2e}")

    results["python"], results["numpy"] = sys.version, np.__version__
    json.dump(results, open("experiments.json", "w"), indent=1)
    print("\nwrote experiments.json")
