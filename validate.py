"""Independent correctness checks and fresh-process benchmarks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
from scipy.spatial.distance import cdist
from main import nearest, load, sample

if len(sys.argv) > 1:
    q, r, y, labels = sample()
    start = time.perf_counter()
    if sys.argv[1] == "full":
        matrix = cdist(q, r)
        ids = matrix.argmin(1)
        distances = matrix[np.arange(len(q)), ids]
    else:
        ids, distances = nearest(q, r)
    elapsed = time.perf_counter() - start
    print(json.dumps(dict(seconds=elapsed,
        peak_process_mib=psutil.Process().memory_info().peak_wset / 2**20,
        index_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        distance_sha256=hashlib.sha256(distances.tobytes()).hexdigest())))
    sys.exit()

rng = np.random.default_rng(7)
checks = 0
for nq, nr, dims in [(17, 23, 5), (1, 1, 3), (0, 4, 2)]:
    q, r = rng.normal(size=(nq, dims)), rng.normal(size=(nr, dims))
    oracle = np.sqrt(((q[:, None] - r[None]) ** 2).sum(2))
    for tile in [1, 4, 32]:
        ids, distances = nearest(q, r, tile)
        np.testing.assert_array_equal(ids, oracle.argmin(1))
        np.testing.assert_allclose(distances, oracle.min(1), rtol=1e-12)
        checks += 1
assert nearest([[0]], [[1], [-1], [1]], 1)[0][0] == 0
checks += 1
for q, r, tile in [([[0]], [], 1), ([[np.nan]], [[1]], 1), ([[0]], [[1, 2]], 1), ([[0]], [[1]], 0)]:
    try:
        nearest(q, r, tile)
        raise AssertionError("Invalid input accepted")
    except ValueError:
        checks += 1
data = load()
assert data.shape == (58509, 49) and np.isfinite(data).all()
q, r, y, labels = sample()
matrix = cdist(q, r)
ids, distances = nearest(q, r)
np.testing.assert_array_equal(ids, matrix.argmin(1))
np.testing.assert_allclose(distances, matrix.min(1), rtol=0, atol=0)
del matrix
runs = {mode: [json.loads(subprocess.check_output([sys.executable, __file__, mode], text=True)) for _ in range(3)] for mode in ["full", "tiled"]}
assert len({v['index_sha256'] for values in runs.values() for v in values}) == 1
assert len({v['distance_sha256'] for values in runs.values() for v in values}) == 1
result = dict(tests=checks, rows=len(data), features=48,
    duplicate_rows=len(data)-len(np.unique(data, axis=0)),
    classes=np.unique(data[:, -1]).astype(int).tolist(),
    archive_sha256=hashlib.sha256(Path('data.zip').read_bytes()).hexdigest(),
    accuracy=float((labels[ids] == y).mean()), exact_matches=len(q),
    full_distance_buffer_mib=256, tile_distance_buffer_mib=0.5,
    runs=runs, python=sys.version, numpy=np.__version__)
Path('results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
