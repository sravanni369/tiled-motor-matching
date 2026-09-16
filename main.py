"""Book-inspired tiled motor-condition matching. Run: python main.py"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
from scipy.spatial.distance import cdist

def nearest(query, reference, tile=256):
    """Exact Euclidean nearest reference; first index wins ties."""
    query, reference = np.asarray(query, float), np.asarray(reference, float)
    if query.ndim != 2 or reference.ndim != 2 or not len(reference):
        raise ValueError("Expected matrices and nonempty references")
    if query.shape[1] != reference.shape[1] or tile < 1:
        raise ValueError("Feature counts must match; tile must be positive")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Inputs must be finite")
    best, ids = np.full(len(query), np.inf), np.zeros(len(query), dtype=int)
    for start in range(0, len(query), tile):
        block = query[start:start + tile]
        for offset in range(0, len(reference), tile):
            distances = cdist(block, reference[offset:offset + tile])
            local = distances.argmin(axis=1)
            values = distances[np.arange(len(block)), local]
            better = values < best[start:start + len(block)]
            best[start:start + len(block)][better] = values[better]
            ids[start:start + len(block)][better] = offset + local[better]
    return ids, best

def load():
    path = Path(__file__).parent / "data.zip"
    if not path.exists():
        urlretrieve("https://archive.ics.uci.edu/static/public/325/dataset%2Bfor%2Bsensorless%2Bdrive%2Bdiagnosis.zip", path)
    with ZipFile(path) as archive:
        return np.loadtxt(archive.open("Sensorless_drive_diagnosis.txt"))

def sample():
    data = np.unique(load(), axis=0)
    data = data[np.random.default_rng(42).permutation(len(data))]
    reference, query = data[:8192], data[8192:12288]
    mean, scale = reference[:, :-1].mean(0), reference[:, :-1].std(0)
    scale[scale == 0] = 1
    return (query[:, :-1] - mean) / scale, (reference[:, :-1] - mean) / scale, query[:, -1], reference[:, -1]

if __name__ == "__main__":
    query, reference, actual, labels = sample()
    ids, distances = nearest(query, reference)
    print(f"Queries: {len(query)} | References: {len(reference)}")
    print(f"Exploratory condition accuracy: {(labels[ids] == actual).mean():.2%}")
    print(f"Distance buffers: full {len(query)*len(reference)*8/2**20:.1f} MiB; tile {256*256*8/2**20:.1f} MiB")
