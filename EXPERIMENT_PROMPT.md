# Second pass: from one number to a curve

Written and executed 2026-09-16, after the first verification pass. Act as a senior performance engineer reviewing a junior's benchmark. The first pass established one point: at 4,096 queries × 8,192 references, the full distance matrix peaks at ~348 MiB and the tiled search at ~130 MiB, with identical answers and no speedup. Three things a reviewer would refuse to sign off without.

## 1. Show the curve, not the point

One size proves nothing about scaling. The full matrix is O(N·M); a tile is O(tile²). Hold queries at 4,096 and sweep references over 8,192 → 16,384 → 32,768 (the dataset has 58,509 rows; 32,768 + 4,096 fit). Measure peak process working set and compute time for both methods, three fresh processes each, medians. The prediction to test: full-matrix peak grows by ~256 MiB per doubling; tiled peak stays at the data-loading floor. Report the measured numbers whether or not they match.

## 2. Justify the tile size

256 was inherited, not chosen. Sweep tile ∈ {32, 64, 128, 256, 512, 1024, 4096} at the 8,192-reference size. Every tile size must reproduce the same output hashes; report time and peak memory. If 256 is not the best, say which is and by how much. If the curve is flat, say tiling granularity does not matter on this CPU and why (per-call overhead vs cache).

## 3. Separate the kernel from the tiling

"No speedup" is a statement about `scipy.spatial.distance.cdist`, which both methods call. Tiling is about *where* distances are computed, not *how*. Try the same tile loop with a BLAS kernel — ‖q‖² + ‖r‖² − 2·q·rᵀ via a matrix multiply — and measure time, memory and agreement with the exact answer. The expansion form is numerically weaker (catastrophic cancellation for near-identical rows), so report the agreement count and the largest distance discrepancy, and if any index differs, say so. Frame any speed gain honestly: it comes from the kernel, not from the tiling.

## Rules

- Every number in the README must be one that printed in `experiments_output.txt` or `experiments.json`.
- Fresh Windows process per measurement (`peak_wset`), same pinned environment as `results.json`.
- Keep `main.py` untouched at 49 lines. Everything here lives in `experiments.py`.
- A prediction that fails is reported as a failed prediction, not reworded.
- Run the `evaluation-auditor` on the result before commit; push only if it passes.
