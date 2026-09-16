# Tiled motor-condition matching in 49 lines

An industrial maintenance analyst needs to find reference motor-current records similar to a new observation. Storing every query-to-reference distance can consume substantial memory. This offline Python prototype computes exact nearest references in blocks and discards each block after reducing it.

**Complete runnable example: [main.py](main.py), 49 physical lines.** Tests and measurement tooling are separate and are not included in that count. NumPy and SciPy provide the numerical kernels.

## Measured result

4,096 query records × 8,192 reference records × 48 features, float64, tile size 256; Windows, Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1. Three fresh processes per method; elapsed times cover distance computation and nearest-match reduction, excluding download and preprocessing.

| Measurement | Full distance matrix | Tiled + immediate reduction |
|---|---:|---:|
| Largest individual distance buffer, calculated | 256 MiB | 0.5 MiB |
| Median peak process working set, measured | 348.48 MiB | 131.51 MiB |
| Median compute time, measured | 0.7794 s | 0.7806 s |
| Nearest-reference answers agreeing with baseline | 4,096 / 4,096 | 4,096 / 4,096 |

![validate.py output in VS Code, 2026-09-16](vscode_run.png)

The individual distance buffer is 512× smaller. Total measured process memory decreased about 62.3%; **512× is not the total-memory improvement**. Python assignment can transiently retain an old tile while constructing the next one. Inputs, loading, scaling, copies, interpreter and library allocations also consume memory. Peak working set covers the entire child process. Where that peak comes from matters: in the tiled process the ~130 MiB peak is reached while loading and deduplicating the table, before `nearest()` runs, and the tiled search never rose above that floor (**+0.00 MiB** in four fresh-process probes). The full-matrix search raised the same process from ~130 MiB to ~347 MiB (**+217 MiB**: the 256 MiB matrix on top of the ~91 MiB the process holds once the loaded table is freed). So the 62% figure depends on how much memory the loader uses on this machine; the search-step difference is 256 MiB of buffer versus no measurable increase. See [extra_checks_output.txt](extra_checks_output.txt). There is no demonstrated speed improvement.

### Reproduction, 2026-09-16

Two re-runs on the same machine and the same pinned environment, after the measurement above:

| Run | Full matrix | Tiled + reduction | Peak-memory reduction |
|---|---:|---:|---:|
| Clean terminal re-run ([results-rerun-2026-09-16.json](results-rerun-2026-09-16.json)) | 347.25 MiB, 0.7829 s | 129.89 MiB, 0.7864 s | 62.6% |
| VS Code task run, captured in [vscode_run.png](vscode_run.png) | 347.02 MiB, 0.8872 s | 129.89 MiB, 0.7322 s | 62.6% |

All six subprocesses in both re-runs produced the same index and distance SHA-256 as the original run. The VS Code run fired while the editor itself was still starting; its first full-matrix timing (3.06 s) is a cold-start outlier, so no timing conclusion is drawn from that run. The memory result and the absence of a speedup held in every run.

## Run the complete example

Use Python 3.12 in a fresh environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The first execution downloads the official UCI archive into `data.zip`. Subsequent runs reuse it. Expected output:

```text
Queries: 4096 | References: 8192
Exploratory condition accuracy: 69.34%
Distance buffers: full 256.0 MiB; tile 0.5 MiB
```

On Windows, run `python validate.py` to regenerate [results.json](results.json). The validator uses Windows `peak_wset`; the minimal example itself is cross-platform. Run the validator from this directory. Allow several hundred MiB of RAM for the full-matrix comparison.

## What the book contributed, and what changed

Source: Chao Wang, *Domain-Specific Computer Architectures for Emerging Applications: Machine Learning and Neural Networks*, CRC Press, 2024. Algorithm 5.2, also labeled “Algorithm 2 Tiled Distance Calculation Algorithm,” printed pp. 136–138; supplied PDF pages 76–77. [Book DOI](https://doi.org/10.1201/9780429355080).

The book describes object and centroid blocks for FPGA data reuse and outputs the full distance array. This independently written CPU implementation applies two-axis tiling to query and reference records. **Immediate nearest-match reduction is an additional adaptation:** tiling alone would not remove the full output array. Partial final tiles and deterministic first-index ties are handled. Work remains O(NMD); retained output is O(N), with bounded tile distance storage. These are CPU measurements, not measurements of FPGA transfers, energy, or acceleration.

## Dataset and evaluation limits

[UCI Dataset for Sensorless Drive Diagnosis](https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis), Bator, M. (2013), [DOI 10.24432/C5VP5F](https://doi.org/10.24432/C5VP5F), CC BY 4.0. Original data are not bundled. The archive SHA-256 is recorded in results.json.

Audit: 58,509 rows, 48 numerical features plus one condition label, 11 classes, all values finite, zero exact duplicate rows. The example sorts/deduplicates rows, shuffles with seed 42, selects 8,192 references and 4,096 disjoint queries, then standardizes using reference means and standard deviations only. The remaining records are unused. Selection is random, not stratified.

Nearest-reference condition-label accuracy is **69.34%**. Baselines on the same split: **majority class 9.38%** (11 near-balanced conditions, so this equals chance) and **nearest centroid 50.76%**. Per-class recall ranges from 55.7% (condition 5) to 99.7% (condition 11); conditions 7 and 11 are near-perfect, the other nine sit between 56% and 75%. Across six shuffle seeds (42, 0–4) 1-NN accuracy spans 69.34%–71.19%, mean 70.39%; seed 42, the published split, is the lowest. All from [extra_checks_output.txt](extra_checks_output.txt). Exact agreement with the full distance algorithm does not mean correct diagnosis. Row-wise splitting may put correlated measurements in both partitions; the downloaded table has no explicit machine/run grouping used here. This is not evidence of generalization to unseen motors or operating runs. Before deployment, use run/machine-separated evaluation, condition-level error analysis, stronger diagnostic baselines and domain validation. No measured downtime reduction, cost saving or autonomous maintenance recommendation is claimed.

## Correctness evidence

14 edge/correctness cases cover an independent direct-subtraction distance oracle, partial tiles, tile sizes 1/4/32, empty queries, singleton inputs, equal-distance ties, nonfinite inputs, mismatched dimensions, empty references and invalid tile size. Additionally, all 4,096 real-data nearest indices and distances match the full SciPy calculation exactly. Output hashes agree across all six benchmark processes. See [validate.py](validate.py) and [results.json](results.json). [extra_checks.py](extra_checks.py) (added 2026-09-16) adds a second oracle using the expansion ‖q‖²+‖r‖²−2q·r (all 4,096 indices agree, distances within 5.3e-14), confirms bit-identical results for tile sizes 37, 1000, 4095, 5000 and 100000 on the real data, confirms zero rows shared between query and reference, and reports the baselines, per-class recall, seed spread and fresh-process memory probes quoted above; its output is in [extra_checks_output.txt](extra_checks_output.txt).

## Review and attribution

See [CLAUDE_REVIEW_PROMPT.md](CLAUDE_REVIEW_PROMPT.md) before publishing. [PROJECT_PROMPT.md](PROJECT_PROMPT.md) records the project specification. No book PDF, credentials, local environments or fabricated execution screenshots should be uploaded. This repository contains an educational adaptation, not the author's original implementation.
