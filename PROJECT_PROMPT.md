# Book algorithm to a verified industrial prototype

Build an independently written Python adaptation of Algorithm 5.2, “Pseudocode for Distance Computation after Refactoring Using Tiling Technique” (also labeled “Algorithm 2 Tiled Distance Calculation Algorithm”), in Chao Wang, *Domain-Specific Computer Architectures for Emerging Applications: Machine Learning and Neural Networks*, CRC Press, 2024, printed pp. 136–138, PDF pages 76–77, DOI 10.1201/9780429355080.

## Business question
Can an industrial maintenance analyst match motor-current feature records to reference operating-condition examples without allocating the complete query-by-reference distance matrix? Use the UCI Sensorless Drive Diagnosis dataset, credit Martyna Bator and its CC BY 4.0 license. This is an offline triage prototype, not an autonomous maintenance decision or proven downtime reduction.

## Work and evidence
Deliver the full runnable example in at most 50 physical lines, including data download and output. Keep rigorous tests and benchmarks in separate files. Create a mobile-readable LinkedIn infographic with small colorful motor, current-signal, tile-grid and matching illustrations; include the source book and only verified results. Execute this prompt, rather than delivering instructions alone.
1. Inventory every PDF in the supplied books folder and search available text for algorithm/code candidates. Report the actual search coverage and extraction failures. Read this algorithm and its surrounding discussion in depth. Inspect prior LinkedIn prompts and posts for style, not as authority to invent results.
2. Check local datasets first. If no suitable motor data exists, download from UCI with URL, SHA-256, row/column audit, finite-value and duplicate checks. Never publish the supplied book PDF.
3. Independently implement a readable core of no more than 50 physical lines, including imports/comments. Keep download, audit, tests and benchmarking outside that core, and disclose total project scope.
4. Preserve two-axis tiling and handle partial final tiles. Extend the book’s distance-output design by reducing each tile immediately to the nearest reference index and distance. Explain this extension: merely tiling while retaining every distance does not remove quadratic output storage.
5. Compare against SciPy’s full squared-Euclidean distance matrix and an independent small direct-subtraction oracle. Test non-divisible dimensions, tile sizes of one and larger than inputs, duplicates/ties, empty queries, invalid sizes, NaN/Inf and shape errors. Specify tie behavior and floating-point tolerances.
6. Deduplicate before splitting. Fit scaling on reference data only. Fix data split and seeds. Report a majority baseline and diagnostic classification metrics, noting the lack of run/motor group IDs and the inability of a row split to prove transfer to new machines.
7. Measure wall time and peak process memory in fresh subprocesses for full and tiled implementations. Report repeated-run medians and ranges. Distinguish actual process memory from analytically computed distance-array bytes. Do not promise that tiling is faster. Do not extrapolate CPU measurements to FPGA memory traffic or energy.
8. Rerun deterministic outputs and validate every published figure. Capture a real application/terminal screenshot only after checks pass. An infographic is not an execution screenshot.
9. Create README, provenance, limitations, exact run instructions, pinned environment, tests, results and screenshots. Publish a new public GitHub repository only after validation and a secret/content review. Do not overwrite existing work.
10. Create a readable recruiter-facing infographic and LinkedIn caption: real problem, book citation, what was adapted, measured results, limitations and repository. Say “adapted the book’s pseudocode” rather than implying the author supplied runnable Python or that FPGA tests occurred. Use measured evidence, not job guarantees. Prepare the post; do not publish it to LinkedIn.

## Acceptance gate
No fabricated accuracy, savings, work experience, hardware claims or screenshots. If a baseline wins, report it. A valid negative result is publishable; a failed correctness check is not.
