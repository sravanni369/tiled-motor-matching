# Review this project before I upload it

Act as an independent Python and numerical-computing reviewer. Read every included file and actually run the example and validation when your environment permits. Do not assume that the README or saved measurements prove correctness.

1. Check main.py is a complete 49-line runnable example, with no omitted helper. It downloads real UCI data, scales using reference rows only and matches queries to references using two-axis tiles.
2. Independently verify the nearest-index and distance results, tie handling, partial tiles, invalid-input behavior and numerical limitations. Assess whether the tests use a sufficiently independent oracle.
3. Verify dataset provenance, 58,509 rows, 48 features, 11 labels, finite values, exact duplicates, disjoint split and archive SHA-256. Identify row-split leakage and generalization limits.
4. Check all numbers against results.json. Separate largest distance-array size from total peak process memory. Inspect whether the Windows process-lifetime memory measurement and three-run timing comparison support the claims. Do not claim speedup or industrial savings.
5. Check the attribution and distinguish the book's full-distance FPGA pseudocode from this CPU nearest-reference reduction adaptation. The book PDF is not included; report citation checks you cannot independently verify.
6. Inspect infographic/caption if included and flag any unsupported headline, missing limitation or misquoted metric. Never treat exact algorithm agreement as diagnostic accuracy.
7. Return findings by severity with file/line, reproduction and suggested minimal fix. List which checks actually ran and which could not run. Recommend publish only after any material issues are resolved.

Keep the complete example short and readable; place expanded tests and benchmark tooling in separate files. Do not upload anything on my behalf.
