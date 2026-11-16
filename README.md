# RoleWeaver

This repository contains the implementation of **RoleWeaver**, including separate index construction and search executables for role-based access control (RBAC) in approximate nearest neighbor search (ANNS).

## Build

A C++17 compiler and a POSIX environment with pthreads are required.

```bash
bash scripts/build.sh
```

The build enables CPU-specific optimizations by default and produces two executables in `build/`:

- `roleweaver_build`: optimizes an index plan from role assignments, builds the indexes, and saves the artifacts and construction statistics.
- `roleweaver_search`: loads built indexes and runs authorized nearest-neighbor queries, reporting throughput and recall.

## Input

| File | Format |
| --- | --- |
| `base.fbin` | Two little-endian int32 values for the vector count and dimension, followed by row-major float32 coordinates. Vector labels are zero-based row IDs. |
| `queries.fbin` | The same binary format as `base.fbin`, with a matching dimension. |
| `role_blocks.txt` | One row per role: `role_id label_1 label_2 ...`. Role IDs must be consecutive integers starting at 0, and labels must refer to valid base vectors. An empty role contains only its ID. |
| `query_roles.txt` | One row per query: `query_id user_id topk role_count role_id_1 ...`. Rows must follow query-vector order. Query and user IDs are metadata; `role_count` must match the number of listed roles. |
| Ground truth (optional) | One row per query: `query_id label_count label_1 ...`, in query-vector order. Each row lists the exact authorized nearest neighbors for the evaluated k. If fewer than k vectors are authorized, the ground-truth row contains only the available neighbors. |

## Usage

**Run from the repository root.** Replace the input paths below.

First, optimize the plan and build the indexes:

```bash
build/roleweaver_build \
  --base data/base.fbin --roles data/role_blocks.txt \
  --index-dir output/run/index \
  --rho 0.3 --M 16 --solver greedy \
  --efc 100 --build-threads 40
```

Then, load the built indexes and run queries:

```bash
build/roleweaver_search \
  --base data/base.fbin --index-dir output/run/index \
  --queries data/queries.fbin --query-roles data/query_roles.txt \
  --tau 4096 --efs '20 40 80 160 320' \
  --k 10 --query-repeats 3 --out-csv output/run/search.csv
```

Use `--groundtruth data/truth.txt` to load prepared ground truth. If the file is absent or fails format, search will generate exact authorized ground truth automatically. Without `--groundtruth`, it generates ground truth directly. Add `--exact-truth-out output/run/truth.txt` to save the ground truth used for evaluation.

Each search width runs `--query-repeats` times; results go to `--out-csv` when specified, or standard output otherwise.

## Output

| Output | Contents |
| --- | --- |
| `output/run/index/plan.txt` | Index plan used for construction and subsequent loading. |
| `output/run/index/index.bin` | Serialized indexes, excluding base vectors. |
| `output/run/index/manifest.txt` | Versioned artifact format, construction settings, vector dimensions, and checksums binding the vectors, plan, and indexes. Written only after construction succeeds. |
| `output/run/index/optimizer_stats.json` | Solver status, objective, selected roles, and paths. Inspect `status` and `exact_finished` to check completion. |
| `output/run/index/build_stats.json` | Planning, index construction, and total construction times. |
| `output/run/search.csv` | One row per search width, without a header: `ef_s,qps,recall`. Throughput and recall are averaged across repeats. |
| `output/run/truth.txt` | Optional ground-truth file saved with `--exact-truth-out`, in the format described under Input. |
| `output/run/diagnostics.csv` | Optional CSV saved with `--query-diagnostics-csv`, containing per-query authorization and index-view statistics. |

## Parameters

The tables below list all command-line options. Defaults apply when an option is omitted; required paths have no default. See Usage for command examples, Input for file formats, and Output for generated artifacts.

| Build parameter | Default | Description |
| --- | --- | --- |
| `--base PATH` | Required | Base-vector input file in `.fbin` format. |
| `--index-dir PATH` | Required | Output directory for the plan, indexes, manifest, and construction statistics. Created automatically; existing build artifacts are overwritten with a notice. |
| `--roles PATH` | Required | Role-assignment input file used to optimize the plan for this build. |
| `--rho` | 0.3 | Index-reuse threshold in [0, 1]. Larger values restrict reuse. |
| `--M` | 16 | HNSW connectivity shared by planning and index construction. |
| `--solver` | `greedy` | Planning solver: `branch-cut` or `greedy`. |
| `--efc` | 100 | HNSW construction search width. |
| `--build-threads` | Hardware concurrency | Number of index-construction workers. |
| `--help` | — | Print command-line usage and exit. |

| Search parameter | Default | Description |
| --- | --- | --- |
| `--base PATH` | Required | Original base-vector input file. Its contents must match the vectors used to build the indexes. |
| `--index-dir PATH` | Required | Existing index directory produced by `roleweaver_build`. |
| `--queries PATH` | Required | Query-vector input file in `.fbin` format, with the same dimension as the base vectors. |
| `--query-roles PATH` | Required | Query-role assignment file in query-vector order. |
| `--groundtruth PATH` | Generate automatically | Prepared ground-truth input file. If missing or invalid, generate exact authorized ground truth automatically. |
| `--out-csv PATH` | Standard output | Output CSV for search width, throughput, and recall. |
| `--exact-truth-out PATH` | Not saved | Save the ground truth used for evaluation, whether loaded or generated. |
| `--query-diagnostics-csv PATH` | Not saved | Save per-query authorization and index-view statistics as CSV. |
| `--efs` | 100 | One search width or a comma-/space-separated list, e.g. `100` or `'20 40 80'`. |
| `--k` | 10 | Number of requested neighbors; 0 uses `topk` from each query row. |
| `--tau` | 4096 | Use exact scan when the authorized vector count is at most this value. |
| `--query-repeats` | 3 | Number of runs per search width. |
| `--max-queries` | 0 | Maximum number of queries to evaluate; 0 evaluates all queries. |
| `--help` | — | Print command-line usage and exit. |
