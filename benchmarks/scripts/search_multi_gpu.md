# Multi-GPU search for `search.py`

## Motivation

An evaluation is a CPU-bound build followed by a GPU-bound benchmark, run back to back.
Measured on a 2x RTX PRO 6000 / 32-core host, benchmark timed over the full runtime workload
space for one compile-time workload, as `Bench.do_run` forms it:

| | build (1 core) | benchmark (1 GPU) | GPU busy |
|---|---|---|---|
| `cub.bench.reduce.sum` | 7.8s | 1.14s | 13% |
| `cub.bench.radix_sort.pairs` | 26.6s | 14.1s | 35% |

Running one evaluation per GPU is a flat 2x that leaves that idle time untouched, because a
worker still alternates build (GPU idle) and benchmark (CPU idle). The goal is variants
covered per hour, so the serialization of build against benchmark matters more than the GPU
count does:

| | `reduce.sum` | `radix_sort.pairs` |
|---|---|---|
| today (1 GPU, serial) | 403 evals/h | 88 evals/h |
| one worker per GPU | 806 evals/h | 177 evals/h |
| pipelined, K=4 | ~3200 evals/h | ~511 evals/h |

Two GPUs are not the binding constraint; 32 cores sitting idle behind a serialized build are.

## Layout

`B = num_gpus * K` build directories, two levels deep so the GPU a directory belongs to is
readable off the path. `K` is the number of lanes per GPU - the pipelining depth that lets
one evaluation build while another benchmarks on the same GPU.

```
build/gpu0/lane0   build/gpu0/lane1   build/gpu0/.lock
build/gpu1/lane0   build/gpu1/lane1   build/gpu1/.lock
```

Worker `i` owns `gpu = i % num_gpus`, `lane = i // num_gpus`, and the directory
`build/gpu<gpu>/lane<lane>`; all three are fixed for the worker's lifetime.

## Processes

```
search.py ── compileiq Search
   │              └── core            subprocess, proposes each generation's pool
   │
   ├── worker 0   gpu0/lane0          the parent itself is a worker
   ├── worker 1   gpu1/lane0
   ├── worker 2   gpu0/lane1
   └── worker 3   gpu1/lane1          B = num_gpus * K processes, pulling from one job queue

worker i, once per evaluation:

   spawn ─> shim   cwd=build/gpu<g>/lane<l>, CUDA_VISIBLE_DEVICES=<g>
                 ├── cmake --build .            CPU, unlocked
                 ├── flock(build/gpu<g>/.lock)
                 ├── bin/<variant>              GPU, exclusive
                 └── release, print score, exit
```

A generation's pool goes onto one queue and the `B` workers drain it, so the spread is within
a generation, not across generations. Each worker is pinned to its GPU and directory for its
whole life; the only shared resource is the GPU, held exclusively but only across the
benchmark. Builds proceed freely, so throughput becomes GPU-bound - once builds are hidden,
build time stops mattering:

```
gpu0/lane0   [--build--][run]          [--build--][run]
gpu0/lane1        [--build--]   [run]       [--build--]   [run]
gpu0 (lock)                [run]  [run]              [run]  [run]
```

**Worker identity.** compileiq spawns anonymous processes - `MultiProcessWorker.run` builds
`num_workers - 1` identical `Process` objects with identical args, plus the parent as the last
worker, and is called once per generation with a fresh pool. So the id is minted on our side:
a `Worker` subclass overrides `run` to refill an id queue with `1..B-1` before delegating to
`super().run()`; the parent permanently holds id 0. Each process caches its id keyed by
`os.getpid()`, so a forked child notices it inherited the parent's id and draws its own. No
per-evaluation negotiation.

**Cache first.** The shim asks for the score before it builds anything: `Bench.is_cached` checks
whether the stored results already cover this variant and its base, and on a hit `score` reads
them straight out of the lane's database - no compile, no GPU, no lock. This matters because the
search re-proposes survivors every generation: one measured run scored 194 distinct variants in
724 evaluations, and the winner alone was proposed 143 times. A repeat costs 0.75s instead of
9.1s. Scope is the lane's own database, which covers 49% of the repeats; sharing one database
across lanes would reach 73%, at the price of eight writers on one sqlite file.

**Evaluation.** The shim is a fresh process per evaluation, spawned with `Popen` - not
required, since the binding is static and a worker could equally `chdir` once and stay
in-process. It buys crash isolation (a hung build or leaked handle costs one evaluation, not
the lane for the rest of the run), bounds memory (`BenchCache` and `RunsCache` accumulate
every result otherwise), and avoids resetting the six `_instance` singletons that `cccl.bench`
caches against a cwd-relative layout (`cmake --build .`, `./bin/<exe>`, `result.json`, the db).
Cost is ~1.2s of cold start (0.25s import, ~0.5s per `--jsonlist-*` call), which lands on the
build side, behind the GPU lock, so it is free in throughput terms.

**Provisioning.** Run from the CCCL source root. The script splits `-D*` args from search args,
configures every lane with the tuning preset, and drives the search from `build/gpu0/lane0`.
Each lane keeps its own `cccl_meta_bench.db`, so a variant is always compared against a
baseline measured in the same lane on the same physical GPU - stricter than the per-GPU
database `docs/cub/tuning_infra.rst` requires, and it keeps lanes off a shared sqlite file.

## Knobs and risks

- `K` defaults to 4 and is overridable. The build:benchmark ratio ranges from 7:1 to 2:1
  across algorithms, so no single value is optimal everywhere; 4 hides the build for heavy
  algorithms and gets light ones most of the way. K=1 reproduces one-worker-per-GPU for an
  apples-to-apples baseline.
- Over-provisioning `K` is self-limiting: surplus workers block on the GPU lock and burn a
  process, not a core. Under-provisioning leaves the GPU idle. Err high.
- With K>1 something is always compiling while something is measuring. Each variant is
  compared to a baseline in its own lane, so the bias is largely common-mode; `taskset` on the
  build is the mitigation if it proves to matter.
- Each lane pays its own base build and ~816 MB of disk, so setup costs minutes and a few GB
  before the first variant is scored.

## Measured

One full search of `cub.bench.reduce.sum` (`T{ct}=I32`, `OffsetT{ct}=I32`), 2x RTX PRO 6000, 32 cores:

| | before | after |
|---|---|---|
| wall | 100.3 min | 22.75 min |
| evaluations | 740 | 699 |
| per evaluation | 8.13s | 1.95s |
| GPU split | one GPU | 353 / 346 |

4.16x, short of the 8x the lane count suggests because concurrent builds slow each other down -
average variant build went from 7.62s serial to 9.22s under 8 lanes, with a worst case of 23.1s.

Both runs predate the cache-first change, which was measured separately at 9.12s to 0.75s per
repeat and should remove roughly half the remaining builds.
