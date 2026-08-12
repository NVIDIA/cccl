# SASS Comparison Scripts

This directory holds the scripts that `.github/workflows/sass-diff.yml` uses to
compare the generated machine code of the CUB benchmarks before and after the
change in a pull request.

The purpose is to tell the PR author when a benchmark run is necessary. A SASS
change does not always change the performance, but it is the signal that the
generated code is not the same.

A SASS change fails the `SASS Diff` job, so that it is visible in the job list.
That job is intentionally left out of the aggregate `ci` job that branch
protection reads, so the check never blocks a merge.

## When the comparison runs

`ci-workflow-pull-request.yml` compares the SASS only when `ci/inspect_changes.py`
marks CUB as dirty. That covers a change to CUB itself and a change to anything
CUB is built from, because `cub_public` lists the libcudacxx and Thrust headers
as dependencies, and any file that belongs to no project marks every project
dirty. A docs-only or CI-only change therefore starts no comparison.

`[skip-sass-diff]` in the commit summary stops a comparison that would otherwise
run. Like every other skip tag, it blocks the merge until it is removed.

## Scripts

- `ci/sass/sass_diff.sh`: adds a worktree for each ref, builds the selected
  benchmark targets in both, dumps the disassembly with `cuobjdump -sass`, and
  calls `compare_sass.py`. Runs inside the devcontainer. The builds go through
  `ci/build_common.sh`, so they get the same sccache, memory-monitor and timeout
  handling as every other CI build.
- `ci/sass/compare_sass.py`: normalizes the dumps and compares them per target
  and per architecture. Writes `report.json`, the normalized text of both sides
  and the unified diff of every changed architecture. Exits 1 when the SASS
  changed, so that the job fails.
- `ci/sass/render_report.py`: turns `report.json` into the markdown fragment for
  the PR comment: the changed targets, an excerpt of the diff of each, and the
  instructions for requesting a benchmark run.
- `ci/sass/parse_matrix.py`: parses the `sass:` section of `ci/matrix.yaml` and
  emits a dispatch matrix. Runs on the GitHub Actions runner, before any
  container starts.
- `ci/sass/test_compare_sass.py` and `ci/sass/test_render_report.py`: the tests.

No GPU is necessary. The benchmarks are compiled and disassembled, not run.

## Usage

Compare the SASS of all CUB benchmarks between two refs:

```bash
./ci/sass/sass_diff.sh "origin/main" "HEAD" -arch "all-major-cccl"
```

An explicit architecture list also works:

```bash
./ci/sass/sass_diff.sh "origin/main" "HEAD" -arch "75;80;90;100;120"
```

Compare one family of benchmarks only:

```bash
./ci/sass/sass_diff.sh "origin/main" "HEAD" \
  -arch "90" \
  -target-filter "^cub\\.bench\\.reduce\\."
```

Add `-render` to write `result/summary.md` and print it. This is the same
markdown that CI puts in the PR comment, but the links in it are not usable,
because a local run has no artifact to link to:

```bash
./ci/sass/sass_diff.sh "origin/main" "HEAD" -arch "90" -render
```

Every option that `ci/build_common.sh` accepts is also accepted, so `-cxx`,
`-std`, `-cuda`, `-cmake-options` and `-configure` work the same way as they do
for `ci/build_cub.sh`.

Compare two directories of dumps by hand. Normalizing is part of comparing, so
the normalized text and the diffs are a by-product of a run, not a separate
step:

```bash
./ci/sass/compare_sass.py --base-dir base/ --test-dir test/ --output-dir result/

cat result/diff/cub.bench.reduce.base.sm_90.diff
```

Add `--verbose` to print each target as it completes, and a count at the end.

Render an existing result by hand. Each input and the output is named, so a
report can be rendered from any location:

```bash
./ci/sass/render_report.py \
  --report result/report.json \
  --meta result/meta.json \
  --output result/summary.md
```

Run the tests:

```bash
python3 -m pytest ci/sass/
```

## What the comparison ignores

`cuobjdump -sass` prints data that is not part of the generated code and that
changes when unrelated code moves. The normalizer removes:

- the instruction addresses (`/*0a30*/`),
- the encoded instruction words (`/* 0x000fe200078e0203 */`), which only repeat
  the instruction that was already disassembled,
- the container metadata (`code for sm_90`, `.target`, `.headerflags`),
- the `NOP` padding that aligns the end of each kernel,
- the order in which the kernels are emitted.

The dump goes through `cu++filt` before it is written, so every kernel name is
demangled. That also removes the path hash and the compiler pid that nvcc puts
in the name of an internal-linkage or anonymous-namespace entity. Both come from
the build location and the compiler process, thus both differ between the two
worktrees, and without demangling every such kernel would compare as changed.

Branch targets are kept, but as a signed delta from the address of the branch
itself. `BRA 0xd40` at address `0x050` becomes `BRA <+0xcf0>`. A block of code
that only moves therefore compares equal, while a branch that starts to point
somewhere else does not.

Everything that describes the executed code is compared: opcodes, modifiers,
predicates, register operands, constant-bank offsets, immediates and the
control flow structure.

## What the comparison reports

Every target and architecture pair is reported as changed or unchanged. A
changed pair also gets a unified diff of the normalized text. The whole diff
goes to the artifacts, and the first 40 lines of it go into the PR comment, so
that the reader sees what changed without an artifact download.

The comment shows the diff of at most 10 targets, and of one architecture per
target, because a header change can touch every target on every architecture
and a GitHub comment holds 65536 characters. It says how many diffs it left out.

Only the diffs, the report and the metadata are uploaded. The disassembly
itself is not: a run writes three complete copies of it, and over 82 targets
and six architectures that reaches several GB. To read the SASS itself, run
`sass_diff.sh` and open the local artifact directory, which keeps every file.

## Configuration

The build configuration is in the `sass:` section of `ci/matrix.yaml`, so that
it can be trimmed without a change to any script:

```yaml
sass:
  pull_request:
    - id: cub-benchmarks
      gpu: h100
      ctk: '13.X'
      cxx: 'gcc13'
      preset: cub-benchmark
      archs: all-major-cccl
      target_filters: ["^cub\\.bench\\."]
```

`ctk` and `cxx` take the same values as every other job in the file, and
`parse_matrix.py` resolves them the same way: `ctk` through the `ctk_versions`
aliases, so `13.X` follows the newest CTK, and `cxx` through `host_compilers`,
which is what turns `clang19` into the `llvm19` image.

The `cub-benchmark` preset sets `CMAKE_CUDA_ARCHITECTURES` to `native`, so
`archs` must be set for a comparison over several architectures.

`archs` accepts an explicit list (`75;80;90;100;120`) or one of the special
CCCL values that `cmake/CCCLCheckCudaArchitectures.cmake` expands from the
installed nvcc: `all-major-cccl`, `all-cccl`, or `native`. `all-major-cccl` is
the default that PR builds use, so the architecture list follows each new CTK
release without an edit here. On CTK 13.3 it expands to
`75;80;90;100;110;120`.

## Artifacts

`sass_diff.sh` writes the following under its output directory:

- `base/<target>.sass` and `test/<target>.sass`: the raw disassembly of each
  side. **Local only**, see below,
- `result/base/<target>.<arch>.sass` and `result/test/...`: the normalized text
  that the comparison acted on. **Local only**, see below,
- `result/diff/<target>.<arch>.diff`: the whole unified diff of every changed
  architecture. This is what the PR comment excerpts and links to,
- `result/report.json`: the per-target and per-architecture comparison result,
- `result/meta.json`: the refs that were compared and the architectures that the
  build used,
- `result/summary.md`: the markdown fragment for the PR comment. Written only
  with `-render`, or by a call to `render_report.py`.

The four disassembly directories stay on the machine that ran the comparison.
The CI job excludes them from the artifact, because they are three complete
copies of the SASS of every target on every architecture, and together they
reach several GB per run. The diffs, the report and the metadata are a few MB.

The configure and build output goes to stdout. Redirect it if a file is
wanted:

```bash
./ci/sass/sass_diff.sh "origin/main" "HEAD" 2>&1 | tee sass-diff.log
```
