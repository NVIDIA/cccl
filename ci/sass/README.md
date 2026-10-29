# SASS Comparison Scripts

This directory holds the scripts that `.github/workflows/sass-diff.yml` uses to
compare the generated machine code of the benchmarks of a project before and
after the change in a pull request. The `sass:` section of `ci/matrix.yaml`
lists the projects.

The purpose is to tell the PR author when a benchmark run is necessary. A SASS
change does not always change the performance, but it is the signal that the
generated code is not the same.

A SASS change does not fail the `SASS Diff` job. It is a result, and the PR
comment reports it. A build, dump or comparison failure fails the job, and the
aggregate `ci` job that branch protection reads then blocks the merge.

## When the comparison runs

`ci-workflow-pull-request.yml` runs `ci/inspect_changes.py` and passes every
project in its `FULL_BUILD` and `LITE_BUILD` outputs to `parse_matrix.py`. Each
`sass:` entry whose `project` is in that list becomes one matrix job, and the
jobs run in parallel. A project is dirty when it changed, or when something it
is built from changed, through the dependencies in
`ci/project_files_and_dependencies.yaml`. Any file that belongs to no project
marks every project dirty. A docs-only or CI-only change therefore starts no
comparison.

`[skip-sass-diff]` in the commit summary stops every comparison that would
otherwise run. Like every other skip tag, it blocks the merge until it is
removed.

## Scripts

- `ci/sass/sass_diff.sh`: compares the benchmarks of one project. It adds a
  worktree for each ref, builds the selected benchmark targets in both, dumps
  their CUDA object files with `cuobjdump -sass`, and calls `compare_sass.py`.
  Runs inside the devcontainer. The builds go through `ci/build_common.sh`, so
  they get the same sccache, memory-monitor and timeout handling as every other
  CI build.
- `ci/sass/compare_sass.py`: normalizes the dumps and compares them per target
  and per architecture. Writes `report.json`, the normalized text of both sides
  and the unified diff of every changed architecture. Exits 1 when the SASS
  changed. `sass_diff.sh` reports that status as a result, not as a failure.
- `ci/sass/render_report.py`: turns the results of one or more `sass_diff.sh`
  runs into one PR comment: the changed targets, an excerpt of the diff of each,
  and the instructions for requesting a benchmark run. Each `--input` is a
  directory with `report.json`, `meta.json` and, when the triage succeeded,
  `analysis.json`. It reads the project names from the `projects:` section of
  `ci/matrix.yaml`.
- `ci/sass/sass_triage_prompt.md`, `ci/sass/sass_triage_config.toml` and
  `ci/sass/model-output.schema.json`: the prompt, the sandbox and the output
  contract of the triage. Only `sass-diff.yml` uses them.
- `ci/sass/parse_matrix.py`: parses the `sass:` section of `ci/matrix.yaml` and
  emits a dispatch matrix with the entries of the projects given to
  `--projects`. Runs on the GitHub Actions runner, before any container starts.
- `ci/sass/test_compare_sass.py`, `ci/sass/test_parse_matrix.py` and
  `ci/sass/test_render_report.py`: the tests.

No GPU is necessary. The benchmarks are compiled and disassembled, not run.

## Usage

The first argument selects the project: `cub` or `thrust`. One run compares
the benchmarks of that project only. The project also selects the default
preset (`<project>-benchmark`) and the default target filter
(`^<project>\.bench\.`).

Compare the SASS of all benchmarks of a project between two refs:

```bash
./ci/sass/sass_diff.sh cub "origin/main" "HEAD" -arch "all-major-cccl"
./ci/sass/sass_diff.sh thrust "origin/main" "HEAD" -arch "all-major-cccl"
```

An explicit architecture list also works:

```bash
./ci/sass/sass_diff.sh cub "origin/main" "HEAD" -arch "75;80;90;100;120"
```

Compare one family of benchmarks only:

```bash
./ci/sass/sass_diff.sh cub "origin/main" "HEAD" \
  -arch "90" \
  -target-filter "^cub\\.bench\\.reduce\\."
```

Add `-render` to write `result/summary.md` and print it. This is the markdown
that CI puts in the PR comment for this one project. The diff links point to the
local `result/diff` directory, because a local run has no artifact to link to:

```bash
./ci/sass/sass_diff.sh cub "origin/main" "HEAD" -arch "90" -render
```

Every option that `ci/build_common.sh` accepts is also accepted, so `-cxx`,
`-std`, `-cuda`, `-cmake-options` and `-configure` work the same way as they do
for the other `ci/build_*.sh` scripts.

Compare two directories of dumps by hand. Normalizing is part of comparing, so
the normalized text and the diffs are a by-product of a run, not a separate
step:

```bash
./ci/sass/compare_sass.py --base-dir base/ --test-dir test/ --output-dir result/

cat result/diff/cub.bench.reduce.base.sm_90.diff
```

Add `--verbose` to print each target as it completes, and a count at the end.

Render existing results by hand, one `--input` for each project:

```bash
./ci/sass/render_report.py \
  --input cub-artifacts/result \
  --input thrust-artifacts/result \
  --output summary.md
```

Run the tests:

```bash
python3 -m pytest ci/sass/
```

## What the comparison ignores

The comparison uses the object files from each benchmark translation unit.
This excludes kernels from linked libraries such as `nvbench_helper`.
The benchmark build of each project generates
`<project>/benchmarks/objects/<target>.objects` in the build directory, with one
object path per line, for each CUDA benchmark target.

`sass_diff.sh` copies `<project>/benchmarks/CMakeLists.txt` from the current
checkout into both worktrees, so an older ref also generates these lists, and
both sides build with the same benchmark rules. The benchmark sources and the
library headers still come from each ref. If the current `CMakeLists.txt` does
not fit the sources of an older ref, the configuration fails.

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

The comment shows the diff of at most 10 targets over all projects, and of one
architecture per target, because a header change can touch every target on
every architecture and a GitHub comment holds 65536 characters. It says how
many diffs it left out.

## The PR comment

Each matrix job uploads its `report.json`, its `meta.json` and, when the triage
succeeded, its `analysis.json` as the `sass-<id>-comment` artifact. The job adds
the URL of its dumps artifact to `meta.json` before the upload.

The `sass-results` job waits for every matrix job and calls `render_report.py`
one time with all the artifacts. The result is one comment with one heading, one
benchmark request block and one table of changed targets for all projects. The
job posts the comment when any project has a SASS change. When no project has a
change, it hides the previous comment.

Only the diffs, the report and the metadata are uploaded. The disassembly
itself is not: a run writes three complete copies of it, and over 82 targets
and six architectures that reaches several GB. To read the SASS itself, run
`sass_diff.sh` and open the local artifact directory, which keeps every file.

## Triage of the differences

In CI, a model reads `report.json` and the diffs and groups them by what
changed. Each group is `benign` (the executed work is the same: register
numbers, instruction order, offsets), `significant` (the executed work differs:
the instruction mix, the control flow, the memory operations), or `unclear`. The
comment shows them as ✅ Potentially Benign, ⚠️ Significant and ❓ Unclear.

Each diff block carries the classification of its own target in the header, and
the model's description inside the box, so the estimate sits beside the evidence
for it. The classification also decides which targets keep a block: the comment
holds 10, and the significant ones come first. Without the triage the comment
shows the first 10 targets it walked, so a header change can hide every real
difference behind the noise.

The triage says what the diffs contain. It does not recommend an action and it
does not predict a performance result. The author decides whether a benchmark
run is necessary.

`ci/sass/sass_triage_config.toml` gives the model read access to the result
directory and nothing else: no network, no write access.

The triage gates nothing, and no model output can stop the comparison result
from reaching the pull request. The step is `continue-on-error`, so a model
outage or a missing API key removes the classifications and leaves the rest of
the comment as it is.

`codex-action` holds the answer to `ci/sass/model-output.schema.json`, and
`render_report.py` checks it again, because it reads a file and not a promise. A
group with an unknown classification, an empty title or explanation, or a bad
diff list becomes ❓ Unclear, and the block says why. So does a difference that
the model left out of every group, and every difference when the whole file is
unreadable. One bad group costs its own diffs and no others; the warnings go to
the job log.

The model writes the title and the explanation, and a diff holds whatever the
compiler emitted. Thus the renderer treats that text as data: it removes the
control characters, escapes the HTML and the inline markdown, and cuts it to
length. The comment is also kept under the GitHub size limit, and a block that
does not fit is dropped.

A diff without a classification sorts with the ❓ Unclear ones. This happens
when the triage of one project failed and the triage of another succeeded.

To render an analysis by hand, put it in the input directory as
`analysis.json`.

## Configuration

The build configuration is in the `sass:` section of `ci/matrix.yaml`, so that
it can be trimmed without a change to any script:

```yaml
sass:
  pull_request:
    - id: cub-benchmarks
      project: cub
      ctk: '13.X'
      cxx: 'gcc13'
      preset: cub-benchmark
      archs: all-major-cccl
      target_filters: ["^cub\\.bench\\."]
    - id: thrust-benchmarks
      project: thrust
      ctk: '13.X'
      cxx: 'gcc13'
      preset: thrust-benchmark
      archs: all-major-cccl
      target_filters: ["^thrust\\.bench\\."]
```

`id` names the artifacts and the build directory, so it must be unique.
`project` is the first argument of `sass_diff.sh`, and it must match a name in
the `FULL_BUILD` and `LITE_BUILD` outputs of `ci/inspect_changes.py` and a key
in the `projects:` section of the same file.

`ctk` and `cxx` take the same values as every other job in the file, and
`parse_matrix.py` resolves them the same way: `ctk` through the `ctk_versions`
aliases, so `13.X` follows the newest CTK, and `cxx` through `host_compilers`,
which is what turns `clang19` into the `llvm19` image.

The `<project>-benchmark` presets set `CMAKE_CUDA_ARCHITECTURES` to `native`,
so `archs` must be set for a comparison over several architectures.

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
- `result/meta.json`: the project, the refs that were compared, the
  architectures that the build used, and the URL that the diff links point to,
- `result/summary.md`: the PR comment for this project. Written only with
  `-render`.

The four disassembly directories stay on the machine that ran the comparison.
The CI job excludes them from the artifact, because they are three complete
copies of the SASS of every target on every architecture, and together they
reach several GB per run. The diffs, the report and the metadata are a few MB.

The configure and build output goes to stdout. Redirect it if a file is
wanted:

```bash
./ci/sass/sass_diff.sh cub "origin/main" "HEAD" 2>&1 | tee sass-diff.log
```
