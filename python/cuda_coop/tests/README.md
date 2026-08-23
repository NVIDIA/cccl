# cuda.coop Test Suite

The suite is organized by ownership and evidence, not by when an implementation
was added. A file's path should answer three questions without opening it:

1. Which semantic contract or backend owns the test?
2. Is it a unit, compile, runtime, link, or qualification test?
3. Does it belong in ordinary pull-request coverage?

## Layout

```text
tests/
  contracts/                 backend-free API and semantic contracts
    coverage.toml            capability, parity, and evidence manifest
    core/                    shared normalization and planning
    public_api/              exports, signatures, and diagnostics
    parity/                  cross-backend semantic obligations
  backends/
    <backend>/
      unit/                  frontend and adapter behavior
      compile/               lowering and compilation without launch
      runtime/               representative GPU execution
      stress/                broad matrices, where needed
  providers/                 provider ABI, link, and qualification
    cutlass/                 CUTLASS provider final-link and activation checks
    qualification/           scheduled final-link qualification suites
  integration/               examples, compiler, and packaging
  support/                   shared cases, oracles, fixtures, and paths
```

Only occupied directories exist. Public backend names define directories:
`numba_mlir` and `cutlass`. Historical implementation phases such as `cute`, `deferred`, `single_phase`,
or `alpha_validation` do not.

A test module owns one backend, primitive family, and evidence layer. Shared
data and helpers belong in `tests/support`; backend-specific helpers stay with
that backend. Under pytest's importlib mode, use package-relative imports. Do
not mutate `sys.path` to make a helper importable.

## Conformance Evidence Contract

`support/cases/api_contracts.py::COMMON_PROFILE_MATRIX` is authoritative for
the common V1 operation roster, signatures, supported groups, result layouts,
mutation rules, certified backends, and role-specific evidence obligations.
`contracts/coverage.toml` complements it with backend-specific surfaces,
support states, provider routes, diagnostics, and general evidence policy.
Together they describe test obligations; neither is a runtime capability
registry.

Tests associate direct evidence with a manifest cell using:

```python
@pytest.mark.evidence_for(
    "scoped.block.reduce", backend="numba_mlir", evidence="runtime"
)
def test_block_reduce_runtime():
    ...
```

The association is many-to-many: several logical tests may support one cell,
and one test may carry several markers when it directly proves several cells.
A required cell is satisfied when at least one logical claim comes from its
declared collection lane and every selected parameter case for that logical
test passes setup, call, and teardown. Skips, xfails, failures, and non-strict
unexpected passes do not earn evidence. A selected case that never reports an
outcome, including after interruption or worker loss, also prevents the logical
test from earning evidence. Collection-only runs validate the inventory but do
not earn evidence.

Collection fails for:

- an unknown scenario/backend or evidence name;
- required evidence that has no declared collection lane or claim;
- a required claim outside its declared collection lane.

The completed test run also fails when a selected required cell was declared
at collection but no supporting logical test earned it. `--forbid-cuda-coop-skips`
remains the stricter lane policy: it rejects any skip, including tests that do
not carry evidence.

Required rows declare the collection lanes permitted to supply each evidence
cell:

```toml
collection_paths_by_evidence = { api = ["tests/contracts/public_api/test_reduce.py"], runtime = ["tests/backends/numba_mlir/runtime/test_block_reduce.py"] }
```

Collecting a declared file or directory enforces only the cells assigned there,
so a contracts lane does not depend on runtime collection and vice versa. An
exact `file.py::test_name` selection remains a focused developer action and
audits that test's own claims without activating missing sibling claims. Serial
and xdist runs use the same selected-case inventory. If collection skips the
module before the selected item exists, its file lane is enforced as a
conservative fallback because no marker inventory is available. The legacy
`collection_paths` list remains a fallback for existing required rows, but new
required rows should assign lanes per evidence. A capability may override
individual evidence lanes from its surface-level map; unmentioned evidence
keeps the surface lane.

The manifest retains `migration` mode for qualified surfaces that still use
broad source globs. Their missing marker cells are reported as migration debt
rather than silently treated as complete. Every common-V1 operation has moved
to direct `evidence_for` ownership with exact per-evidence collection lanes.
Adding another common operation therefore requires backend-neutral API and
semantics evidence plus CUTLASS and Numba-CUDA-MLIR lowering, positive
compilation, runtime, and final-link evidence from the start.

Common Reduce evidence uses `max` so it is distinct from the Sum alias. The
block algorithm selector and physical-warp valid-count selector both require
`broadcast=False`; every member participates, but only rank zero consumes the
root-owned scalar. Runtime evidence compares common and qualified calls with an
independent oracle and proves that inputs remain unchanged.

Common Histogram evidence covers only a complete block and a fixed-size
`ThreadData` samples payload. It must prove that the input is unchanged; `bins`
and `bins_per_thread` are positive static values; capacity satisfies
`bins <= group_size * bins_per_thread`; member rank `r` owns striped bin
`r + i * group_size`; excess result slots are zero; and every input sample
satisfies the precondition `0 <= sample < bins`.

Common keys-only Merge Sort evidence covers complete blocks and physical warps,
with multi-item integral `ThreadData`, duplicate keys, ascending and descending
orders, full and partial tiles, paired `valid_items`/`oob_default` controls, and
input preservation. A block must contain a power-of-two number of threads. For
a partial tile, `oob_default` must sort after every valid key: greater than
every valid key for ascending order and less than every valid key for
descending order. Runtime evidence asserts those preconditions and compares the
common and qualified results with an independent sort oracle; final-link
evidence proves that their shared public-CUB BlockMergeSort and WarpMergeSort
provider wrappers disappear.

Common keys-only Radix Sort evidence covers complete physical blocks and full
multi-item `ThreadData` tiles. It exercises `int32`, `uint32`, `int64`, and
`uint64` keys, duplicate and signed/unsigned high-bit values, both directions,
the default full-width range, a nonzero `begin_bit` with `end_bit=None`, an
explicit subrange, and explicit `TempStorage`. Runtime evidence compares common
and qualified results with an independent CUB bit-order oracle and proves input
preservation. Final-link evidence proves that the shared public-CUB
`BlockRadixSort` provider wrappers disappear.

Each role has a dedicated collection owner so host, cluster, and
architecture-specific selections do not activate evidence they intentionally
exclude. A compatible compiler artifact remains a prerequisite for each
required backend lane; absence of that artifact is a lane failure, not passing
evidence.

The support states have precise meanings:

- `executable`: a supported route requiring API, semantics, lowering, compile,
  runtime, and final-link evidence;
- `signature_only`: a parity signature with no executable lowering;
- `placeholder`: a reserved public name with an exact not-implemented error;
- `native_equivalent`: the DSL's native operation owns the semantic role;
- `blocked`: intended support is stopped by a named dependency or provider gap;
- `unsupported`: an intentionally unavailable operation with an exact error;
- `not_applicable`: the value model deliberately has no such concept.

Parity compares semantics only where value models overlap. Numba-CUDA-MLIR and
CUTLASS use scoped and/or group-first values. The qualified CUTLASS and Numba
scoped value models are retained only as compatibility coverage; their public
interfaces are group-first.

## Markers and Cadence

Directory selection owns functional scope. Markers express backend,
resources, and cadence:

- `backend_core`, `backend_numba_mlir`, `backend_cutlass`;
- `contract`, `unit`, `compile`, `runtime`, `gpu`, and `link`;
- `large`, `stress`, and `qualification`;
- `requires_sm100` and `requires_sm120`.

The harness applies backend and layer markers from the directory path, so moved
legacy tests do not need decorator boilerplate. Explicit marks add finer
resource or scenario detail.

Markers are registered and strict. Unknown markers and non-strict xfails fail
the run. An unqualified `pytest tests` collection ignores optional backend and
provider subtrees, keeping the minimal contract gate independent of locally
installed DSL stacks. Select a backend/provider directory directly, or pass
`--require-cuda-coop-backend`, to opt in. Qualification, large, link-sensitive,
and GPU tests run serially unless the runner explicitly assigns independent
GPUs. Host and compile tests may use a bounded worker count; do not use an
unbounded `-n auto`.

From `python/cuda_coop`:

```bash
# Backend-free pull-request gate
python -m pytest tests/contracts

# Show the remaining qualified-surface migration cells
python -m pytest tests/contracts --coop-coverage-report

# One backend's host and compile evidence
python -m pytest tests/backends/numba_mlir/unit tests/backends/numba_mlir/compile \
  -m "not gpu" -n 6

# Compile diagnostics that require launching the compiler through a GPU runtime
python -m pytest tests/backends/numba_mlir/compile -m gpu -n 0 \
  --require-cuda-coop-backend numba_mlir

# Representative GPU execution
python -m pytest tests/backends/numba_mlir/runtime -m gpu -n 0 \
  --require-cuda-coop-backend numba_mlir

# Scheduled stress and qualification audit
python -m pytest tests/providers/qualification tests/backends/numba_mlir/stress \
  -m "qualification or stress" -n 0
```

An optional backend may skip during an exploratory local run. A blocking PR
lane must pass `--require-cuda-coop-backend BACKEND`, preflight its environment,
and pass `--forbid-cuda-coop-skips`. Scheduled full and qualification lanes
retain the same import, distribution, GPU, environment, and non-empty-selection
preflights, but report documented case-level skips instead of rejecting every
skip. Missing declared prerequisites therefore still fail rather than turning a
lane silently green.

### Static typing probe

The contracts lane validates that every public stub declares its intended
common or backend-qualified exports without adding a static-checker dependency.
When Pyright is available, run the optional editor-facing probe from
`python/cuda_coop`:

```bash
PYTHONPATH="$PWD" pyright --pythonversion 3.10 \
  tests/support/fixtures/typing_public_surfaces.py
```

The probe resolves the restricted `cuda.coop` root and the public group-first
CUTLASS and Numba-CUDA-MLIR roots. Separate fixtures validate the private
`_block` and `_warp` compatibility stubs while explicitly allowing
private-name access. Built-wheel tests separately require the root and
backend-local `py.typed` markers plus the retained internal stubs to appear in
wheel metadata.

### CI lanes

From the repository root, the CI runner exposes the same ownership boundaries:

```bash
./ci/test_cuda_coop_python.sh -py-version 3.12 -stage contracts
./ci/test_cuda_coop_python.sh -py-version 3.12 -stage numba-mlir
./ci/test_cuda_coop_python.sh -py-version 3.12 -stage cutlass
```

Pull requests use the full contracts lane. CUTLASS uses an explicit
`cutlass-host` PR lane while its compile/runtime stack is still
qualification-only. Numba-MLIR remains scheduled-only until a clean
published stack exports its required whole-function planner API;
`numba-mlir-host` is available for validating a candidate stack but is
deliberately not a green-by-skip PR promise. The wheel producer first builds
`cccl-headers`, then builds `cuda-coop` and verifies its exact header
dependency against that local artifact. Consumers install the combined local
wheelhouse, reject `cuda-cccl`, and reject imports that resolve into the
source tree. Numba-MLIR installs its dedicated test extras in an isolated
environment. CUTLASS jobs install an exact compiler-artifact lock supplied by
the lane; cuda-coop metadata does not choose a CUTLASS compiler runtime.
The Numba-MLIR preflight checks `WholeFunctionPlanner`, `register_planner`, and
`require_launch_config` through `numba_cuda_mlir.extending`; cuda.coop tests do
not inspect the backend's private planner registry.

The scheduled breadth stage is `numba-mlir-qualification`, which audits the
final-link qualification suites under `tests/providers/qualification`.
Contracts plus host/compile selection use at most six workers. Runtime and
qualification selection is serial (`-n 0`) unless the lane assigns independent
GPUs. Calling the script without `-stage` preserves the explicitly
backend-free contracts gate; backend CI should always select its named lane.

## Determinism and Process State

The autouse `deterministic_rng` fixture seeds and restores legacy
`random`/`numpy.random` global state and supplies an isolated NumPy Generator.
`deterministic_seed`, `python_rng`, and `numpy_rng` are available for explicit
use. The seed is derived from the node ID, can be mixed with
`--coop-random-seed`, and is printed on failure.

Use `scoped_env` for environment updates and `compiler_cache_dir` for isolated
compiler caches. Tests that produce a useful reproducer can write it under
`failure_artifact_dir`; pytest retains the directory and the failure report
prints its exact path. Import optional backends inside fixtures or tests. Never
compile, initialize a backend, inspect a GPU, mutate the environment, or change
`sys.path` at module import time.

CUTLASS backend and provider tests automatically receive one session-scoped
provider cache with `CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT=ltoir`. The
harness restores both environment variables at session end; tests must not
create module-scoped cache directories or mutate these settings themselves.

## Case Selection

Test cheap normalization, enums, boundaries, planning, and diagnostics
exhaustively in contracts. Compile once per distinct lowering/provider route.
Run one canonical case per route plus cases for genuinely different semantic or
ABI branches.

Pull-request matrices should cover dtype, topology, item count, algorithm,
partial-tile, and custom-type classes with curated pairwise cases. A
parametrization producing more than 32 pull-request cases fails collection
unless it is marked `stress`, `large`, or `qualification`. Temporary legacy
exceptions must be bounded and expiring `matrix_waiver` records in
`coverage.toml`; they may not silently grow. Move broad Cartesian products, randomized
differential runs, architecture grids, SASS comparisons, and stress cases to
qualification.

An xfail must be strict, conditional, linked to a tracked issue, and carry an
expiry date. Do not convert arbitrary exceptions or numerical mismatches into
xfails.

Branch coverage is a host-contract signal only. The contracts lane measures
`cuda.coop._core` with branch coverage and currently ratchets from the cleaned
Phase 0 baseline of 87%. Do not extend that percentage to compiler adapters,
GPU kernels, or provider execution; their correctness is owned by lowering,
compile, link, runtime, and qualification evidence.

## Adding or Changing a Primitive

1. Add or update the backend-specific row in `coverage.toml`.
2. Define semantic cases and an oracle under `contracts` or `support`.
3. Add API and exact-diagnostic evidence.
4. Add lowering and compile evidence once per distinct provider route.
5. Add representative runtime and final-link evidence when executable.
6. Add qualification only for a concrete type, topology, performance, or
   regression question.
7. Run the contract audit, targeted backend tests, and pre-commit.
