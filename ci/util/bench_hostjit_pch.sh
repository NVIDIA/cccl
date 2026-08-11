#!/usr/bin/env bash
# Compare HostJIT build times with and without precompiled headers.
#
# Header parsing dominates a cccl.c.parallel.v2 build, and one cached PCH pair
# serves every algorithm. This script measures what that is worth, across the
# cases where the answer differs:
#
#   cold-nopch   empty cache, PCH off            -- the baseline
#   cold-pch     empty cache, PCH on             -- first build pays generation
#   warm-pch     populated cache, PCH on         -- the steady state
#   repeat-*     a second build in the SAME process, PCH on vs off
#
# Every build is timed in a fresh process, so "cold" and "warm" mean what they
# say: a warm measurement reuses only the on-disk cache, never in-process state.
# The repeat-* cases are the exception by design -- they isolate the per-build
# win from the per-process one.
#
# Self-contained: compiles its own harness against an existing v2 build tree,
# runs the matrix, writes CSV, and plots. Nothing is installed or modified
# outside the output directory (and the scratch PCH cache it manages).
#
# Usage:
#   ci/util/bench_hostjit_pch.sh [options]
#
#   -b, --build-dir DIR   v2 CMake build tree (default: autodetected)
#   -o, --output DIR      results directory (default: ./pch-bench-results)
#   -a, --algos LIST      comma-separated: reduce,scan,merge_sort,transform
#                         (default: all four)
#   -n, --repeats N       timed repetitions per case (default: 3)
#       --cc MAJOR.MINOR  target compute capability (default: from nvidia-smi,
#                         else 8.9 -- no GPU is required to compile)
#       --keep-harness    do not delete the compiled harness on exit
#   -h, --help            this message

set -euo pipefail

ALGOS="reduce,scan,merge_sort,transform"
REPEATS=3
BUILD_DIR=""
OUT_DIR="$(pwd)/pch-bench-results"
CC_OVERRIDE=""
KEEP_HARNESS=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-dir)   BUILD_DIR="$2"; shift 2 ;;
    -o|--output)      OUT_DIR="$2"; shift 2 ;;
    -a|--algos)       ALGOS="$2"; shift 2 ;;
    -n|--repeats)     REPEATS="$2"; shift 2 ;;
    --cc)             CC_OVERRIDE="$2"; shift 2 ;;
    --keep-harness)   KEEP_HARNESS=1; shift ;;
    -h|--help)        sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                die "unknown option: $1 (try --help)" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------- build tree

if [[ -z "$BUILD_DIR" ]]; then
  # Prefer the most recently built tree, so this follows whatever the user has
  # actually been working in rather than guessing a CTK version.
  BUILD_DIR="$(find "$REPO_ROOT/build" -maxdepth 2 -type d -name 'cccl-c-parallel-v2' \
                 -exec test -e '{}/lib/libcccl.c.parallel.v2.so' \; \
                 -exec ls -dt '{}' + 2>/dev/null | head -1 || true)"
fi
[[ -n "$BUILD_DIR" ]] || die "no v2 build tree found; build one first:
    CCCL_BUILD_INFIX=<infix> cmake --build --preset cccl-c-parallel-v2 -j4 --target cccl.c.parallel.v2
  or pass --build-dir explicitly."

LIB="$BUILD_DIR/lib/libcccl.c.parallel.v2.so"
[[ -f "$LIB" ]] || die "library not found: $LIB"

# The harness loads this library by path at every invocation. If a build
# relinks it mid-run, cases silently start failing and the ones that already
# ran were measured against different code — so remember what we started with
# and check at the end rather than reporting a quietly mixed dataset.
LIB_STAMP_BEFORE="$(stat -c %Y "$LIB" 2>/dev/null || echo unknown)"

info "build tree: $BUILD_DIR"

# ------------------------------------------------------------------- target

if [[ -n "$CC_OVERRIDE" ]]; then
  CC_MAJOR="${CC_OVERRIDE%%.*}"
  CC_MINOR="${CC_OVERRIDE##*.}"
else
  cc_raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ -n "$cc_raw" && "$cc_raw" == *.* ]]; then
    CC_MAJOR="${cc_raw%%.*}"; CC_MINOR="${cc_raw##*.}"
  else
    # Compiling needs no device -- the arch is just a compiler flag.
    CC_MAJOR=8; CC_MINOR=9
  fi
fi
info "target: sm_${CC_MAJOR}${CC_MINOR}"

mkdir -p "$OUT_DIR"
HARNESS="$OUT_DIR/pch_bench_harness"
CACHE_DIR="$OUT_DIR/.pch-cache"
CSV="$OUT_DIR/timings.csv"

cleanup() {
  # HARNESS_SRC is assigned further down, so it may be unset if we exit early.
  [[ "$KEEP_HARNESS" -eq 1 ]] || rm -f "$HARNESS" "${HARNESS_SRC:-}"
  rm -rf "$CACHE_DIR"
}
trap cleanup EXIT

# ------------------------------------------------------------------ harness

HARNESS_SRC="$OUT_DIR/pch_bench_harness.cpp"
cat > "$HARNESS_SRC" <<'CPPEOF'
// Times cccl.c.parallel.v2 build calls. One build per process invocation by
// default, so the caller controls exactly what is cached between measurements.
//
// The operator is supplied as C++ source (CCCL_OP_CPP_SOURCE) so the harness
// needs no NVRTC/LTO-IR toolchain of its own -- hostjit's Clang compiles it,
// which is the path a real cuda.compute build takes anyway.
#define CCCL_C_EXPERIMENTAL

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cccl/c/merge_sort.h>
#include <cccl/c/reduce.h>
#include <cccl/c/scan.h>
#include <cccl/c/transform.h>

namespace
{
cccl_type_info int32_type()
{
  return cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
}

// A pointer "iterator" needs no real allocation here: the build phase only
// inspects types and sizes, and this harness never launches the kernel.
cccl_iterator_t int32_pointer()
{
  cccl_iterator_t it{};
  it.size       = sizeof(int);
  it.alignment  = alignof(int);
  it.type       = CCCL_POINTER;
  it.value_type = int32_type();
  it.state      = nullptr;
  return it;
}

// CCCL_OP_CPP_SOURCE operators take type-erased pointers: (in..., out).
const char* kSumOp = R"(
extern "C" __device__ void op(void* a, void* b, void* out) {
  *(int*) out = *(int*) a + *(int*) b;
})";

const char* kLessOp = R"(
extern "C" __device__ void op(void* lhs, void* rhs, void* result) {
  *(bool*) result = *(int*) lhs < *(int*) rhs;
})";

const char* kDoubleOp = R"(
extern "C" __device__ void op(void* in, void* out) {
  *(int*) out = *(int*) in * 2;
})";

cccl_op_t cpp_op(const char* name, const char* code)
{
  cccl_op_t op{};
  op.type      = CCCL_STATELESS;
  op.name      = name;
  op.code      = code;
  op.code_size = std::strlen(code);
  op.code_type = CCCL_OP_CPP_SOURCE;
  op.size      = 1;
  op.alignment = 1;
  return op;
}

int build_once(const std::string& algo, int cc_major, int cc_minor, cccl_build_config* cfg)
{
  auto in  = int32_pointer();
  auto out = int32_pointer();

  if (algo == "reduce")
  {
    int init_value = 0;
    cccl_value_t init{int32_type(), &init_value};
    auto op = cpp_op("op", kSumOp);
    cccl_device_reduce_build_result_t build{};
    CUresult r = cccl_device_reduce_build_ex(
      &build, in, out, op, init, CCCL_RUN_TO_RUN,
      cc_major, cc_minor, nullptr, nullptr, nullptr, nullptr, cfg);
    return r == CUDA_SUCCESS ? 0 : 1;
  }
  if (algo == "scan")
  {
    auto op = cpp_op("op", kSumOp);
    cccl_device_scan_build_result_t build{};
    CUresult r = cccl_device_scan_build_ex(
      &build, in, out, op, int32_type(), /*force_inclusive=*/false, CCCL_VALUE_INIT,
      cc_major, cc_minor, nullptr, nullptr, nullptr, nullptr, cfg);
    return r == CUDA_SUCCESS ? 0 : 1;
  }
  if (algo == "merge_sort")
  {
    auto op       = cpp_op("op", kLessOp);
    auto in_items = int32_pointer();
    auto out_items = int32_pointer();
    cccl_device_merge_sort_build_result_t build{};
    CUresult r = cccl_device_merge_sort_build_ex(
      &build, in, in_items, out, out_items, op,
      cc_major, cc_minor, nullptr, nullptr, nullptr, nullptr, cfg);
    return r == CUDA_SUCCESS ? 0 : 1;
  }
  if (algo == "transform")
  {
    auto op = cpp_op("op", kDoubleOp);
    cccl_device_transform_build_result_t build{};
    CUresult r = cccl_device_unary_transform_build_ex(
      &build, in, out, op,
      cc_major, cc_minor, nullptr, nullptr, nullptr, nullptr, cfg);
    return r == CUDA_SUCCESS ? 0 : 1;
  }
  std::fprintf(stderr, "unknown algo: %s\n", algo.c_str());
  return 2;
}
} // namespace

int main(int argc, char** argv)
{
  if (argc < 5)
  {
    std::fprintf(stderr, "usage: %s <algo> <0|1 pch> <cc_major> <cc_minor> [n_builds]\n", argv[0]);
    return 2;
  }
  const std::string algo = argv[1];
  const int use_pch      = std::atoi(argv[2]);
  const int cc_major     = std::atoi(argv[3]);
  const int cc_minor     = std::atoi(argv[4]);
  const int n_builds     = argc > 5 ? std::atoi(argv[5]) : 1;

  cccl_build_config cfg{};
  cfg.enable_pch = use_pch;

  const auto t0 = std::chrono::steady_clock::now();

  int rc = 0;
  for (int i = 0; i < n_builds && rc == 0; ++i)
  {
    rc = build_once(algo, cc_major, cc_minor, &cfg);
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Timing goes to stdout, everything else to stderr, so the caller can just
  // read one number.
  std::printf("%.1f\n", ms);
  return rc;
}
CPPEOF

info "compiling harness"
CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 -o "$HARNESS" "$HARNESS_SRC" \
  -I"$REPO_ROOT/c/parallel.v2/include" \
  -I"${CUDA_HOME:-/usr/local/cuda}/include" \
  -L"$BUILD_DIR/lib" -lcccl.c.parallel.v2 \
  -Wl,-rpath,"$BUILD_DIR/lib" \
  || die "harness failed to compile"

# --------------------------------------------------------------- run matrix

run_case() {
  # run_case <label> <algo> <pch 0|1> <cold|warm> [n_builds]
  local label="$1" algo="$2" pch="$3" cache="$4" n="${5:-1}"

  if [[ "$cache" == "cold" ]]; then
    rm -rf "$CACHE_DIR"
  fi
  mkdir -p "$CACHE_DIR"

  # Keep the harness's stderr: a bare FAILED with no diagnostics is unusable.
  local ms err_log="$OUT_DIR/${algo}-${label}.stderr"
  if ! ms="$(CCCL_PCH_CACHE_DIR="$CACHE_DIR" "$HARNESS" "$algo" "$pch" "$CC_MAJOR" "$CC_MINOR" "$n" 2>"$err_log")"; then
    printf '    %-14s %-12s FAILED (stderr: %s)\n' "$algo" "$label" "$err_log" >&2
    FAILURES=$((FAILURES + 1))
    return 1
  fi
  rm -f "$err_log"
  printf '    %-14s %-12s %8.1f ms\n' "$algo" "$label" "$ms" >&2
  echo "$algo,$label,$ms" >> "$CSV"
}

echo "algo,case,ms" > "$CSV"
FAILURES=0

IFS=',' read -ra ALGO_LIST <<< "$ALGOS"

for algo in "${ALGO_LIST[@]}"; do
  info "measuring $algo"
  for ((r = 0; r < REPEATS; ++r)); do
    # Cold, PCH off: the baseline every other number is compared against.
    run_case "cold-nopch" "$algo" 0 cold || true

    # Cold, PCH on: pays generation, so this is the worst case for PCH and the
    # one that decides whether warm-up at import is worth doing.
    run_case "cold-pch" "$algo" 1 cold || true

    # Warm, PCH on: cache already populated by the cold-pch run above, fresh
    # process. This is the steady state a user actually sees.
    run_case "warm-pch" "$algo" 1 warm || true
  done

  # Two builds in one process, to separate the per-build win from the
  # per-process one. Both arms run against the populated cache left by the
  # warm-pch case above: the PCH-off arm ignores it, and clearing it between
  # the arms would make the PCH-on arm pay generation and measure the wrong
  # thing entirely.
  info "measuring $algo (2 builds per process)"
  for ((r = 0; r < REPEATS; ++r)); do
    run_case "repeat-nopch" "$algo" 0 warm 2 || true
    run_case "repeat-pch" "$algo" 1 warm 2 || true
  done
done

LIB_STAMP_AFTER="$(stat -c %Y "$LIB" 2>/dev/null || echo unknown)"
if [[ "$LIB_STAMP_BEFORE" != "$LIB_STAMP_AFTER" ]]; then
  printf '\n\033[1;33mwarning:\033[0m %s was rebuilt during this run.\n' "$LIB" >&2
  printf '         Timings below span two different builds and cases may have\n' >&2
  printf '         failed spuriously. Re-run with the build tree left alone.\n\n' >&2
fi

if [[ "$FAILURES" -gt 0 ]]; then
  printf '\n\033[1;33mwarning:\033[0m %d measurement(s) failed; the table below is incomplete.\n\n' "$FAILURES" >&2
fi
info "results: $CSV"

# ------------------------------------------------------------------- report

python3 - "$CSV" "$OUT_DIR" <<'PYEOF'
import csv, statistics, sys
from collections import defaultdict

csv_path, out_dir = sys.argv[1], sys.argv[2]

rows = defaultdict(list)
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows[(r["algo"], r["case"])].append(float(r["ms"]))

if not rows:
    print("no measurements recorded", file=sys.stderr)
    sys.exit(1)

med = {k: statistics.median(v) for k, v in rows.items()}
algos = sorted({a for a, _ in med})
CASES = ["cold-nopch", "cold-pch", "warm-pch", "repeat-nopch", "repeat-pch"]

# ---- text summary (always produced; the plot is a bonus, not the result) ----
lines = []
# Generation cost is not measured directly; it is what a cold PCH-on build pays
# over a warm one.
gens = [med[(a, "cold-pch")] - med[(a, "warm-pch")]
        for a in algos if (a, "cold-pch") in med and (a, "warm-pch") in med]
warm = statistics.median(gens) if gens else None
if warm is not None:
    lines.append(f"PCH generation (one-time, per key): {warm:8.1f} ms  [cold-pch - warm-pch]")
    lines.append("")

head = f"{'algorithm':<14}" + "".join(f"{c:>15}" for c in CASES) + f"{'speedup':>10}"
lines.append(head)
lines.append("-" * len(head))
for a in algos:
    line = f"{a:<14}"
    for c in CASES:
        v = med.get((a, c))
        line += f"{v:>14.1f} " if v is not None else f"{'-':>15}"
    base, warm_pch = med.get((a, "cold-nopch")), med.get((a, "warm-pch"))
    line += f"{base / warm_pch:>9.2f}x" if base and warm_pch else f"{'-':>10}"
    lines.append(line)

lines.append("")
lines.append("cold-nopch    empty cache, PCH off (baseline)")
lines.append("cold-pch      empty cache, PCH on -- includes one-time generation")
lines.append("warm-pch      populated cache, fresh process -- the steady state")
lines.append("repeat-*      two builds in one process")
lines.append("speedup       cold-nopch / warm-pch")

summary = "\n".join(lines)
print(summary)
with open(f"{out_dir}/summary.txt", "w") as f:
    f.write(summary + "\n")

# ------------------------------- plots --------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print(f"\n[matplotlib not installed -- skipping plots; see {out_dir}/summary.txt]")
    sys.exit(0)

# Sequential-ish ramp for the three single-build cases, plus a muted pair for
# the in-process repeats, so the eye groups them without needing the legend.
COLORS = {
    "cold-nopch":   "#B7B7C4",
    "cold-pch":     "#7A7DE8",
    "warm-pch":     "#2A2D8F",
    "repeat-nopch": "#D8C7A8",
    "repeat-pch":   "#A8813F",
}

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.1, 1]}
)

n = len(CASES)
width = 0.8 / n
for i, c in enumerate(CASES):
    xs = [j + (i - (n - 1) / 2) * width for j in range(len(algos))]
    ys = [med.get((a, c), 0.0) for a in algos]
    ax1.bar(xs, ys, width, label=c, color=COLORS[c], edgecolor="white", linewidth=0.6)

ax1.set_xticks(range(len(algos)))
ax1.set_xticklabels(algos)
ax1.set_ylabel("build time (ms)")
ax1.set_title("HostJIT build time by case (median)", loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9, ncol=2)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#E6E6EC", linewidth=0.8)
ax1.set_axisbelow(True)

speedups = [
    med[(a, "cold-nopch")] / med[(a, "warm-pch")]
    for a in algos
    if (a, "cold-nopch") in med and (a, "warm-pch") in med
]
labels = [a for a in algos if (a, "cold-nopch") in med and (a, "warm-pch") in med]
ax2.barh(labels, speedups, color="#2A2D8F", height=0.55)
ax2.axvline(1.0, color="#8A8A99", linewidth=1, linestyle="--")
for i, s in enumerate(speedups):
    ax2.text(s + 0.05, i, f"{s:.2f}x", va="center", fontsize=9, color="#2A2D8F")
ax2.set_xlabel("speedup (cold-nopch / warm-pch)")
ax2.set_title("Steady-state speedup", loc="left", fontsize=12)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="x", color="#E6E6EC", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.set_xlim(0, max(speedups + [1.0]) * 1.25)

if warm is not None:
    fig.text(
        0.5, 0.015,
        f"One-time PCH generation: {warm:.0f} ms, amortized across every algorithm and process.",
        ha="center", fontsize=9, color="#5A5A68",
    )

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{out_dir}/pch_comparison.png", dpi=150)
print(f"\nplot written to {out_dir}/pch_comparison.png")
PYEOF

info "done -- $OUT_DIR"
