#!/usr/bin/env bash
# Per-binary histogram sweep with incremental postprocessing: sweep one binary
# (even_single, range_single, even_multi, range_multi), merge its perf data + run
# its hit-rate pass, then regenerate figures -- so partial results land after each
# binary instead of only at the very end.
#
# Usage: run from the worktree root.
#   bash cub/benchmarks/bench/histogram/run_perbinary_sweep.sh <run_tag> [extra sweep args...]
set -uo pipefail

HERE="cub/benchmarks/bench/histogram"
PY=/home/shadeform/.local/viz-venv/bin/python3
RUN_TAG="${1:?usage: run_perbinary_sweep.sh <run_tag> [extra args]}"; shift || true
OUT="autocuda/results/${RUN_TAG}"
BRANCH_BIN="build/autocuda/cub-benchmark/bin"
MAIN_BIN="/home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin"
mkdir -p "$OUT"

# Sweep grid (override by editing here). Bins span the SMEM tier .. 1M incl. the
# medium 32k/128k; 6 element sizes; both sample types; full 14 shapes.
BINS="256 1024 4096 16384 32768 65536 131072 262144 1048576"
ELEMENTS="1048576 16777216 67108864 268435456 1073741824 2000000000"
SAMPLES="I32 F64"
SHAPES="concentrated:1.0 concentrated:0.75 concentrated:0.5 concentrated:0.25 concentrated:0.0 powerlaw:0.75 powerlaw:0.5 powerlaw:0.25 zipf:1.0 hash_synonym stale_resident temporal_phases strided_sweep sawtooth"

# Characterization figures once (input-shape, binary-independent).
echo "=== [$(date +%H:%M:%S)] characterization figures ==="
HIST_SWEEP_OUTDIR="$OUT" "$PY" "$HERE/histogram_input_characterization.py" --outdir "$OUT/input_shape_figs" >/dev/null 2>&1 || true

MERGED="$OUT/algo_sweep_full.json"
# Order: do single-channel even/range first (faster, no 1G/2G skip), then multi.
for B in even range multi_even multi_range; do
  PARTIAL="$OUT/_partial_${B}.json"
  echo "=== [$(date +%H:%M:%S)] PERF SWEEP: $B ==="
  HIST_SWEEP_OUTDIR="$OUT" "$PY" "$HERE/histogram_algo_sweep.py" \
    --branch-bin-dir "$BRANCH_BIN" --main-bin-dir "$MAIN_BIN" \
    --binaries "$B" --samples $SAMPLES --bins $BINS --elements $ELEMENTS --shapes $SHAPES \
    --repeats 3 --min-time 0.02 --timeout 180 --out "$PARTIAL" "$@"
  if [[ ! -s "$PARTIAL" ]]; then echo "!! $B produced no JSON; skipping"; continue; fi

  echo "=== [$(date +%H:%M:%S)] HIT-RATE PASS: $B ==="
  HIST_SWEEP_OUTDIR="$OUT" HIST_BENCH_BINDIR="$BRANCH_BIN" \
    "$PY" "$HERE/histogram_hitrate_sweep.py" "$B" || true

  # Merge this binary's perf data into the cumulative MERGED file.
  "$PY" - "$MERGED" "$PARTIAL" <<'PYMERGE'
import json, os, sys
merged_path, partial_path = sys.argv[1], sys.argv[2]
merged = json.load(open(merged_path)) if os.path.exists(merged_path) else {}
partial = json.load(open(partial_path))
merged.update(partial)  # per-binary keys are disjoint
json.dump(merged, open(merged_path, "w"), indent=1)
print(f"  merged {list(partial)} -> {merged_path} ({len(merged)} binaries)")
PYMERGE

  echo "=== [$(date +%H:%M:%S)] FIGURES (cumulative through $B) ==="
  rm -rf "$OUT/algo_perf_figs"
  HR_ARG=(); [[ -s "$OUT/hitrate_results.json" ]] && HR_ARG=(--hitrate "$OUT/hitrate_results.json")
  "$PY" "$HERE/histogram_algo_perf.py" --results "$MERGED" "${HR_ARG[@]}" --outdir "$OUT/algo_perf_figs" >/dev/null 2>&1
  echo "  partial results ready: $(find "$OUT/algo_perf_figs" -name '*.png' | wc -l) figs covering $(${PY} -c "import json;print(sorted(json.load(open('$MERGED')).keys()))")"
done
echo "=== [$(date +%H:%M:%S)] DONE: $OUT ==="
