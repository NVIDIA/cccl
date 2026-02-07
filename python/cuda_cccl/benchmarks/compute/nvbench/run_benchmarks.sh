#!/bin/bash
# Run Python cuda.compute and C++ CUB benchmarks
#
# Prerequisites: C++ benchmarks must be built first via:
#   cd /path/to/cccl && ./ci/build_cub.sh -arch 89
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   -d, --device ID     GPU device ID [default: 0]
#   -b, --benchmark NAME Run specific benchmark only [default: all]
#   --py                Only run Python benchmarks
#   --cpp               Only run C++ benchmarks
#   -h, --help          Show this help message
#
# Examples:
#   ./run_benchmarks.sh                    # Run all benchmarks (C++ and Python)
#   ./run_benchmarks.sh -b fill            # Only fill benchmark
#   ./run_benchmarks.sh -b fill -d 0       # Fill benchmark on device 0
#   ./run_benchmarks.sh -b fill --py       # Only Python
#   ./run_benchmarks.sh -b fill --cpp      # Only C++

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCCL_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CUB_BENCH_DIR="$CCCL_ROOT/build/cub/bin"

# Default values
DEVICE="0"
BENCHMARK=""
RUN_PY=true
RUN_CPP=true

# Supported benchmarks (Python implementations available)
SUPPORTED_BENCHMARKS=(
    "fill"
    "babelstream"
    "heavy"
    # Add more as implemented:
    # "reduce_sum"
    # "scan_exclusive_sum"
    # "histogram_even"
    # "select_if"
    # "radix_sort_keys"
    # "segmented_reduce_sum"
    # "unique_by_key"
)

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

print_section() {
    echo ""
    echo "------------------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------------------"
}

show_help() {
    sed -n '/^# Run Python/,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --py)
            RUN_PY=true
            RUN_CPP=false
            shift
            ;;
        --cpp)
            RUN_PY=false
            RUN_CPP=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Validate benchmark name if specified
if [[ -n "$BENCHMARK" ]]; then
    if [[ ! " ${SUPPORTED_BENCHMARKS[@]} " =~ " ${BENCHMARK} " ]]; then
        error_exit "Benchmark '$BENCHMARK' not supported. Available: ${SUPPORTED_BENCHMARKS[*]}"
    fi
    BENCHMARKS_TO_RUN=("$BENCHMARK")
else
    BENCHMARKS_TO_RUN=("${SUPPORTED_BENCHMARKS[@]}")
fi

# ============================================================================
# Main Script
# ============================================================================

print_banner "CCCL Benchmark Runner"

echo "Configuration:"
echo "  CCCL Root:      $CCCL_ROOT"
echo "  C++ Binaries:   $CUB_BENCH_DIR"
echo "  Results Dir:    $RESULTS_DIR"
echo "  Device:         $DEVICE"
echo "  Benchmarks:     ${BENCHMARKS_TO_RUN[*]}"
echo "  Run C++:        $RUN_CPP"
echo "  Run Python:     $RUN_PY"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check C++ binaries exist
if [[ "$RUN_CPP" == true ]]; then
    if [[ ! -d "$CUB_BENCH_DIR" ]]; then
        error_exit "C++ benchmark directory not found: $CUB_BENCH_DIR

Please build C++ benchmarks first:
  cd $CCCL_ROOT
  ./ci/build_cub.sh -arch <your_gpu_arch>  # e.g., 89 for RTX 4090"
    fi
fi

# ============================================================================
# Run Benchmarks
# ============================================================================

cd "$SCRIPT_DIR"

for bench in "${BENCHMARKS_TO_RUN[@]}"; do
    print_section "Benchmark: $bench"

    # Map benchmark name to C++ binary and Python script
    case "$bench" in
        fill)
            CPP_BINARY="cub.bench.transform.fill.base"
            PY_SCRIPT="bench_fill.py"
            ;;
        babelstream)
            CPP_BINARY="cub.bench.transform.babelstream.base"
            PY_SCRIPT="bench_babelstream.py"
            ;;
        heavy)
            CPP_BINARY="cub.bench.transform.heavy.base"
            PY_SCRIPT="bench_heavy.py"
            ;;
        # Add more mappings as benchmarks are implemented
        *)
            echo "WARNING: Unknown benchmark mapping for '$bench', skipping"
            continue
            ;;
    esac

    CPP_RESULT="$RESULTS_DIR/${bench}_cpp.json"
    PY_RESULT="$RESULTS_DIR/${bench}_py.json"

    # Run C++ benchmark
    if [[ "$RUN_CPP" == true ]]; then
        echo "Running C++ benchmark: $CPP_BINARY"

        CPP_BIN="$CUB_BENCH_DIR/$CPP_BINARY"

        if [[ ! -f "$CPP_BIN" ]]; then
            echo "WARNING: C++ binary not found: $CPP_BIN"
            echo "Available benchmarks:"
            ls "$CUB_BENCH_DIR"/*.base 2>/dev/null | head -10 || echo "  (none found)"
            continue
        fi

        "$CPP_BIN" --json "$CPP_RESULT" --devices "$DEVICE"
        echo "  ✓ C++ results: $CPP_RESULT"
    fi

    # Run Python benchmark
    if [[ "$RUN_PY" == true ]]; then
        echo "Running Python benchmark: $PY_SCRIPT"

        if [[ ! -f "$PY_SCRIPT" ]]; then
            echo "WARNING: Python script not found: $PY_SCRIPT"
            continue
        fi

        python "$PY_SCRIPT" --json "$PY_RESULT" --devices "$DEVICE"
        echo "  ✓ Python results: $PY_RESULT"
    fi

    echo ""
done

# ============================================================================
# Summary
# ============================================================================

print_banner "Summary"

echo "Results directory: $RESULTS_DIR"
echo ""
echo "Generated files:"
for bench in "${BENCHMARKS_TO_RUN[@]}"; do
    echo "  $bench:"
    [[ -f "$RESULTS_DIR/${bench}_cpp.json" ]] && echo "    ✓ C++ results:    ${bench}_cpp.json"
    [[ -f "$RESULTS_DIR/${bench}_py.json" ]] && echo "    ✓ Python results: ${bench}_py.json"
done

echo ""
echo "To compare results, run:"
echo "  python analysis/python_vs_cpp_summary.py -b <benchmark> -d $DEVICE"
echo ""
