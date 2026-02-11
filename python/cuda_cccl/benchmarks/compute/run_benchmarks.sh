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
# Benchmark names follow CUB structure:
#   transform/fill, transform/babelstream, transform/heavy
#   reduce/sum
#   scan/exclusive/sum
#
# Examples:
#   ./run_benchmarks.sh                           # Run all benchmarks
#   ./run_benchmarks.sh -b transform/fill         # Only fill benchmark
#   ./run_benchmarks.sh -b reduce/sum -d 0        # Reduce sum on device 0
#   ./run_benchmarks.sh -b scan/exclusive/sum --py # Only Python

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCCL_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CUB_BENCH_DIR="$CCCL_ROOT/build/cub/bin"

# Default values
DEVICE="0"
BENCHMARK=""
RUN_PY=true
RUN_CPP=true

# Supported benchmarks (Python implementations available)
# Format: "category/name" or "category/subcategory/name"
SUPPORTED_BENCHMARKS=(
    "transform/fill"
    "transform/babelstream"
    "transform/heavy"
    "reduce/sum"
    "reduce/min"
    "scan/exclusive/sum"
    "histogram/even"
    "select/if"
    "select/unique_by_key"
    "radix_sort/keys"
    "radix_sort/pairs"
    "merge_sort/keys"
    "segmented_reduce/sum"
    "partition/three_way"
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
    echo ""
    echo "Supported benchmarks:"
    for bench in "${SUPPORTED_BENCHMARKS[@]}"; do
        echo "  $bench"
    done
    exit 0
}

error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Get C++ binary name from benchmark path
# e.g., "transform/fill" -> "cub.bench.transform.fill.base"
get_cpp_binary() {
    local bench="$1"
    echo "cub.bench.${bench//\//.}.base"
}

# Get Python script path from benchmark path
# e.g., "transform/fill" -> "transform/fill.py"
get_py_script() {
    local bench="$1"
    echo "${bench}.py"
}

# Get results file path (with subdirectory)
# e.g., "transform/fill", "cpp" -> "results/transform/fill_cpp.json"
get_result_path() {
    local bench="$1"
    local suffix="$2"
    local dir=$(dirname "$bench")
    local name=$(basename "$bench")
    echo "$RESULTS_DIR/${dir}/${name}_${suffix}.json"
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
        error_exit "Benchmark '$BENCHMARK' not supported.

Available benchmarks:
$(printf '  %s\n' "${SUPPORTED_BENCHMARKS[@]}")"
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

    CPP_BINARY=$(get_cpp_binary "$bench")
    PY_SCRIPT=$(get_py_script "$bench")
    CPP_RESULT=$(get_result_path "$bench" "cpp")
    PY_RESULT=$(get_result_path "$bench" "py")

    # Ensure results subdirectory exists
    mkdir -p "$(dirname "$CPP_RESULT")"

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
        echo "  Results: $CPP_RESULT"
    fi

    # Run Python benchmark
    if [[ "$RUN_PY" == true ]]; then
        echo "Running Python benchmark: $PY_SCRIPT"

        if [[ ! -f "$PY_SCRIPT" ]]; then
            echo "WARNING: Python script not found: $PY_SCRIPT"
            continue
        fi

        python "$PY_SCRIPT" --json "$PY_RESULT" --devices "$DEVICE"
        echo "  Results: $PY_RESULT"
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
    CPP_RESULT=$(get_result_path "$bench" "cpp")
    PY_RESULT=$(get_result_path "$bench" "py")
    echo "  $bench:"
    [[ -f "$CPP_RESULT" ]] && echo "    C++ results:    $CPP_RESULT"
    [[ -f "$PY_RESULT" ]] && echo "    Python results: $PY_RESULT"
done

echo ""
echo "To compare results, run:"
for bench in "${BENCHMARKS_TO_RUN[@]}"; do
    echo "  python analysis/python_vs_cpp_summary.py -b $bench -d $DEVICE"
done
echo ""
