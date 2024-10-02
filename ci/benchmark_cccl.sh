#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

SOURCE_DIR="$(pwd)/.."
BUILD_NAME="Benchmarks"
PRESET="benchmark"
CMAKE_OPTIONS=""

run_command "⚙️  Installing pip dependencies" pip3 install fpzip pandas scipy

configure_and_build_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"

pushd "${BUILD_DIR}/${PRESET}"

# Need to figure out if this works on nvks
ninja nvbench-ctl || echo "Failed to build nvbench-ctl."
sudo bin/nvbench-ctl --lgc base || echo "Failed to lock gpu clocks to base rate."

# For some reason this has to be set on the GHA runners, not sure why. The script directory
# should be automatically added to sys.path, but isn't when run in CI.
export PYTHONPATH="${SOURCE_DIR}/benchmarks/scripts"
bench_rc=0 # Delay exit-on-error to reset clocks, etc
run_command "⏱️  Running CCCL Benchmarks" ${SOURCE_DIR}/benchmarks/scripts/run.py -P0 || bench_rc=$?

# Attempting to change clocks back to default
sudo bin/nvbench-ctl --lgc reset || echo "Failed to reset GPU clocks."

ls -lh cccl_meta_bench.* || :

echo "::group::cccl_meta_bench.log"
cat cccl_meta_bench.log || :
echo "::endgroup::"

echo "::group::cccl_meta_bench.log Compression"
grep "Compressing" cccl_meta_bench.log || :
echo "::endgroup::"

popd

print_time_summary

if [[ $bench_rc -ne 0 ]]; then
    echo "Failed to run benchmarks."
    exit $bench_rc
fi
