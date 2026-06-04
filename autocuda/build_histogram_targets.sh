#!/usr/bin/env bash
set -euo pipefail

cmake_bin="/home/shadeform/.local/cmake-venv/bin/cmake"
if [[ ! -x "${cmake_bin}" ]]; then
  cmake_bin="cmake"
fi

common_cmake_args=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=native
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++
  -DCCCL_ENABLE_CUB=ON
  -DCCCL_ENABLE_THRUST=OFF
  -DCCCL_ENABLE_LIBCUDACXX=OFF
  -DCCCL_ENABLE_CUDAX=OFF
  -DCCCL_ENABLE_C_PARALLEL=OFF
  -DCCCL_ENABLE_C_EXPERIMENTAL_STF=OFF
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
)

if [[ ! -f build/autocuda/cub-benchmark/build.ninja ]]; then
  "${cmake_bin}" -S . -B build/autocuda/cub-benchmark \
    "${common_cmake_args[@]}" \
    -DCCCL_ENABLE_BENCHMARKS=ON \
    -DCCCL_ENABLE_TESTING=OFF \
    -DCCCL_ENABLE_EXAMPLES=OFF \
    -DCUB_ENABLE_EXAMPLES=OFF \
    -DCUB_ENABLE_TESTING=OFF \
    -DCUB_ENABLE_HEADER_TESTING=OFF
fi

"${cmake_bin}" --build build/autocuda/cub-benchmark \
  --target \
    cub.bench.histogram.even.base \
    cub.bench.histogram.range.base \
    cub.bench.histogram.multi.even.base \
    cub.bench.histogram.multi.range.base \
  -j5

if [[ ! -f build/autocuda/cub-test/build.ninja ]]; then
  "${cmake_bin}" -S . -B build/autocuda/cub-test \
    "${common_cmake_args[@]}" \
    -DCCCL_ENABLE_BENCHMARKS=OFF \
    -DCCCL_ENABLE_TESTING=ON \
    -DCCCL_ENABLE_EXAMPLES=OFF \
    -DCUB_ENABLE_EXAMPLES=OFF \
    -DCUB_ENABLE_TESTING=ON \
    -DCUB_ENABLE_HEADER_TESTING=OFF
fi

"${cmake_bin}" --build build/autocuda/cub-test \
  --target \
    cub.test.device.histogram.lid_0 \
    cub.test.device.histogram_api.lid_0 \
    cub.test.device.histogram_env.lid_0 \
    cub.test.device.histogram_thread_local_cache.lid_0 \
  -j5
