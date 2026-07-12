// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

// Verifies the strict unsupported-architecture diagnostic in cub::DeviceBatchedTopK (see
// _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/dispatch/dispatch_batched_topk.cuh). A deterministic
// (gpu_to_gpu) request can only be served by the SM90+ cluster backend; when the translation unit targets an
// architecture that cannot run it (this target is pinned to a pre-SM90 arch via CUDA_ARCHITECTURES in CMakeLists.txt),
// the default (strict) mode must fail to compile rather than defer the failure to a runtime cudaErrorNotSupported.
//
// This is the only batched/segmented top-k test built *without* _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT, so it
// exercises the strict path that CUB's other top-k tests intentionally opt out of.

int main()
{
  namespace ex = cuda::execution;

  int** d_keys_in    = nullptr;
  int** d_keys_out   = nullptr;
  auto segment_sizes = cuda::args::constant<8>{};
  auto k_arg         = cuda::args::constant<3>{};
  auto num_segments  = cuda::args::immediate{cuda::std::int64_t{2}};

  // A deterministic result set routes to the cluster backend, which is unsupported on the pinned pre-SM90 target.
  auto requirements =
    ex::require(ex::determinism::gpu_to_gpu, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error {{"is not supported on at least one architecture"}}

  auto env                  = cuda::std::execution::env{requirements};
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
