// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% CUB_TEST_TOPK_UNSUPPORTED_ARCH arch 0:1

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

// Verifies the strict unsupported-architecture diagnostic in cub::DeviceBatchedTopK (see
// CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/device_batched_topk.cuh). A deterministic (gpu_to_gpu) request
// can only be served by the SM90+ cluster backend; when the translation unit targets an architecture that cannot run
// it, the default (strict) mode must fail to compile rather than defer the failure to a runtime cudaErrorNotSupported.
// Built *without* CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT, so it exercises the strict path that CUB's other top-k
// tests intentionally opt out of.
//
// The two %PARAM% variants compile this identical source against different architecture lists (pinned per-variant in
// CMakeLists.txt, independent of the preset): variant 0 pins a single pre-SM90 arch, variant 1 a mixed pre-/post-SM90
// list. The latter documents the "fail if *any* target architecture is unsupported" contract -- the static_assert
// (guarded by any_target_cc_unsupported, which scans every targeted compute capability) fires because at least one
// target is unsupported, even though the co-pinned SM90+ target could serve the request. CUB_TEST_TOPK_UNSUPPORTED_ARCH
// only spawns the two variants; the architecture list itself is a CMake target property.

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
  // expected-error {{"cannot be served on at least one architecture"}}

  auto env                             = cuda::std::execution::env{requirements};
  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
