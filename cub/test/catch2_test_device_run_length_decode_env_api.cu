// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_decode.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("cub::DeviceRunLengthDecode::Decode accepts env with stream", "[run_length_decode][env]", CUB_SMALL)
{
  // example-begin decode-run-lengths-env
  constexpr int num_runs = 5;
  auto run_values        = thrust::device_vector<int>{0, 2, 9, 5, 8};
  auto run_lengths       = thrust::device_vector<int>{1, 2, 0, 3, 1};
  auto decoded           = thrust::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error =
    cub::DeviceRunLengthDecode::Decode(run_values.begin(), run_lengths.begin(), decoded.begin(), num_runs, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::Decode failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  // example-end decode-run-lengths-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(decoded == expected);
}

CUB_TEST("cub::DeviceRunLengthDecode::DecodeFromOffsets accepts env with stream", "[run_length_decode][env]", CUB_SMALL)
{
  // example-begin decode-run-offsets-env
  constexpr int num_runs = 5;
  auto run_values        = thrust::device_vector<int>{0, 2, 9, 5, 8};
  auto run_offsets       = thrust::device_vector<int>{0, 1, 3, 3, 6, 7};
  auto decoded           = thrust::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRunLengthDecode::DecodeFromOffsets(
    run_values.begin(), run_offsets.begin(), decoded.begin(), num_runs, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::DecodeFromOffsets failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  // example-end decode-run-offsets-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(decoded == expected);
}

#if _CCCL_STD_VER >= 2020

// nvcc turns the `.member = value,` C++ syntax into GNU's `member: value,` when clang (14 - 21) is used
_CCCL_DIAG_PUSH
#  if _CCCL_COMPILER(CLANG)
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-designator")
#  endif // _CCCL_COMPILER(CLANG)

// example-begin decode-policy-selectors
struct BatchedCopyPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::BatchedCopyPolicy
  {
    return {
      .algorithm = cub::BatchedCopyAlgorithm::lookback,
      .lookback  = cub::BatchedCopyLookbackPolicy{
        .small_buffer =
          cub::BatchedCopySmallBufferPolicy{
            .threads_per_block     = 128,
            .buffers_per_thread    = 4,
            .bytes_per_thread      = 8,
            .prefer_pow2_bits      = false,
            .block_level_tile_size = 256 * 32,
            .warp_level_threshold  = 128,
            .block_level_threshold = 8 * 1024,
            .buffer_lookback_delay = {},
            .block_lookback_delay  = {}},
        .large_buffer = cub::BatchedCopyLargeBufferPolicy{.threads_per_block = 256, .bytes_per_thread = 32}}};
  }
};

struct ScanPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    return {
      .algorithm = cub::ScanAlgorithm::lookback,
      .lookback =
        cub::ScanLookbackPolicy{
          .threads_per_block = 256,
          .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 15 : 12,
          .load_algorithm    = cub::BLOCK_LOAD_WARP_TRANSPOSE,
          .load_modifier     = cub::LOAD_DEFAULT,
          .store_algorithm   = cub::BLOCK_STORE_WARP_TRANSPOSE,
          .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
          .lookback_delay =
            cub::LookbackDelayPolicy{
              .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 832, .l2_write_latency = 1165}},
      .lookahead = cub::ScanLookaheadPolicy{} // ignored since algorithm is lookback
    };
  }
};
// example-end decode-policy-selectors

_CCCL_DIAG_POP

CUB_TEST("cub::DeviceRunLengthDecode::Decode accepts custom policy selectors", "[run_length_decode][env]", CUB_SMALL)
{
  // example-begin decode-tuning
  constexpr int num_runs = 5;
  auto run_values        = thrust::device_vector<int>{0, 2, 9, 5, 8};
  auto run_lengths       = thrust::device_vector<int>{1, 2, 0, 3, 1};
  auto decoded           = thrust::device_vector<int>(7, thrust::no_init);

  // The batched copy policy selector is applied to copying the runs, the scan policy selector to the prefix sum over
  // the run lengths
  const auto error = cub::DeviceRunLengthDecode::Decode(
    run_values.begin(),
    run_lengths.begin(),
    decoded.begin(),
    num_runs,
    cuda::execution::tune(BatchedCopyPolicySelector{}, ScanPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::Decode failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  // example-end decode-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(decoded == expected);
}

#endif // _CCCL_STD_VER >= 2020
