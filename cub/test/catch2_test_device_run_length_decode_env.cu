// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_decode.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRunLengthDecode::Decode, run_length_decode);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRunLengthDecode::DecodeFromOffsets, run_length_decode_from_offsets);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

template <int ThreadsPerBlock>
struct batch_copy_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::BatchedCopyPolicy
  {
    return {
      cub::BatchedCopyAlgorithm::lookback,
      {
        {ThreadsPerBlock, 4, 8, false, 256 * 32, 128, 8 * 1024, {}, {}},
        {256, 32},
      },
    };
  }
};

template <int ThreadsPerBlock>
struct scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ScanPolicy
  {
    return {cub::ScanAlgorithm::lookback,
            {ThreadsPerBlock,
             1,
             cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
             cub::CacheLoadModifier::LOAD_DEFAULT,
             cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
             cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
             cub::detail::default_delay_constructor_policy(true)},
            {}};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

// Decode reads the run lengths in the scan and when copying the runs, but the run values only when copying the runs.
// Tuning the scan to a larger block size than the copy lets both block sizes be recovered from the two inputs.
template <unsigned int ScanBlockSize, unsigned int CopyBlockSize>
struct decode_block_sizes
{
  static_assert(ScanBlockSize > CopyBlockSize);
  static constexpr unsigned int scan = ScanBlockSize;
  static constexpr unsigned int copy = CopyBlockSize;
};

using decode_block_sizes_list = c2h::type_list<decode_block_sizes<128, 64>, decode_block_sizes<256, 128>>;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceRunLengthDecode::Decode works with default environment", "[run_length_decode][device]", CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_lengths = c2h::device_vector<int>{1, 2, 0, 3, 1};
  auto d_out         = c2h::device_vector<int>(7, thrust::no_init);

  REQUIRE(
    cudaSuccess == cub::DeviceRunLengthDecode::Decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 5));

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceRunLengthDecode::DecodeFromOffsets works with default environment",
              "[run_length_decode][device]",
              CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_offsets = c2h::device_vector<int>{0, 1, 3, 3, 6, 7};
  auto d_out         = c2h::device_vector<int>(7, thrust::no_init);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRunLengthDecode::DecodeFromOffsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 5));

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceRunLengthDecode::Decode uses environment", "[run_length_decode][device]", CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_run_lengths = c2h::device_vector<int>{3, 2, 1, 4};
  auto d_out         = c2h::device_vector<int>(10, thrust::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthDecode::Decode(
            nullptr, expected_bytes_allocated, d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 4));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 4, env);

  const c2h::device_vector<int> expected{1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
  REQUIRE(d_out == expected);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets uses environment", "[run_length_decode][device]", CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_run_offsets = c2h::device_vector<int>{0, 3, 5, 6, 10};
  auto d_out         = c2h::device_vector<int>(10, thrust::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthDecode::DecodeFromOffsets(
            nullptr, expected_bytes_allocated, d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 4));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 4, env);

  const c2h::device_vector<int> expected{1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
  REQUIRE(d_out == expected);
}

#if TEST_LAUNCH != 1

CUB_TEST_CASE("DeviceRunLengthDecode::Decode uses custom stream", "[run_length_decode][device]", CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_lengths = c2h::device_vector<int>{1, 2, 0, 3, 1};
  auto d_out         = c2h::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 5, env);

  stream.sync();

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceRunLengthDecode::DecodeFromOffsets uses custom stream", "[run_length_decode][device]", CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_offsets = c2h::device_vector<int>{0, 1, 3, 3, 6, 7};
  auto d_out         = c2h::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 5, env);

  stream.sync();

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST("DeviceRunLengthDecode::Decode can be tuned", "[run_length_decode][device]", CUB_SMALL, decode_block_sizes_list)
{
  using target_block_sizes = c2h::get<0, TestType>;

  c2h::device_vector<unsigned int> d_block_sizes(2, 0);
  const block_size_extracting_constant_iterator d_run_values(42, thrust::raw_pointer_cast(d_block_sizes.data()));
  const block_size_extracting_constant_iterator d_run_lengths(2, thrust::raw_pointer_cast(d_block_sizes.data()) + 1);
  auto d_out = c2h::device_vector<int>(6, thrust::no_init);

  auto env =
    cuda::execution::tune(scan_tuning<target_block_sizes::scan>{}, batch_copy_tuning<target_block_sizes::copy>{});

  run_length_decode(d_run_values, d_run_lengths, d_out.begin(), 3, env);

  const c2h::device_vector<int> expected(6, 42);
  REQUIRE(d_out == expected);
  REQUIRE(d_block_sizes[0] == target_block_sizes::copy);
  REQUIRE(d_block_sizes[1] == target_block_sizes::scan);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets can be tuned", "[run_length_decode][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  c2h::device_vector<unsigned int> d_block_size(1, 0);
  const block_size_extracting_constant_iterator d_run_values(42, thrust::raw_pointer_cast(d_block_size.data()));
  const auto d_run_offsets = c2h::device_vector<int>{0, 2, 4, 6};
  auto d_out               = c2h::device_vector<int>(6, thrust::no_init);

  auto env = cuda::execution::tune(batch_copy_tuning<target_block_size>{});

  run_length_decode_from_offsets(d_run_values, d_run_offsets.begin(), d_out.begin(), 3, env);

  const c2h::device_vector<int> expected(6, 42);
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

// The two-phase overloads take the same environment as the single-phase ones but never allocate, so they do not go
// through the launch wrappers and would run identically in every TEST_LAUNCH variant. Test them with host launch only.
//
// A failed REQUIRE unwinds past EndCapture, and an abandoned capture poisons every later
// default-stream CUDA call in the process, so the destructor closes the capture on unwind.
// (Mirrors the guard in catch2_test_device_for_env.cu.)
struct stream_capture_guard
{
  cudaStream_t stream;
  bool active = true;

  explicit stream_capture_guard(cudaStream_t stream_in)
      : stream(stream_in)
  {
    REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }

  stream_capture_guard(const stream_capture_guard&)            = delete;
  stream_capture_guard& operator=(const stream_capture_guard&) = delete;

  ~stream_capture_guard()
  {
    if (active)
    {
      cudaGraph_t graph{};
      cudaStreamEndCapture(stream, &graph);
      cudaGraphDestroy(graph);
      cudaGetLastError(); // reset the sticky error state left by the aborted capture
    }
  }

  cudaGraph_t finish()
  {
    active = false;
    cudaGraph_t graph{};
    REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
    return graph;
  }
};

// Every call is recorded under graph capture of a stream, so its kernels only run when the graph is replayed. The
// batched copy launches through triple_chevron directly, which the launcher-factory stream check cannot see, so a
// kernel launched on any other stream is detected by its effect instead: it runs during the capture and writes d_out
// before the replay, or, if it only writes temporary storage, its result is erased by clearing the temporary storage
// before the replay.
template <typename CallCubApi, typename T>
void test_two_phase_with_custom_streams(
  CallCubApi call_cub_api, const c2h::device_vector<T>& d_out, c2h::device_vector<cuda::std::uint8_t>& temp_storage)
{
  const c2h::device_vector<T> d_out_before = d_out;
  const cuda::stream stream                = c2h::make_current_device_stream();

  stream_capture_guard capture{stream.get()};
  test_with_custom_streams(call_cub_api, stream);
  const cudaGraph_t graph = capture.finish();

  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_out == d_out_before);
  REQUIRE(cudaSuccess == cudaMemset(thrust::raw_pointer_cast(temp_storage.data()), 0, temp_storage.size()));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream.get()));
  stream.sync();

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
}

CUB_TEST_CASE("DeviceRunLengthDecode::Decode works with user provided memory and environment",
              "[run_length_decode][device]",
              CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_lengths = c2h::device_vector<int>{1, 2, 0, 3, 1};
  auto d_out         = c2h::device_vector<int>(7, -1); // sentinel to detect writes outside of the captured graph

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthDecode::Decode(
            nullptr, expected_bytes, d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 5));
  c2h::device_vector<cuda::std::uint8_t> temp_storage(expected_bytes, thrust::no_init);

  test_two_phase_with_custom_streams(
    [&](const auto& env) {
      size_t temp_storage_bytes = 0;
      REQUIRE(cudaSuccess
              == cub::DeviceRunLengthDecode::Decode(
                nullptr, temp_storage_bytes, d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 5, env));
      REQUIRE(temp_storage_bytes == expected_bytes);
      REQUIRE(
        cudaSuccess
        == cub::DeviceRunLengthDecode::Decode(
          thrust::raw_pointer_cast(temp_storage.data()),
          temp_storage_bytes,
          d_run_values.begin(),
          d_run_lengths.begin(),
          d_out.begin(),
          5,
          env));
    },
    d_out,
    temp_storage);

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceRunLengthDecode::DecodeFromOffsets works with user provided memory and environment",
              "[run_length_decode][device]",
              CUB_SMALL)
{
  auto d_run_values  = c2h::device_vector<int>{0, 2, 9, 5, 8};
  auto d_run_offsets = c2h::device_vector<int>{0, 1, 3, 3, 6, 7};
  auto d_out         = c2h::device_vector<int>(7, -1); // sentinel to detect writes outside of the captured graph

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthDecode::DecodeFromOffsets(
            nullptr, expected_bytes, d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 5));
  c2h::device_vector<cuda::std::uint8_t> temp_storage(expected_bytes, thrust::no_init);

  test_two_phase_with_custom_streams(
    [&](const auto& env) {
      size_t temp_storage_bytes = 0;
      REQUIRE(cudaSuccess
              == cub::DeviceRunLengthDecode::DecodeFromOffsets(
                nullptr, temp_storage_bytes, d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 5, env));
      REQUIRE(temp_storage_bytes == expected_bytes);
      REQUIRE(
        cudaSuccess
        == cub::DeviceRunLengthDecode::DecodeFromOffsets(
          thrust::raw_pointer_cast(temp_storage.data()),
          temp_storage_bytes,
          d_run_values.begin(),
          d_run_offsets.begin(),
          d_out.begin(),
          5,
          env));
    },
    d_out,
    temp_storage);

  const c2h::device_vector<int> expected{0, 2, 2, 5, 5, 5, 8};
  REQUIRE(d_out == expected);
}

#endif // TEST_LAUNCH == 0
