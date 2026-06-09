// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Simple example of cub::DeviceBatchedTopK::MaxKeys().
//! For each (small) segment, find the K largest float keys. This example also demonstrates the argument annotation
//! framework: the (small) segment size and K are passed as compile-time `cuda::args::constant<>` values, while
//! the number of segments is a runtime `cuda::args::immediate`.
//!
//! To compile using the command line:
//!   nvcc -arch=sm_XX example_device_batched_topk_keys.cu -I../.. -lcudart -O3

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_batched_topk.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/stream>

#include <algorithm>
#include <iostream>
#include <vector>

#include "../../test/test_util.h"

using namespace cub;

// Whether to display input/output to console
bool g_verbose = false;

// Compile-time, statically-known (small) segment size and K. Passing them as compile-time constants lets
// cub::DeviceBatchedTopK specialize the kernel for a single thread block per segment.
static constexpr int segment_size = 256;
static constexpr int k            = 8;

int main(int argc, char** argv)
{
  int num_segments = 4000; // runtime number of segments

  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("num-segments", num_segments);

  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << "Usage: " << argv[0] << " [--num-segments=<segments>] [--device=<device-id>] [--v]\n";
    exit(0);
  }

  CubDebugExit(args.DeviceInit());

  const int num_items = num_segments * segment_size;
  std::cout << "cub::DeviceBatchedTopK::MaxKeys() finds the " << k << " largest items in each of " << num_segments
            << " segments of " << segment_size << " items (" << sizeof(float) << "-byte keys)\n";

  // Initialize host input
  thrust::host_vector<float> h_keys_in(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys_in[i]);
  }

  // Compute the reference solution: largest K of each segment, sorted descending
  std::vector<float> reference(static_cast<size_t>(num_segments) * k);
  for (int s = 0; s < num_segments; ++s)
  {
    std::vector<float> seg(h_keys_in.begin() + s * segment_size, h_keys_in.begin() + (s + 1) * segment_size);
    std::partial_sort(seg.begin(), seg.begin() + k, seg.end(), std::greater<float>{});
    std::copy(seg.begin(), seg.begin() + k, reference.begin() + static_cast<size_t>(s) * k);
  }

  // Allocate device arrays
  thrust::device_vector<float> d_keys_in_buffer(h_keys_in);
  thrust::device_vector<float> d_keys_out_buffer(static_cast<size_t>(num_segments) * k, thrust::no_init);

  // Build per-segment iterators: d_keys_in[s] points to the start of segment s.
  auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_in_buffer.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_out_buffer.data())), k);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);
  cudaStream_t stream = nullptr;
  CubDebugExit(cudaStreamCreate(&stream));
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, requirements};

  // Annotate the arguments: segment size and K are compile-time constants; the number of segments and the upper bound
  // on the total number of items are runtime values.
  auto segment_sizes  = cuda::args::constant<segment_size>{};
  auto k_arg          = cuda::args::constant<k>{};
  auto num_segs_arg   = cuda::args::immediate{static_cast<cuda::std::int64_t>(num_segments)};
  auto total_num_args = cuda::args::immediate{static_cast<cuda::std::int64_t>(num_items)};

  // Query temporary storage requirements
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs_arg, total_num_args, env));

  // Allocate temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run the segmented top-k algorithm
  CubDebugExit(DeviceBatchedTopK::MaxKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs_arg, total_num_args, env));

  // Check for correctness: the per-segment output is unordered, so sort each output segment descending before
  // comparing against the (descending) reference.
  thrust::host_vector<float> h_keys_out(d_keys_out_buffer);
  int compare = 0;
  for (int s = 0; s < num_segments && compare == 0; ++s)
  {
    std::sort(h_keys_out.begin() + s * k, h_keys_out.begin() + (s + 1) * k, std::greater<float>{});
    compare = CompareResults(
      reference.data() + static_cast<size_t>(s) * k,
      thrust::raw_pointer_cast(h_keys_out.data()) + static_cast<size_t>(s) * k,
      k,
      g_verbose);
  }
  AssertEquals(0, compare);

  CubDebugExit(cudaStreamDestroy(stream));
  std::cout << "\n\n";
  return 0;
}
