// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Simple example of cub::DeviceBatchedTopK::MaxPairs().
//! For each (small) segment, find the K largest float keys and gather their associated values. This example
//! demonstrates the argument annotation framework with *runtime* values that carry a *compile-time* upper bound:
//! the segment size and K are passed as `cuda::args::immediate{value, cuda::args::bounds<1, MAX>()}`.
//!
//! To compile using the command line:
//!   nvcc -arch=sm_XX example_device_batched_topk_pairs.cu -I../.. -lcudart -O3

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

bool g_verbose = false;

// Compile-time upper bounds on the (small) segment size and K. The actual values are provided at runtime but are
// guaranteed not to exceed these bounds, which lets cub::DeviceBatchedTopK specialize for a single block per segment.
using offset_t                         = cuda::std::int64_t;
static constexpr offset_t max_seg_size = 1024;
static constexpr offset_t max_k        = 32;

int main(int argc, char** argv)
{
  offset_t segment_size = 256;
  offset_t k            = 8;
  int num_segments      = 2000;

  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("segment-size", segment_size);
  args.GetCmdLineArgument("k", k);
  args.GetCmdLineArgument("num-segments", num_segments);

  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << "Usage: " << argv[0]
              << " [--segment-size=<size, <=1024>] [--k=<k, <=32>] [--num-segments=<segments>] [--device=<id>] [--v]\n";
    exit(0);
  }
  CubDebugExit(args.DeviceInit());

  segment_size             = std::min(segment_size, max_seg_size);
  k                        = std::min(k, segment_size);
  const offset_t num_items = static_cast<offset_t>(num_segments) * segment_size;

  std::cout << "cub::DeviceBatchedTopK::MaxPairs() finds the " << k << " largest items in each of " << num_segments
            << " segments of " << segment_size << " items\n";

  // Initialize host input keys; values are the global item indices.
  thrust::host_vector<float> h_keys_in(num_items);
  for (offset_t i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys_in[i]);
  }

  thrust::device_vector<float> d_keys_in_buffer(h_keys_in);
  thrust::device_vector<float> d_keys_out_buffer(static_cast<size_t>(num_segments) * k, thrust::no_init);
  thrust::device_vector<offset_t> d_values_out_buffer(static_cast<size_t>(num_segments) * k, thrust::no_init);

  // Per-segment iterators. Input values are the global indices [0, num_items) via a counting iterator.
  auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_in_buffer.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_out_buffer.data())), k);
  auto d_values_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(cuda::make_counting_iterator(offset_t{0})), segment_size);
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_values_out_buffer.data())), k);

  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);
  cudaStream_t stream = nullptr;
  CubDebugExit(cudaStreamCreate(&stream));
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, requirements};

  // Annotate the arguments: runtime segment size and K, each with a compile-time upper bound.
  auto segment_sizes  = cuda::args::immediate{segment_size, cuda::args::bounds<offset_t{1}, max_seg_size>()};
  auto k_arg          = cuda::args::immediate{k, cuda::args::bounds<offset_t{1}, max_k>()};
  auto num_segs_arg = cuda::args::immediate{static_cast<offset_t>(num_segments)};

  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceBatchedTopK::MaxPairs(
    nullptr,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs_arg,
    env));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  CubDebugExit(DeviceBatchedTopK::MaxPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs_arg,
    env));

  // Validate: (1) each output value indexes back to the matching output key, and (2) the selected keys are the K
  // largest of their segment.
  thrust::host_vector<float> h_keys_out(d_keys_out_buffer);
  thrust::host_vector<offset_t> h_values_out(d_values_out_buffer);
  bool ok = true;
  for (offset_t s = 0; s < num_segments && ok; ++s)
  {
    // (1) value/key association
    for (offset_t j = 0; j < k; ++j)
    {
      const offset_t global_idx = h_values_out[s * k + j];
      ok = ok && global_idx >= 0 && global_idx < num_items && h_keys_in[global_idx] == h_keys_out[s * k + j];
    }
    // (2) key correctness against a host reference (top-k largest, order-independent)
    std::vector<float> seg(h_keys_in.begin() + s * segment_size, h_keys_in.begin() + (s + 1) * segment_size);
    std::partial_sort(seg.begin(), seg.begin() + k, seg.end(), std::greater<float>{});
    std::vector<float> res(h_keys_out.begin() + s * k, h_keys_out.begin() + (s + 1) * k);
    std::sort(res.begin(), res.end(), std::greater<float>{});
    ok = ok && std::equal(seg.begin(), seg.begin() + k, res.begin());
  }
  AssertEquals(true, ok);

  CubDebugExit(cudaStreamDestroy(stream));
  std::cout << "\n\n";
  return 0;
}
