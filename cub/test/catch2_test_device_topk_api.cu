// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("DeviceTopK::MinKeys API example for non-deterministic, unsorted results", "[device][device_transform]")
{
  // example-begin topk-min-keys-non-deterministic-unsorted
  const int k = 4;
  auto input  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6};
  auto output = thrust::device_vector<int>(k, thrust::no_init);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  // Prepare CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cuda::stream_ref stream_ref{stream};

  // Create the environment with the stream and requirements
  auto env = cuda::std::execution::env{stream_ref, requirements};

  // Query temporary storage requirements
  size_t temp_storage_bytes{};
  cub::DeviceTopK::MinKeys(nullptr, temp_storage_bytes, input.begin(), output.begin(), input.size(), k, env);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceTopK::MinKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    input.begin(),
    output.begin(),
    input.size(),
    k,
    env);

  // Get the top-k results into sorted order for easy comparison
  thrust::sort(output.begin(), output.end());
  thrust::host_vector<int> expected{-3, 1, 2, 4};
  // example-end topk-min-keys-non-deterministic-unsorted

  REQUIRE(output == expected);
}

C2H_TEST("DeviceTopK::MaxKeys API example for non-deterministic, unsorted results", "[device][device_transform]")
{
  // example-begin topk-max-keys-non-deterministic-unsorted
  const int k = 4;
  auto input  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6};
  auto output = thrust::device_vector<int>(k, thrust::no_init);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  // Prepare CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cuda::stream_ref stream_ref{stream};

  // Create the environment with the stream and requirements
  auto env = cuda::std::execution::env{stream_ref, requirements};

  // Query temporary storage requirements
  size_t temp_storage_bytes{};
  cub::DeviceTopK::MaxKeys(nullptr, temp_storage_bytes, input.begin(), output.begin(), input.size(), k, env);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    input.begin(),
    output.begin(),
    input.size(),
    k,
    env);

  // Get the top-k results into sorted order for easy comparison
  thrust::sort(output.begin(), output.end(), cuda::std::greater{});
  thrust::host_vector<int> expected{8, 7, 6, 5};
  // example-end topk-max-keys-non-deterministic-unsorted

  REQUIRE(output == expected);
}

C2H_TEST("DeviceTopK::MinPairs API example for non-deterministic, unsorted results", "[device][device_transform]")
{
  // example-begin topk-min-pairs-non-deterministic-unsorted
  const int k     = 4;
  auto keys       = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6};
  auto values     = cuda::make_counting_iterator<int>(0);
  auto keys_out   = thrust::device_vector<int>(k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(k, thrust::no_init);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  // Prepare CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cuda::stream_ref stream_ref{stream};

  // Create the environment with the stream and requirements
  auto env = cuda::std::execution::env{stream_ref, requirements};

  // Query temporary storage requirements
  size_t temp_storage_bytes{};
  cub::DeviceTopK::MinPairs(
    nullptr, temp_storage_bytes, keys.begin(), keys_out.begin(), values, values_out.begin(), keys.size(), k, env);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceTopK::MinPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    keys.begin(),
    keys_out.begin(),
    values,
    values_out.begin(),
    keys.size(),
    k,
    env);

  // Get the top-k results into sorted order for easy comparison
  thrust::sort_by_key(keys_out.begin(), keys_out.end(), values_out.begin());
  thrust::host_vector<int> expected_keys{-3, 1, 2, 4};
  thrust::host_vector<int> expected_values{1, 2, 5, 6};
  // example-end topk-min-pairs-non-deterministic-unsorted

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceTopK::MaxPairs API example for non-deterministic, unsorted results", "[device][device_transform]")
{
  // example-begin topk-max-pairs-non-deterministic-unsorted
  const int k     = 4;
  auto keys       = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6};
  auto values     = cuda::make_counting_iterator<int>(0);
  auto keys_out   = thrust::device_vector<int>(k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(k, thrust::no_init);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  // Prepare CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cuda::stream_ref stream_ref{stream};

  // Create the environment with the stream and requirements
  auto env = cuda::std::execution::env{stream_ref, requirements};

  // Query temporary storage requirements
  size_t temp_storage_bytes{};
  cub::DeviceTopK::MaxPairs(
    nullptr, temp_storage_bytes, keys.begin(), keys_out.begin(), values, values_out.begin(), keys.size(), k, env);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceTopK::MaxPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    keys.begin(),
    keys_out.begin(),
    values,
    values_out.begin(),
    keys.size(),
    k,
    env);

  // Get the top-k results into sorted order for easy comparison
  thrust::sort_by_key(keys_out.begin(), keys_out.end(), values_out.begin(), cuda::std::greater<>{});
  thrust::host_vector<int> expected_keys{8, 7, 6, 5};
  thrust::host_vector<int> expected_values{4, 3, 7, 0};
  // example-end topk-max-pairs-non-deterministic-unsorted

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}
