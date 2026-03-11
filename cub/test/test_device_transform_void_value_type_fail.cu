// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Example from Yunsong Wang, where cub::DeviceTransform fails because the host and device compilation path selected
// different tunings, because the value type of the input iterator is void in host code. This test verifies that CUB now
// correctly fails to compile in such a case,

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/functional>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

using size_type       = std::int32_t;
using hash_value_type = std::uint32_t;

enum class rhs_index_type : size_type
{
};

struct probe_key_type
{
  hash_value_type first;
  rhs_index_type second;
};

struct primitive_row_hasher
{
  size_type const* values;

  __host__ __device__ static hash_value_type mix(hash_value_type x)
  {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
  }

  __device__ hash_value_type operator()(size_type row_index) const
  {
    return mix(static_cast<hash_value_type>(values[row_index]) + 0x9e3779b9U);
  }
};

template <typename Hasher>
struct masked_key_fn
{
  __host__ __device__ constexpr masked_key_fn(Hasher const& hasher)
      : _hasher{hasher}
  {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return probe_key_type{_hasher(i), static_cast<rhs_index_type>(i)};
  }

private:
  Hasher _hasher;
};

int main()
{
  thrust::device_vector<size_type> probe{0, 1, 3};
  thrust::device_vector<probe_key_type> actual(probe.size(), thrust::no_init);

  auto const transform_key_fn =
    masked_key_fn<primitive_row_hasher>{primitive_row_hasher{thrust::raw_pointer_cast(probe.data())}};

  auto stream = cudaStream_t{};
  auto status = cudaStreamCreate(&stream);
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "cudaStreamCreate: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }

  auto const input = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}), transform_key_fn);
  status           = cub::DeviceTransform::Transform(
    input, thrust::raw_pointer_cast(actual.data()), static_cast<int>(probe.size()), cuda::std::identity{}, stream);
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "DeviceTransform: %s\n", cudaGetErrorString(status));
    cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }

  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "cudaStreamSynchronize: %s\n", cudaGetErrorString(status));
    cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }

  cudaStreamDestroy(stream);
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
