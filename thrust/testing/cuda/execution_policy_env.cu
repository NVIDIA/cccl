// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/stream>

#include <unittest/unittest.h>

struct counting_resource
{
  cudaStream_t expected_stream   = nullptr;
  std::size_t* bytes_allocated   = nullptr;
  std::size_t* bytes_deallocated = nullptr;

  void* allocate_sync(std::size_t, std::size_t)
  {
    FAIL("Thrust shouldn't use synchronous allocation");
    return nullptr;
  }

  void deallocate_sync(void*, std::size_t, std::size_t) {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t)
  {
    REQUIRE(stream.get() == expected_stream);
    *bytes_allocated += bytes;
    void* ptr = nullptr;
    REQUIRE(cudaSuccess == cudaMallocAsync(&ptr, bytes, stream.get()));
    return ptr;
  }

  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t)
  {
    REQUIRE(stream.get() == expected_stream);
    *bytes_deallocated += bytes;
    REQUIRE(cudaSuccess == cudaFreeAsync(ptr, stream.get()));
  }

  bool operator==(const counting_resource& other) const
  {
    return expected_stream == other.expected_stream;
  }
  bool operator!=(const counting_resource& other) const
  {
    return !(*this == other);
  }
};
static_assert(cuda::mr::resource<counting_resource>);

using env_t = cuda::std::execution::env<cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>,
                                        cuda::std::execution::prop<cuda::mr::get_memory_resource_t, counting_resource>>;

template <typename ExecutionPolicy>
void test_with_stream_and_resource(ExecutionPolicy policy)
{
  cudaStream_t s;
  REQUIRE(cudaSuccess == cudaStreamCreate(&s));

  std::size_t bytes_allocated   = 0;
  std::size_t bytes_deallocated = 0;
  counting_resource mr{s, &bytes_allocated, &bytes_deallocated};

  const auto env = env_t{{cuda::get_stream, cuda::stream_ref{s}}, {cuda::mr::get_memory_resource, mr}};

  thrust::device_vector<int> v{3, -2, 1};
  const auto env_policy = policy.with(env);
  REQUIRE(get_stream(env_policy) == s);

  thrust::sort(env_policy, v.begin(), v.end());
  REQUIRE(thrust::reduce(env_policy, v.begin(), v.end()) == 2);
  REQUIRE(cudaSuccess == cudaStreamSynchronize(s));

  REQUIRE(v == thrust::device_vector<int>{-2, 1, 3});
  REQUIRE(bytes_allocated > 0);
  REQUIRE(bytes_allocated == bytes_deallocated);

  REQUIRE(cudaSuccess == cudaStreamDestroy(s));
}

template <typename ExecutionPolicy>
void test_with_stream_only(ExecutionPolicy policy)
{
  cudaStream_t s;
  REQUIRE(cudaSuccess == cudaStreamCreate(&s));

  thrust::device_vector<int> v{3, -2, 1};
  const auto env_policy = policy.with(cuda::std::execution::prop{cuda::get_stream, cuda::stream_ref{s}});
  REQUIRE(get_stream(env_policy) == s);
  REQUIRE(thrust::reduce(env_policy, v.begin(), v.end()) == 2);

  REQUIRE(cudaSuccess == cudaStreamDestroy(s));
}

TEST_CASE("TestParWithEnv", "[execution_policy_env]")
{
  test_with_stream_and_resource(thrust::cuda::par);
  test_with_stream_only(thrust::cuda::par);
}

TEST_CASE("TestParNoSyncWithEnv", "[execution_policy_env]")
{
  test_with_stream_and_resource(thrust::cuda::par_nosync);
  test_with_stream_only(thrust::cuda::par_nosync);
}

TEST_CASE("TestParNoSyncWithEnvSkipsOptionalSync", "[execution_policy_env]")
{
  cudaStream_t s;
  REQUIRE(cudaSuccess == cudaStreamCreate(&s));

  std::size_t bytes_allocated   = 0;
  std::size_t bytes_deallocated = 0;
  counting_resource mr{s, &bytes_allocated, &bytes_deallocated};
  const auto env = env_t{{cuda::get_stream, cuda::stream_ref{s}}, {cuda::mr::get_memory_resource, mr}};

  auto par_policy        = thrust::cuda::par.with(env);
  auto par_nosync_policy = thrust::cuda::par_nosync.with(env);
  REQUIRE(thrust::cuda_cub::must_perform_optional_synchronization(par_policy));
  REQUIRE_FALSE(thrust::cuda_cub::must_perform_optional_synchronization(par_nosync_policy));

  REQUIRE(cudaSuccess == cudaStreamDestroy(s));
}
