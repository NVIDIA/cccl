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

// example-begin topk-custom-type
struct custom_t
{
  float f;
  int unused;
  long long int lli;

  custom_t() = default;
  custom_t(float f, long long int lli)
      : f(f)
      , unused(42)
      , lli(lli)
  {}
};

struct decomposer_t
{
  __host__ __device__ cuda::std::tuple<float&, long long int&> operator()(custom_t& key) const
  {
    return {key.f, key.lli};
  }
};
// example-end topk-custom-type

static __host__ std::ostream& operator<<(std::ostream& os, const custom_t& self)
{
  return os << "{ " << self.f << ", " << self.lli << " }";
}

static __host__ __device__ bool operator==(const custom_t& lhs, const custom_t& rhs)
{
  return lhs.f == rhs.f && lhs.lli == rhs.lli;
}

static __host__ __device__ bool operator<(const custom_t& lhs, const custom_t& rhs)
{
  return lhs.lli == rhs.lli ? lhs.f < rhs.f : lhs.lli < rhs.lli;
}

static __host__ __device__ bool operator>(const custom_t& lhs, const custom_t& rhs)
{
  return rhs < lhs;
}

C2H_TEST("DeviceTopK works with custom types and decomposer", "[device][topk]")
{
  SECTION("MaxKeys")
  {
    // example-begin topk-max-keys-custom-type
    constexpr int num_items = 6;
    constexpr int k         = 3;

    thrust::device_vector<custom_t> in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> out(k);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceTopK::MaxKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, k, decomposer_t{}, requirements);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Find the top-k largest keys
    cub::DeviceTopK::MaxKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, k, decomposer_t{}, requirements);

    // Sort output for comparison (output order is not guaranteed)
    thrust::sort(out.begin(), out.end(), cuda::std::greater<>{});
    thrust::device_vector<custom_t> expected = {
      {+3.7f, 5}, //
      {+2.5f, 4}, //
      {+1.1f, 3} //
    };
    // example-end topk-max-keys-custom-type

    REQUIRE(expected == out);
  }

  SECTION("MinKeys")
  {
    // example-begin topk-min-keys-custom-type
    constexpr int num_items = 6;
    constexpr int k         = 3;

    thrust::device_vector<custom_t> in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> out(k);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceTopK::MinKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, k, decomposer_t{}, requirements);

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceTopK::MinKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, k, decomposer_t{}, requirements);

    // Sort output for comparison (output order is not guaranteed)
    thrust::sort(out.begin(), out.end());
    thrust::device_vector<custom_t> expected = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2} //
    };
    // example-end topk-min-keys-custom-type

    REQUIRE(expected == out);
  }

  SECTION("MaxPairs")
  {
    // example-begin topk-max-pairs-custom-type
    constexpr int num_items = 6;
    constexpr int k         = 3;

    thrust::device_vector<custom_t> keys_in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> keys_out(k);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());

    thrust::device_vector<int> vals_in = {0, 1, 2, 3, 4, 5};
    thrust::device_vector<int> vals_out(k);

    const int* d_vals_in = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out      = thrust::raw_pointer_cast(vals_out.data());

    auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceTopK::MaxPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      k,
      decomposer_t{},
      requirements);

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceTopK::MaxPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      k,
      decomposer_t{},
      requirements);

    // Sort by key for comparison (output order is not guaranteed)
    thrust::sort_by_key(keys_out.begin(), keys_out.end(), vals_out.begin(), cuda::std::greater<>{});

    thrust::device_vector<custom_t> expected_keys = {
      {+3.7f, 5}, //
      {+2.5f, 4}, //
      {+1.1f, 3} //
    };

    thrust::device_vector<int> expected_vals = {5, 0, 2};
    // example-end topk-max-pairs-custom-type

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }

  SECTION("MinPairs")
  {
    // example-begin topk-min-pairs-custom-type
    constexpr int num_items = 6;
    constexpr int k         = 3;

    thrust::device_vector<custom_t> keys_in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> keys_out(k);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());

    thrust::device_vector<int> vals_in = {0, 1, 2, 3, 4, 5};
    thrust::device_vector<int> vals_out(k);

    const int* d_vals_in = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out      = thrust::raw_pointer_cast(vals_out.data());

    auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceTopK::MinPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      k,
      decomposer_t{},
      requirements);

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceTopK::MinPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      k,
      decomposer_t{},
      requirements);

    // Sort by key for comparison (output order is not guaranteed)
    thrust::sort_by_key(keys_out.begin(), keys_out.end(), vals_out.begin());

    thrust::device_vector<custom_t> expected_keys = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2} //
    };

    thrust::device_vector<int> expected_vals = {1, 3, 4};
    // example-end topk-min-pairs-custom-type

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }
}
