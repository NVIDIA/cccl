// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/execution>

#include <algorithm>

#include <c2h/catch2_test_helper.h>

template <class T>
inline T to_bound(const unsigned long long bound)
{
  return static_cast<T>(bound);
}

template <>
inline ulonglong2 to_bound(const unsigned long long bound)
{
  return {bound, bound};
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
template <>
inline ulonglong4 to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
_CCCL_SUPPRESS_DEPRECATED_POP

#if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline ulonglong4_16a to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline long2 to_bound(const unsigned long long bound)
{
  return {static_cast<long>(bound), static_cast<long>(bound)};
}

template <>
inline c2h::custom_type_t<c2h::equal_comparable_t> to_bound(const unsigned long long bound)
{
  c2h::custom_type_t<c2h::equal_comparable_t> val;
  val.key = bound;
  val.val = bound;
  return val;
}
template <typename EqualityOpT>
struct project_first
{
  EqualityOpT equality_op;
  template <typename Tuple>
  __host__ __device__ bool operator()(const Tuple& lhs, const Tuple& rhs) const
  {
    return equality_op(cuda::std::get<0>(lhs), cuda::std::get<0>(rhs));
  }
};

template <typename T>
struct custom_equality_op
{
  T div_val;
  __host__ __device__ __forceinline__ bool operator()(const T& lhs, const T& rhs) const
  {
    return (lhs / div_val) == (rhs / div_val);
  }
};

C2H_TEST("DeviceSelect::UniqueByKey works with user provided memory and environment", "[device][select_unique_by_key]")
{
  using type     = int;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_in(num_items, thrust::default_init);
  c2h::device_vector<type> keys_out(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_out(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::UniqueByKey(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    keys_in.begin(),
    vals_in.begin(),
    keys_out.begin(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_unique_by_key = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::UniqueByKey(
      static_cast<void*>(nullptr),
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::UniqueByKey(
      temp_storage,
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Ensure that we create the same output as std
    c2h::host_vector<type> reference_keys     = keys_in;
    c2h::host_vector<val_type> reference_vals = vals_in;
    const auto zip_begin                      = cuda::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
    const auto zip_end                        = cuda::make_zip_iterator(reference_keys.end(), reference_vals.end());
    const auto boundary =
      std::unique(zip_begin, zip_end, project_first<cuda::std::equal_to<>>{cuda::std::equal_to<>{}});

    keys_out.resize(num_selected_out[0]);
    vals_out.resize(num_selected_out[0]);
    reference_keys.resize(num_selected_out[0]);
    reference_vals.resize(num_selected_out[0]);
    REQUIRE(static_cast<int>(boundary - zip_begin) == num_selected_out[0]);
    REQUIRE(reference_keys == keys_out);
    REQUIRE(reference_vals == vals_out);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceSelect::UniqueByKey works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_unique_by_key(stream.get());
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_unique_by_key(stream);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_unique_by_key(stream_ref);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_unique_by_key(env);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_unique_by_key(policy);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_unique_by_key(policy);
  }
}

C2H_TEST("DeviceSelect::UniqueByKey works with user provided operator, memory and environment",
         "[device][select_unique_by_key]")
{
  using type        = int;
  using custom_op_t = custom_equality_op<type>;
  using val_type    = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_in(num_items, thrust::default_init);
  c2h::device_vector<type> keys_out(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_out(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  custom_op_t op{static_cast<type>(8)};

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::UniqueByKey(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    keys_in.begin(),
    vals_in.begin(),
    keys_out.begin(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items,
    op);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_unique_by_key = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::UniqueByKey(
      static_cast<void*>(nullptr),
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      op,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::UniqueByKey(
      temp_storage,
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      op,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Ensure that we create the same output as std
    c2h::host_vector<type> reference_keys     = keys_in;
    c2h::host_vector<val_type> reference_vals = vals_in;
    const auto zip_begin                      = cuda::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
    const auto zip_end                        = cuda::make_zip_iterator(reference_keys.end(), reference_vals.end());
    const auto boundary                       = std::unique(zip_begin, zip_end, project_first<custom_op_t>{op});

    keys_out.resize(num_selected_out[0]);
    vals_out.resize(num_selected_out[0]);
    reference_keys.resize(num_selected_out[0]);
    reference_vals.resize(num_selected_out[0]);
    REQUIRE(static_cast<int>(boundary - zip_begin) == num_selected_out[0]);
    REQUIRE(reference_keys == keys_out);
    REQUIRE(reference_vals == vals_out);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceSelect::UniqueByKey works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_unique_by_key(stream.get());
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_unique_by_key(stream);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_unique_by_key(stream_ref);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_unique_by_key(env);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_unique_by_key(policy);
  }

  SECTION("DeviceSelect::UniqueByKey works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_unique_by_key(policy);
  }
}
