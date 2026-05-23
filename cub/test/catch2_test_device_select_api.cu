// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/memory.h>

#include <cstddef>

#include <c2h/catch2_test_helper.h>
#include <c2h/operator.cuh>

C2H_TEST("cub::DeviceSelect::FlaggedIf works with int data elements", "[select][device]")
{
  // example-begin segmented-select-flaggedif
  constexpr int num_items         = 8;
  c2h::device_vector<int> d_in    = {0, 1, 2, 3, 4, 5, 6, 7};
  c2h::device_vector<int> d_flags = {8, 6, 7, 5, 3, 0, 9, 3};
  c2h::device_vector<int> d_out(num_items);
  c2h::device_vector<int> d_num_selected_out(num_items);
  c2h::is_even;

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::FlaggedIf(
    nullptr,
    temp_storage_bytes,
    d_in.begin(),
    d_flags.begin(),
    d_out.begin(),
    d_num_selected_out.data(),
    num_items,
    c2h::is_even);

  // Allocate temporary storage
  c2h::device_vector<char> temp_storage(temp_storage_bytes);

  // Run selection
  cub::DeviceSelect::FlaggedIf(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_in.begin(),
    d_flags.begin(),
    d_out.begin(),
    d_num_selected_out.data(),
    num_items,
    c2h::is_even);

  c2h::device_vector<int> expected{0, 1, 5};
  // example-end segmented-select-flaggedif

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  d_out.resize(d_num_selected_out[0]);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSelect::FlaggedIf in-place works with int data elements", "[select][device]")
{
  // example-begin segmented-select-flaggedif-inplace
  constexpr int num_items         = 8;
  c2h::device_vector<int> d_data  = {0, 1, 2, 3, 4, 5, 6, 7};
  c2h::device_vector<int> d_flags = {8, 6, 7, 5, 3, 0, 9, 3};
  c2h::device_vector<int> d_num_selected_out(num_items);
  c2h::is_even;

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::FlaggedIf(
    nullptr, temp_storage_bytes, d_data.begin(), d_flags.begin(), d_num_selected_out.data(), num_items, c2h::is_even);

  // Allocate temporary storage
  c2h::device_vector<char> temp_storage(temp_storage_bytes);

  // Run selection
  cub::DeviceSelect::FlaggedIf(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_data.begin(),
    d_flags.begin(),
    d_num_selected_out.data(),
    num_items,
    c2h::is_even);

  c2h::device_vector<int> expected{0, 1, 5};
  // example-end segmented-select-flaggedif-inplace

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  d_data.resize(d_num_selected_out[0]);
  REQUIRE(d_data == expected);
}

C2H_TEST("cub::DeviceSelect::Unique in-place works with int data elements", "[select][device]")
{
  // example-begin select-unique-inplace
  constexpr int num_items                       = 8;
  thrust::device_vector<int> d_data             = {0, 2, 2, 9, 5, 5, 5, 8};
  thrust::device_vector<int> d_num_selected_out = {0};

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, d_data.begin(), d_num_selected_out.begin(), num_items);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  // Run selection
  cub::DeviceSelect::Unique(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_data.begin(),
    d_num_selected_out.begin(),
    num_items);

  // Resize input to new length
  d_data.resize(d_num_selected_out[0]);

  thrust::device_vector<int> expected{0, 2, 9, 5, 8};
  // example-end select-unique-inplace

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  REQUIRE(d_data == expected);
}

// example-begin select-unique-inplace-eqop-myequalityop
struct my_equality_op
{
  __host__ __device__ bool operator()(int lhs, int rhs) const
  {
    return lhs == rhs;
  }
};
// example-end select-unique-inplace-eqop-myequalityop

C2H_TEST("cub::DeviceSelect::Unique in-place with equality_op works with int data elements", "[select][device]")
{
  // example-begin select-unique-inplace-eqop
  constexpr int num_items                       = 8;
  thrust::device_vector<int> d_data             = {0, 2, 2, 9, 5, 5, 5, 8};
  thrust::device_vector<int> d_num_selected_out = {0};
  my_equality_op equality_op{};

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Unique(
    nullptr, temp_storage_bytes, d_data.begin(), d_num_selected_out.begin(), num_items, equality_op);

  // Allocate temporary storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  // Run selection
  cub::DeviceSelect::Unique(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_data.begin(),
    d_num_selected_out.begin(),
    num_items,
    equality_op);

  // Resize input to new length
  d_data.resize(d_num_selected_out[0]);

  thrust::device_vector<int> expected{0, 2, 9, 5, 8};
  // example-end select-unique-inplace-eqop

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  REQUIRE(d_data == expected);
}

// Guard tests: each public DeviceSelect method must resolve unambiguously
// to the legacy temp-storage overload when called in its minimal form
// (no explicit stream, all defaults left implicit), even though the env
// overloads are also in scope. If env-overload SFINAE drifts, these become
// "ambiguous overload" compile errors.

struct select_always_true_t
{
  __host__ __device__ bool operator()(int) const
  {
    return true;
  }
};

C2H_TEST("DeviceSelect::Flagged legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_flags              = nullptr;
  int* d_out                = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, n));
}

C2H_TEST("DeviceSelect::Flagged in-place legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_data               = nullptr;
  int* d_flags              = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(
    cudaSuccess == cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_data, d_flags, d_num_selected, n));
}

C2H_TEST("DeviceSelect::If legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_out                = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, n, select_always_true_t{}));
}

C2H_TEST("DeviceSelect::If in-place legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_data               = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_data, d_num_selected, n, select_always_true_t{}));
}

C2H_TEST("DeviceSelect::FlaggedIf legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_flags              = nullptr;
  int* d_out                = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::FlaggedIf(
            d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, n, select_always_true_t{}));
}

C2H_TEST("DeviceSelect::FlaggedIf in-place legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_data               = nullptr;
  int* d_flags              = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::FlaggedIf(
            d_temp_storage, temp_storage_bytes, d_data, d_flags, d_num_selected, n, select_always_true_t{}));
}

C2H_TEST("DeviceSelect::Unique legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_out                = nullptr;
  int* d_num_selected       = nullptr;
  ::cuda::std::int64_t n    = 0;

  REQUIRE(cudaSuccess == cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, n));
}

C2H_TEST("DeviceSelect::UniqueByKey legacy size-query is unambiguous", "[select][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_keys_in            = nullptr;
  int* d_values_in          = nullptr;
  int* d_keys_out           = nullptr;
  int* d_values_out         = nullptr;
  int* d_num_selected       = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::UniqueByKey(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected, n));
}
