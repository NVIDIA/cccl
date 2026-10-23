// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>

#include <cuda/std/cstdint>

#include <algorithm>
#include <iterator>
#include <numeric>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

using tile_state_t = cub::ScanTileState<cuda::std::int32_t>;
using count_t      = cuda::std::int64_t;
using init_pointer = void (*)(tile_state_t, int, count_t*);

int init_getter_calls = 0;

// A non-constexpr specialization permits the host launcher to be defined in another translation unit.
template <>
CUB_RUNTIME_FUNCTION init_pointer cub::detail::scan::DeviceCompactInitKernelSource<tile_state_t, count_t*>::InitKernel()
{
  NV_IF_TARGET(NV_IS_HOST, (++init_getter_calls;))
  return &cub::detail::scan::DeviceCompactInitKernel<tile_state_t, count_t*>;
}

using default_source_t = cub::detail::scan::DeviceCompactInitKernelSource<tile_state_t, cuda::std::int32_t*>;
static_assert(default_source_t::InitKernel()
              == &cub::detail::scan::DeviceCompactInitKernel<tile_state_t, cuda::std::int32_t*>);

struct is_even
{
  __host__ __device__ bool operator()(cuda::std::int32_t value) const
  {
    return value % 2 == 0;
  }
};

CUB_TEST("Select uses the shared compact initialization kernel source", "[select][device]", CUB_SMALL)
{
  using value_t       = cuda::std::int32_t;
  const int num_items = GENERATE(0, 1, 31, 65'537);

  c2h::host_vector<value_t> h_input(num_items);
  std::iota(h_input.begin(), h_input.end(), value_t{0});
  c2h::host_vector<value_t> expected;
  std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected), is_even{});

  c2h::device_vector<value_t> d_input = h_input;
  c2h::device_vector<value_t> d_output(num_items);
  c2h::device_vector<count_t> d_count(1);

  auto* input  = thrust::raw_pointer_cast(d_input.data());
  auto* output = thrust::raw_pointer_cast(d_output.data());
  auto* count  = thrust::raw_pointer_cast(d_count.data());

  init_getter_calls    = 0;
  size_t storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceSelect::If(nullptr, storage_bytes, input, output, count, num_items, is_even{}));
  c2h::device_vector<char> storage(storage_bytes);
  REQUIRE(cudaSuccess
          == cub::DeviceSelect::If(
            thrust::raw_pointer_cast(storage.data()), storage_bytes, input, output, count, num_items, is_even{}));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<value_t> actual = d_output;
  actual.resize(static_cast<size_t>(d_count[0]));
  REQUIRE(actual == expected);
  REQUIRE(init_getter_calls > 0);
}
