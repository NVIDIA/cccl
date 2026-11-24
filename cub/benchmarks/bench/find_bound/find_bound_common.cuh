// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

//! Shared setup for `cub::DeviceFind` bounds benchmarks. Data layout and
//! generation mirror `thrust/benchmarks/bench/vectorized_search/{lower,upper}_bound.cu`
//! (Elements pow2 16..28 step 4, int8..int64, NeedlesRatio {1, 25, 50}).

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cstddef>

#include <nvbench_helper.cuh>

template <typename T>
struct bounds_bench_data
{
  thrust::device_vector<T> data{};
  thrust::device_vector<std::ptrdiff_t> result{};
  std::size_t elements{};
  std::size_t needles{};

  _CCCL_HOST_API explicit bounds_bench_data(nvbench::state& state)
  {
    elements                 = static_cast<std::size_t>(state.get_int64("Elements{io}"));
    const auto needles_ratio = static_cast<std::size_t>(state.get_int64("NeedlesRatio"));
    needles                  = needles_ratio * static_cast<std::size_t>(static_cast<double>(elements) / 100.0);

    data   = generate(elements + needles);
    result = thrust::device_vector<std::ptrdiff_t>(needles, thrust::no_init);

    thrust::sort(data.begin(),
                 data.begin() + static_cast<typename thrust::device_vector<T>::difference_type>(elements));
  }

  _CCCL_HOST_API void sort_needles()
  {
    thrust::sort(data.begin() + static_cast<typename thrust::device_vector<T>::difference_type>(elements), data.end());
  }

  _CCCL_HOST_API T* range_ptr()
  {
    return thrust::raw_pointer_cast(data.data());
  }

  _CCCL_HOST_API T* values_ptr()
  {
    return thrust::raw_pointer_cast(
      data.data() + static_cast<typename thrust::device_vector<T>::difference_type>(elements));
  }

  _CCCL_HOST_API std::ptrdiff_t* output_ptr()
  {
    return thrust::raw_pointer_cast(result.data());
  }
};
