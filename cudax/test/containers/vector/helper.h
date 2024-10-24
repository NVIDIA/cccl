//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_CONTAINER_VECTOR_HELPER_H
#define CUDAX_TEST_CONTAINER_VECTOR_HELPER_H

#include <thrust/equal.h>

#include <cuda/functional>
#include <cuda/std/__algorithm_>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_resources.h"

namespace cudax = cuda::experimental;

// Default data to compare against
__device__ constexpr int device_data[] = {1, 42, 1337, 0, 12, -1};
constexpr int host_data[]              = {1, 42, 1337, 0, 12, -1};

template <class Vector>
constexpr bool equal_range(const Vector& vec)
{
  _CCCL_IF_CONSTEXPR (Vector::__is_host_only)
  {
    return cuda::std::equal(vec.begin(), vec.end(), cuda::std::begin(host_data), cuda::std::end(host_data));
  }
  else
  {
    return vec.size() == cuda::std::size(device_data)
        && thrust::equal(thrust::device, vec.begin(), vec.end(), cuda::get_device_address(device_data[0]));
  }
}

template <bool HostOnly, class T>
constexpr bool compare_value(const T& value, const T& expected)
{
  _CCCL_IF_CONSTEXPR (HostOnly)
  {
    return value == expected;
  }
  else
  {
    // copy the value to host
    T host_value;
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpy,
      "failed to copy value",
      cuda::std::addressof(host_value),
      cuda::std::addressof(value),
      sizeof(T),
      ::cudaMemcpyDeviceToHost);
    return host_value == expected;
  }
}

template <bool HostOnly, class T>
void assign_value(T& value, const T& input)
{
  _CCCL_IF_CONSTEXPR (HostOnly)
  {
    value = input;
  }
  else
  {
    // copy the input to device
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpy,
      "failed to copy value",
      cuda::std::addressof(value),
      cuda::std::addressof(input),
      sizeof(T),
      ::cudaMemcpyHostToDevice);
  }
}

// Helper to compare a range with all equal values
struct equal_to_value
{
  int value_;

  template <class T>
  __host__ __device__ bool operator()(const T lhs, const T) const noexcept
  {
    return lhs == static_cast<T>(value_);
  }
};

template <class Vector>
constexpr bool equal_size_value(const Vector& vec, const size_t size, const int value)
{
  _CCCL_IF_CONSTEXPR (Vector::__is_host_only)
  {
    return vec.size() == size
        && cuda::std::equal(vec.begin(), vec.end(), cuda::std::begin(host_data), equal_to_value{value});
  }
  else
  {
    return vec.size() == size
        && thrust::equal(thrust::device, vec.begin(), vec.end(), cuda::std::begin(device_data), equal_to_value{value});
  }
}

// Helper function to compare two ranges
template <class Range1, class Range2>
constexpr bool equal_range(const Range1& range1, const Range2& range2)
{
  _CCCL_IF_CONSTEXPR (Range1::__is_host_only)
  {
    return cuda::std::equal(range1.begin(), range1.end(), range2.begin(), range2.end());
  }
  else
  {
    return range1.size() == range2.size()
        && thrust::equal(thrust::device, range1.begin(), range1.end(), range2.begin());
  }
}

// helper class as we need to pass the properties in a tuple to the catch tests
template <class>
struct extract_properties;

template <class... Properties>
struct extract_properties<cuda::std::tuple<Properties...>>
{
  using vector = cudax::vector<int, Properties...>;
  using resource =
    caching_resource<cuda::std::conditional_t<cuda::mr::__is_host_device_accessible<Properties...>,
                                              cuda::mr::pinned_memory_resource,
                                              cuda::std::conditional_t<cuda::mr::__is_host_accessible<Properties...>,
                                                                       host_memory_resource<int>,
                                                                       cuda::mr::device_memory_resource>>>;
  using resource_ref   = cuda::mr::resource_ref<Properties...>;
  using iterator       = cudax::heterogeneous_iterator<int, false, Properties...>;
  using const_iterator = cudax::heterogeneous_iterator<int, true, Properties...>;

  using matching_vector       = cudax::vector<int, other_property, Properties...>;
  using matching_resource     = memory_resource_wrapper<other_property, Properties...>;
  using matching_resource_ref = cuda::mr::resource_ref<other_property, Properties...>;
};

#endif // CUDAX_TEST_CONTAINER_VECTOR_HELPER_H
