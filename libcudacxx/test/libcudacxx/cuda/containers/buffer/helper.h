//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TEST_CONTAINER_VECTOR_HELPER_H
#define CUDA_TEST_CONTAINER_VECTOR_HELPER_H

#include <thrust/equal.h>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_resources.h"

// Default data to compare against

inline constexpr ::cuda::std::initializer_list<int> compare_data_initializer_list{1, 42, 1337, 0, 12, -1};
__device__ constexpr int device_data[] = {1, 42, 1337, 0, 12, -1};
constexpr int host_data[]              = {1, 42, 1337, 0, 12, -1};

template <typename Iter>
__global__ void check_equal_kernel(Iter ptr)
{
  for (int i = 0; i < cuda::std::size(device_data); i++)
  {
    if (ptr[i] != device_data[i])
    {
      __trap();
    }
  }
}

template <typename Iter, typename Val>
__global__ void check_equal_value_kernel(Iter ptr, size_t size, Val value)
{
  for (size_t i = 0; i < size; i++)
  {
    if (ptr[i] != value)
    {
      __trap();
    }
  }
}

template <class Buffer>
bool equal_range(const Buffer& buf)
{
  if constexpr (Buffer::properties_list::has_property(cuda::mr::host_accessible{}))
  {
    buf.stream().sync();
    return cuda::std::equal(buf.begin(), buf.end(), cuda::std::begin(host_data), cuda::std::end(host_data));
  }
  else
  {
    if (buf.size() != cuda::std::size(device_data))
    {
      return false;
    }
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    check_equal_kernel<<<1, 1, 0, buf.stream().get()>>>(buf.begin());
    CCCLRT_CHECK(cudaGetLastError() == cudaSuccess);
    buf.stream().sync();
    return true;
  }
}

template <class Buffer, class T>
bool compare_value(const T& value, const T& expected)
{
  if constexpr (!Buffer::properties_list::has_property(cuda::mr::device_accessible{}))
  {
    return value == expected;
  }
  else
  {
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    // copy the value to host
    T host_value;
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpy,
      "failed to copy value",
      cuda::std::addressof(host_value),
      cuda::std::addressof(value),
      sizeof(T),
      ::cudaMemcpyDefault);
    return host_value == expected;
  }
}

template <class Buffer, class T>
void assign_value(T& value, const T& input)
{
  if constexpr (!Buffer::properties_list::has_property(cuda::mr::device_accessible{}))
  {
    value = input;
  }
  else
  {
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    // copy the input to device
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpy,
      "failed to copy value",
      cuda::std::addressof(value),
      cuda::std::addressof(input),
      sizeof(T),
      ::cudaMemcpyDefault);
  }
}

// Helper to compare a range with all equal values
template <class T>
struct equal_to_value
{
  T value_;

  explicit equal_to_value(T value) noexcept
      : value_(value)
  {}

  __host__ __device__ bool operator()(const T lhs, const T) const noexcept
  {
    return lhs == value_;
  }
};

template <class Buffer>
bool equal_size_value(const Buffer& buf, const size_t size, const int value)
{
  if constexpr (Buffer::properties_list::has_property(cuda::mr::host_accessible{}))
  {
    buf.stream().sync();
    return buf.size() == size
        && cuda::std::equal(buf.begin(),
                            buf.end(),
                            cuda::std::begin(host_data),
                            equal_to_value{static_cast<typename Buffer::value_type>(value)});
  }
  else
  {
    if (buf.size() != size)
    {
      return false;
    }
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    check_equal_value_kernel<<<1, 1, 0, buf.stream().get()>>>(buf.begin(), size, value);
    CCCLRT_CHECK(cudaGetLastError() == cudaSuccess);
    buf.stream().sync();
    return true;
  }
}

// Helper function to compare two ranges
template <class Range1, class Range2>
bool equal_range(const Range1& range1, const Range2& range2)
{
  if constexpr (!Range1::properties_list::has_property(cuda::mr::device_accessible{}))
  {
    range1.stream().sync();
    return cuda::std::equal(range1.begin(), range1.end(), range2.begin(), range2.end());
  }
  else
  {
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    return range1.size() == range2.size()
        && thrust::equal(thrust::cuda::par.on(range1.stream().get()), range1.begin(), range1.end(), range2.begin());
  }
}

// helper class as we need to pass the properties in a tuple to the catch tests
template <class>
struct extract_properties;

template <class T, class... Properties>
struct extract_properties<cuda::std::tuple<T, Properties...>>
{
  static auto get_resource()
  {
    if constexpr (cuda::mr::__is_host_accessible<Properties...>)
    {
#if _CCCL_CTK_AT_LEAST(12, 6)
      return offset_by_alignment_resource(cuda::pinned_default_memory_pool());
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 6) ^^^ / vvv _CCCL_CTK_BELOW(12, 6) vvv
      return offset_by_alignment_resource(cuda::device_default_memory_pool(cuda::device_ref{0}));
#endif // ^^^ _CCCL_CTK_BELOW(12, 6) ^^^
    }
    else
    {
      return offset_by_alignment_resource(cuda::device_default_memory_pool(cuda::device_ref{0}));
    }
  }

  using buffer         = cuda::buffer<T, Properties...>;
  using resource       = decltype(get_resource());
  using iterator       = cuda::heterogeneous_iterator<T, Properties...>;
  using const_iterator = cuda::heterogeneous_iterator<const T, Properties...>;

  using matching_vector   = cuda::buffer<T, other_property, Properties...>;
  using matching_resource = memory_resource_wrapper<other_property, Properties...>;
};

#endif // CUDA_TEST_CONTAINER_VECTOR_HELPER_H
