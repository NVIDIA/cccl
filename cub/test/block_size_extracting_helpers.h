// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/iterator>
#include <cuda/std/utility>

// Wrapper for an operation used to extract the thread block size. Useful to check if a tuning was applied.
template <typename InnerOp>
struct block_size_extracting_op
{
  unsigned int* ptr;

  template <typename... Ts>
  __device__ auto operator()(Ts&&... args) const -> decltype(InnerOp{}(::cuda::std::forward<Ts>(args)...))
  {
    atomicMax(ptr, blockDim.x); // not every thread may reach this, so avoid guarding the atomic by threadIdx
    return InnerOp{}(::cuda::std::forward<Ts>(args)...);
  }
};

// Iterator used to extract the thread block size. Useful to check if a tuning was applied.
struct block_size_extracting_constant_iterator
{
  using value_type        = int;
  using reference         = int;
  using pointer           = int*;
  using difference_type   = ptrdiff_t;
  using iterator_category = ::cuda::std::random_access_iterator_tag;

  int value;
  unsigned int* block_size_ptr;
  difference_type offset;

  __host__ __device__ block_size_extracting_constant_iterator(int val, unsigned int* bs_ptr, difference_type off = 0)
      : value(val)
      , block_size_ptr(bs_ptr)
      , offset(off)
  {}

  __device__ reference operator[](difference_type) const
  {
    atomicMax(block_size_ptr, blockDim.x); // not every thread may reach this, so avoid guarding the atomic by threadIdx
    return value;
  }

  __device__ reference operator*() const
  {
    atomicMax(block_size_ptr, blockDim.x); // not every thread may reach this, so avoid guarding the atomic by threadIdx
    return value;
  }

  __host__ __device__ block_size_extracting_constant_iterator operator+(difference_type n) const
  {
    return {value, block_size_ptr, offset + n};
  }

  __host__ __device__ block_size_extracting_constant_iterator& operator+=(difference_type n)
  {
    offset += n;
    return *this;
  }

  __host__ __device__ difference_type operator-(const block_size_extracting_constant_iterator& other) const
  {
    return offset - other.offset;
  }

  __host__ __device__ bool operator==(const block_size_extracting_constant_iterator& other) const
  {
    return offset == other.offset;
  }

  __host__ __device__ bool operator!=(const block_size_extracting_constant_iterator& other) const
  {
    return offset != other.offset;
  }
};
