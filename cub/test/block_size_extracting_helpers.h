// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/utility>

// Wrapper for an operation used to extract the thread block size. Useful to check if a tuning was applied.
template <typename InnerOp>
struct block_size_extracting_op
{
  unsigned int* ptr;

  template <typename... Ts>
  [[nodiscard]] _CCCL_DEVICE_API auto operator()(Ts&&... args) const
    -> decltype(InnerOp{}(::cuda::std::forward<Ts>(args)...))
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
  using difference_type   = ::cuda::std::ptrdiff_t;
  using iterator_category = ::cuda::std::random_access_iterator_tag;

  int value;
  unsigned int* block_size_ptr;
  difference_type offset;

  _CCCL_API block_size_extracting_constant_iterator(int val, unsigned int* bs_ptr, difference_type off = 0) noexcept
      : value(val)
      , block_size_ptr(bs_ptr)
      , offset(off)
  {}

  [[nodiscard]] _CCCL_DEVICE_API reference operator[](difference_type) const
  {
    atomicMax(block_size_ptr, blockDim.x); // not every thread may reach this, so avoid guarding the atomic by threadIdx
    return value;
  }

  [[nodiscard]] _CCCL_DEVICE_API reference operator*() const
  {
    atomicMax(block_size_ptr, blockDim.x); // not every thread may reach this, so avoid guarding the atomic by threadIdx
    return value;
  }

  [[nodiscard]] _CCCL_API block_size_extracting_constant_iterator operator+(difference_type n) const noexcept
  {
    return {value, block_size_ptr, offset + n};
  }

  _CCCL_API block_size_extracting_constant_iterator& operator+=(difference_type n) noexcept
  {
    offset += n;
    return *this;
  }

  [[nodiscard]] _CCCL_API difference_type operator-(const block_size_extracting_constant_iterator& other) const noexcept
  {
    return offset - other.offset;
  }

  [[nodiscard]] _CCCL_API bool operator==(const block_size_extracting_constant_iterator& other) const noexcept
  {
    return offset == other.offset;
  }

  [[nodiscard]] _CCCL_API bool operator!=(const block_size_extracting_constant_iterator& other) const noexcept
  {
    return offset != other.offset;
  }
};
