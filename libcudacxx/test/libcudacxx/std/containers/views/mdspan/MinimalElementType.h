//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H

#include <cuda/std/utility>

#include "CommonHelpers.h"
#include "test_macros.h"

// Idiosyncratic element type for mdspan
// Make sure we don't assume copyable, default constructible, movable etc.
struct MinimalElementType
{
  template <class T, size_t N>
  friend struct ElementPool;

  struct tag
  {
    __host__ __device__ constexpr tag(int) noexcept {}

    __host__ __device__ constexpr operator int() noexcept
    {
      return 42;
    }
  };

  int val;
  constexpr MinimalElementType()                                     = delete;
  constexpr MinimalElementType& operator=(const MinimalElementType&) = delete;
  __host__ __device__ constexpr explicit MinimalElementType(int v) noexcept
      : val(v)
  {}

  __host__ __device__ constexpr MinimalElementType(tag) noexcept
      : val(42)
  {}

  // MSVC cannot list init the element and complains about the deleted copy constructor. Emulate via private
#if _CCCL_COMPILER(MSVC)

private:
  constexpr MinimalElementType(const MinimalElementType&) = default;
#else // ^^^ _CCCL_COMPILER(MSVC2019) ^^^ / vvv !_CCCL_COMPILER(MSVC2019) vvv
  constexpr MinimalElementType(const MinimalElementType&) = delete;
#endif // !_CCCL_COMPILER(MSVC2019)
};

// Helper class to create pointer to MinimalElementType
template <class T, size_t N>
struct ElementPool
{
private:
  template <int... Indices>
  __host__ __device__ constexpr ElementPool(cuda::std::integer_sequence<int, Indices...>)
      : ptr_{T{MinimalElementType::tag{Indices}}...}
  {}

public:
  __host__ __device__ constexpr ElementPool()
      : ElementPool(cuda::std::make_integer_sequence<int, N>())
  {}

  __host__ __device__ constexpr T* get_ptr()
  {
    return ptr_;
  }

private:
  T ptr_[N];
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
