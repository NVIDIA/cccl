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

#include <memory>

#include "CommonHelpers.h"
#include "test_macros.h"

// Idiosyncratic element type for mdspan
// Make sure we don't assume copyable, default constructible, movable etc.
struct MinimalElementType
{
  int val;
  constexpr MinimalElementType()                                     = delete;
  constexpr MinimalElementType(const MinimalElementType&)            = delete;
  constexpr MinimalElementType& operator=(const MinimalElementType&) = delete;
  __host__ __device__ constexpr explicit MinimalElementType(int v) noexcept
      : val(v)
  {}
};

// Helper class to create pointer to MinimalElementType
template <class T, size_t N>
struct ElementPool
{
  __host__ __device__ TEST_CONSTEXPR_CXX20 ElementPool()
  {
#if TEST_STD_VER >= 2020
    if (cuda::std::is_constant_evaluated())
    {
      // clang-format off
      NV_IF_TARGET(NV_IS_HOST, (
        std::construct_at(&constexpr_ptr_, std::allocator<cuda::std::remove_const_t<T>>{}.allocate(N));
        for (int i = 0; i != N;++i)
        {
          std::construct_at(constexpr_ptr_ + i, 42);
        }
      ))
      // clang-format on
    }
    else
#endif // TEST_STD_VER >= 2020
    {
      T* ptr = reinterpret_cast<T*>(ptr_);
      for (int i = 0; i != N; ++i)
      {
        cuda::std::__construct_at(ptr + i, 42);
      }
    }
  }

  __host__ __device__ constexpr T* get_ptr()
  {
#if TEST_STD_VER >= 2020
    if (cuda::std::is_constant_evaluated())
    {
      // clang-format off
      NV_IF_ELSE_TARGET(NV_IS_HOST, (
        return constexpr_ptr_;
      ),(
        return nullptr;
      ))
      // clang-format on
    }
    else
#endif // TEST_STD_VER >= 2020
    {
      return reinterpret_cast<T*>(ptr_);
    }
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 ~ElementPool()
  {
#if TEST_STD_VER >= 2020
    if (cuda::std::is_constant_evaluated())
    {
      // clang-format off
      NV_IF_TARGET(NV_IS_HOST,(
        for (int i = 0; i != N; ++i) {
          std::destroy_at(constexpr_ptr_ + i);
        }
        std::allocator<cuda::std::remove_const_t<T>>{}.deallocate(constexpr_ptr_, N);
      ))
      return;
      // clang-format on
    }
    else
#endif // TEST_STD_VER >= 2020
    {
      for (int i = 0; i != N; ++i)
      {
        cuda::std::__destroy_at(ptr_ + i);
      }
    }
  }

private:
  union
  {
    char ptr_[N * sizeof(T)] = {};
    cuda::std::remove_const_t<T>* constexpr_ptr_;
  };
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
