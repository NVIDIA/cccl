//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class F, class I1, class I2 = I1>
// concept indirect_strict_weak_order;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "indirectly_readable.h"
#include "test_macros.h"

using It1 = IndirectlyReadable<struct Token1>;
using It2 = IndirectlyReadable<struct Token2>;

template <class I1, class I2>
struct GoodOrder
{
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_value_t<I1>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I2>&, cuda::std::iter_value_t<I2>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_value_t<I2>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I2>&, cuda::std::iter_value_t<I1>&) const;

  __host__ __device__ bool operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_reference_t<I2>) const;
  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I2>, cuda::std::iter_value_t<I1>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I2>, cuda::std::iter_reference_t<I2>) const;

  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_value_t<I2>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I2>&, cuda::std::iter_reference_t<I1>) const;
  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_reference_t<I1>) const;

  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_reference_t<I2>) const;
  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I2>, cuda::std::iter_reference_t<I1>) const;

  __host__ __device__ bool
  operator()(cuda::std::iter_common_reference_t<I1>, cuda::std::iter_common_reference_t<I1>) const;
  __host__ __device__ bool
  operator()(cuda::std::iter_common_reference_t<I2>, cuda::std::iter_common_reference_t<I2>) const;
  __host__ __device__ bool
  operator()(cuda::std::iter_common_reference_t<I1>, cuda::std::iter_common_reference_t<I2>) const;
  __host__ __device__ bool
  operator()(cuda::std::iter_common_reference_t<I2>, cuda::std::iter_common_reference_t<I1>) const;
};

// Should work when all constraints are satisfied
static_assert(cuda::std::indirect_strict_weak_order<GoodOrder<It1, It2>, It1, It2>);
static_assert(cuda::std::indirect_strict_weak_order<bool (*)(int, long), int*, long*>);

#ifdef TEST_COMPILER_CLANG_CUDA
#  pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#endif // TEST_COMPILER_CLANG_CUDA
#ifndef __CUDA_ARCH__
auto lambda = [](int i, long j) {
  return i == j;
};
static_assert(cuda::std::indirect_strict_weak_order<decltype(lambda), int*, long*>);
#endif

// Should fail when either of the iterators is not indirectly_readable
#if TEST_STD_VER > 2017
struct NotIndirectlyReadable
{};
static_assert(!cuda::std::indirect_strict_weak_order<GoodOrder<It1, NotIndirectlyReadable>, It1, NotIndirectlyReadable>);
static_assert(!cuda::std::indirect_strict_weak_order<GoodOrder<NotIndirectlyReadable, It2>, NotIndirectlyReadable, It2>);
#endif

// Should fail when the function is not copy constructible
struct BadOrder1
{
  BadOrder1(BadOrder1 const&) = delete;
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder1, It1, It2>);

// Should fail when the function can't be called with (iter_value_t&, iter_value_t&)
struct BadOrder2
{
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
  bool operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder2, It1, It2>);

// Should fail when the function can't be called with (iter_value_t&, iter_reference_t)
struct BadOrder3
{
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
  bool operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder3, It1, It2>);

// Should fail when the function can't be called with (iter_reference_t, iter_value_t&)
struct BadOrder4
{
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
  bool operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder4, It1, It2>);

// Should fail when the function can't be called with (iter_reference_t, iter_reference_t)
struct BadOrder5
{
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
  bool operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder5, It1, It2>);

// Should fail when the function can't be called with (iter_common_reference_t, iter_common_reference_t)
struct BadOrder6
{
  template <class T, class U>
  __host__ __device__ bool operator()(T const&, U const&) const;
  bool operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_strict_weak_order<BadOrder6, It1, It2>);

int main(int, char**)
{
  return 0;
}
