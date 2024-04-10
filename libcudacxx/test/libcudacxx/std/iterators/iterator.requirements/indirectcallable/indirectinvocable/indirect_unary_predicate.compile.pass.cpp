//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class F, class I>
// concept indirect_unary_predicate;

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "indirectly_readable.h"
#include "test_macros.h"

using It = IndirectlyReadable<struct Token>;

template <class I>
struct GoodPredicate
{
  __host__ __device__ bool operator()(cuda::std::iter_reference_t<I>) const;
  __host__ __device__ bool operator()(cuda::std::iter_value_t<I>&) const;
  __host__ __device__ bool operator()(cuda::std::iter_common_reference_t<I>) const;
};

// Should work when all constraints are satisfied
static_assert(cuda::std::indirect_unary_predicate<GoodPredicate<It>, It>);
static_assert(cuda::std::indirect_unary_predicate<bool (*)(int), int*>);

#ifdef TEST_COMPILER_CLANG_CUDA
#  pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#endif // TEST_COMPILER_CLANG_CUDA
#ifndef __CUDA_ARCH__
auto lambda = [](int i) {
  return i % 2 == 0;
};
static_assert(cuda::std::indirect_unary_predicate<decltype(lambda), int*>);
#endif

// Should fail when the iterator is not indirectly_readable
#if TEST_STD_VER > 2017
struct NotIndirectlyReadable
{};
static_assert(!cuda::std::indirect_unary_predicate<GoodPredicate<NotIndirectlyReadable>, NotIndirectlyReadable>);
#endif

// Should fail when the predicate is not copy constructible
struct BadPredicate1
{
  BadPredicate1(BadPredicate1 const&) = delete;
  template <class T>
  __host__ __device__ bool operator()(T const&) const;
};
static_assert(!cuda::std::indirect_unary_predicate<BadPredicate1, It>);

// Should fail when the predicate can't be called with cuda::std::iter_value_t<It>&
struct BadPredicate2
{
  template <class T>
  __host__ __device__ bool operator()(T const&) const;
  bool operator()(cuda::std::iter_value_t<It>&) const = delete;
};
static_assert(!cuda::std::indirect_unary_predicate<BadPredicate2, It>);

// Should fail when the predicate can't be called with cuda::std::iter_reference_t<It>
struct BadPredicate3
{
  template <class T>
  __host__ __device__ bool operator()(T const&) const;
  bool operator()(cuda::std::iter_reference_t<It>) const = delete;
};
static_assert(!cuda::std::indirect_unary_predicate<BadPredicate3, It>);

// Should fail when the predicate can't be called with cuda::std::iter_common_reference_t<It>
struct BadPredicate4
{
  template <class T>
  __host__ __device__ bool operator()(T const&) const;
  bool operator()(cuda::std::iter_common_reference_t<It>) const = delete;
};
static_assert(!cuda::std::indirect_unary_predicate<BadPredicate4, It>);

int main(int, char**)
{
  return 0;
}
