//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class F, class I1, class I2>
// concept indirect_binary_predicate;

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "indirectly_readable.h"

using It1 = IndirectlyReadable<struct Token1>;
using It2 = IndirectlyReadable<struct Token2>;

template <class I1, class I2>
struct GoodPredicate {
    __host__ __device__ bool operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_value_t<I2>&) const;
    __host__ __device__ bool operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_reference_t<I2>) const;
    __host__ __device__ bool operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_value_t<I2>&) const;
    __host__ __device__ bool operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_reference_t<I2>) const;
    __host__ __device__ bool operator()(cuda::std::iter_common_reference_t<I1>, cuda::std::iter_common_reference_t<I2>) const;
};

// Should work when all constraints are satisfied
static_assert(cuda::std::indirect_binary_predicate<GoodPredicate<It1, It2>, It1, It2>);
static_assert(cuda::std::indirect_binary_predicate<bool(*)(int, float), int*, float*>);

#ifdef TEST_COMPILER_CLANG_CUDA
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#endif // TEST_COMPILER_CLANG_CUDA
#ifndef __CUDA_ARCH__
auto lambda = [](int i, long j) { return i == j; };
static_assert(cuda::std::indirect_binary_predicate<decltype(lambda), int*, int*>);
#endif

// Should fail when either of the iterators is not indirectly_readable
#if TEST_STD_VER > 17
struct NotIndirectlyReadable { };
static_assert(!cuda::std::indirect_binary_predicate<GoodPredicate<It1, NotIndirectlyReadable>, It1, NotIndirectlyReadable>);
static_assert(!cuda::std::indirect_binary_predicate<GoodPredicate<NotIndirectlyReadable, It2>, NotIndirectlyReadable, It2>);
#endif

// Should fail when the predicate is not copy constructible
struct BadPredicate1 {
    BadPredicate1(BadPredicate1 const&) = delete;
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate1, It1, It2>);

// Should fail when the predicate can't be called with (iter_value_t&, iter_value_t&)
struct BadPredicate2 {
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
    bool operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate2, It1, It2>);

// Should fail when the predicate can't be called with (iter_value_t&, iter_reference_t)
struct BadPredicate3 {
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
    bool operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate3, It1, It2>);

// Should fail when the predicate can't be called with (iter_reference_t, iter_value_t&)
struct BadPredicate4 {
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
    bool operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate4, It1, It2>);

// Should fail when the predicate can't be called with (iter_reference_t, iter_reference_t)
struct BadPredicate5 {
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
    bool operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate5, It1, It2>);

// Should fail when the predicate can't be called with (iter_common_reference_t, iter_common_reference_t)
struct BadPredicate6 {
    template <class T, class U>
    __host__ __device__ bool operator()(T const&, U const&) const;
    bool operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::indirect_binary_predicate<BadPredicate6, It1, It2>);

int main(int, char**)
{
  return 0;
}
