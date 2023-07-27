//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template <class ElementType, size_t Extent>
//     span<const byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_bytes(span<ElementType, Extent> s) noexcept;

#include <cuda/std/span>
#include <cuda/std/cassert>

#include "test_macros.h"

#ifdef TEST_COMPILER_MSVC_2017
#pragma warning(disable: 4307) // integral constant overflow
#endif // TEST_COMPILER_MSVC_2017

template <typename Span>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(cuda::std::as_bytes(sp));

    auto spBytes = cuda::std::as_bytes(sp);
    using SB = decltype(spBytes);
    ASSERT_SAME_TYPE(const cuda::std::byte, typename SB::element_type);

    if (sp.extent == cuda::std::dynamic_extent)
        assert(spBytes.extent == cuda::std::dynamic_extent);
    else
        assert(spBytes.extent == sizeof(typename Span::element_type) * sp.extent);

    assert((void *) spBytes.data() == (void *) sp.data());
#ifndef TEST_COMPILER_MSVC_2017
    assert(spBytes.size() == sp.size_bytes());
#endif // !TEST_COMPILER_MSVC_2017
}

struct A{};
__device__ int iArr2[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

int main(int, char**)
{
    testRuntimeSpan(cuda::std::span<int>        ());
    testRuntimeSpan(cuda::std::span<long>       ());
    testRuntimeSpan(cuda::std::span<double>     ());
    testRuntimeSpan(cuda::std::span<A>          ());

    testRuntimeSpan(cuda::std::span<int, 0>        ());
    testRuntimeSpan(cuda::std::span<long, 0>       ());
    testRuntimeSpan(cuda::std::span<double, 0>     ());
    testRuntimeSpan(cuda::std::span<A, 0>          ());

    testRuntimeSpan(cuda::std::span<int>(iArr2, 1));
    testRuntimeSpan(cuda::std::span<int>(iArr2, 2));
    testRuntimeSpan(cuda::std::span<int>(iArr2, 3));
    testRuntimeSpan(cuda::std::span<int>(iArr2, 4));
    testRuntimeSpan(cuda::std::span<int>(iArr2, 5));

    testRuntimeSpan(cuda::std::span<int, 1>(iArr2 + 5, 1));
    testRuntimeSpan(cuda::std::span<int, 2>(iArr2 + 4, 2));
    testRuntimeSpan(cuda::std::span<int, 3>(iArr2 + 3, 3));
    testRuntimeSpan(cuda::std::span<int, 4>(iArr2 + 2, 4));
    testRuntimeSpan(cuda::std::span<int, 5>(iArr2 + 1, 5));

  return 0;
}
