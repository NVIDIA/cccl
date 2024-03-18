//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     static constexpr size_type max_size(const allocator_type& a) noexcept;
//     ...
// };

#include <cuda/std/limits>
#include <cuda/std/__memory>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct A
{
    typedef T value_type;

};

template <class T>
struct B
{
    typedef T value_type;

    __host__ __device__ TEST_CONSTEXPR_CXX20 cuda::std::size_t max_size() const
    {
        return 100;
    }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
    {
        B<int> b;
        assert(cuda::std::allocator_traits<B<int> >::max_size(b) == 100);
    }
    {
        const B<int> b = {};
        assert(cuda::std::allocator_traits<B<int> >::max_size(b) == 100);
    }
    {
        typedef IncompleteHolder* VT;
        typedef B<VT> Alloc;
        Alloc a;
        assert(cuda::std::allocator_traits<Alloc >::max_size(a) == 100);
    }
#if TEST_STD_VER >= 11
    {
        A<int> a;
        assert(cuda::std::allocator_traits<A<int> >::max_size(a) ==
               cuda::std::numeric_limits<cuda::std::size_t>::max() / sizeof(int));
    }
    {
        const A<int> a = {};
        assert(cuda::std::allocator_traits<A<int> >::max_size(a) ==
               cuda::std::numeric_limits<cuda::std::size_t>::max() / sizeof(int));
    }
#ifdef _LIBCUDACXX_HAS_ALLOCATOR
    {
        cuda::std::allocator<int> a;
        static_assert(noexcept(cuda::std::allocator_traits<cuda::std::allocator<int>>::max_size(a)) == true, "");
    }
#endif // 0
#endif // TEST_STD_VER >= 11

    return true;
}

int main(int, char**)
{
    test();

#if TEST_STD_VER >= 2020
    static_assert(test());
#endif // TEST_STD_VER >= 2020

    return 0;
}
