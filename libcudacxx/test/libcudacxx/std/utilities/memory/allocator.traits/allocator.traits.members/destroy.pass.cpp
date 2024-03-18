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
//     template <class Ptr>
//     static constexpr void destroy(allocator_type& a, Ptr p);
//     ...
// };

// Currently no suppport for std::allocator
// XFAIL: true

#include <cuda/std/__memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct NoDestroy
{
    typedef T value_type;

    __host__ __device__ TEST_CONSTEXPR_CXX20 T* allocate(cuda::std::size_t n)
    {
        return cuda::std::allocator<T>().allocate(n);
    }

    __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(T* p, cuda::std::size_t n)
    {
        return cuda::std::allocator<T>().deallocate(p, n);
    }
};

template <class T>
struct CountDestroy
{
    __host__ __device__ TEST_CONSTEXPR explicit CountDestroy(int* counter)
        : counter_(counter)
    { }

    typedef T value_type;

    __host__ __device__ TEST_CONSTEXPR_CXX20 T* allocate(cuda::std::size_t n)
    {
        return cuda::std::allocator<T>().allocate(n);
    }

    __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(T* p, cuda::std::size_t n)
    {
        return cuda::std::allocator<T>().deallocate(p, n);
    }

    template <class U>
    __host__ __device__ TEST_CONSTEXPR_CXX20 void destroy(U* p)
    {
        ++*counter_;
        p->~U();
    }

    int* counter_;
};

struct CountDestructor
{
    __host__ __device__ TEST_CONSTEXPR explicit CountDestructor(int* counter)
        : counter_(counter)
    { }

    __host__ __device__ TEST_CONSTEXPR_CXX20 ~CountDestructor() { ++*counter_; }

    int* counter_;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
    {
        typedef NoDestroy<CountDestructor> Alloc;
        int destructors = 0;
        Alloc alloc;
        CountDestructor* pool = cuda::std::allocator_traits<Alloc>::allocate(alloc, 1);

        cuda::std::allocator_traits<Alloc>::construct(alloc, pool, &destructors);
        assert(destructors == 0);

        cuda::std::allocator_traits<Alloc>::destroy(alloc, pool);
        assert(destructors == 1);

        cuda::std::allocator_traits<Alloc>::deallocate(alloc, pool, 1);
    }
    {
        typedef IncompleteHolder* T;
        typedef NoDestroy<T> Alloc;
        Alloc alloc;
        T* pool = cuda::std::allocator_traits<Alloc>::allocate(alloc, 1);
        cuda::std::allocator_traits<Alloc>::construct(alloc, pool, nullptr);
        cuda::std::allocator_traits<Alloc>::destroy(alloc, pool);
        cuda::std::allocator_traits<Alloc>::deallocate(alloc, pool, 1);
    }
    {
        typedef CountDestroy<CountDestructor> Alloc;
        int destroys_called = 0;
        int destructors_called = 0;
        Alloc alloc(&destroys_called);

        CountDestructor* pool = cuda::std::allocator_traits<Alloc>::allocate(alloc, 1);
        cuda::std::allocator_traits<Alloc>::construct(alloc, pool, &destructors_called);
        assert(destroys_called == 0);
        assert(destructors_called == 0);

        cuda::std::allocator_traits<Alloc>::destroy(alloc, pool);
        assert(destroys_called == 1);
        assert(destructors_called == 1);

        cuda::std::allocator_traits<Alloc>::deallocate(alloc, pool, 1);
    }
    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 2020  \
 && !defined(TEST_COMPILER_NVCC) \
 && !defined(TEST_COMPILER_NVRTC)
    static_assert(test());
#endif // TEST_STD_VER >= 2020
    return 0;
}
