//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// This test hangs forever when built against libstdc++ (Oct 2016).
// UNSUPPORTED: stdlib=libstdc++

// This test isn't quite standards-conforming: it's testing our specific
// algorithm, where when lx.try_lock() fails we start the next attempt
// with an unconditional lx.lock(). Thus our algorithm can handle a list
// of mutexes where at-most-one of them is of the evil type `class L1`,
// but will loop forever if two or more of them are `class L1`.

// <mutex>

// template <class L1, class L2, class... L3>
//   void lock(L1&, L2&, L3&...);

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

class L0
{
    bool locked_;

public:
    __host__ __device__ L0() : locked_(false) {}

    __host__ __device__ void lock()
    {
        locked_ = true;
    }

    __host__ __device__ bool try_lock()
    {
        locked_ = true;
        return locked_;
    }

    __host__ __device__ void unlock() {locked_ = false;}

    __host__ __device__ bool locked() const {return locked_;}
};

class L1
{
    bool locked_;

public:
    __host__ __device__ L1() : locked_(false) {}

    __host__ __device__ void lock()
    {
        locked_ = true;
    }

    __host__ __device__ bool try_lock()
    {
        locked_ = false;
        return locked_;
    }

    __host__ __device__ void unlock() {locked_ = false;}

    __host__ __device__ bool locked() const {return locked_;}
};

class L2
{
    bool locked_;

public:
    __host__ __device__ L2() : locked_(false) {}

    __host__ __device__ void lock()
    {
        TEST_THROW(1);
    }

    __host__ __device__ bool try_lock()
    {
        TEST_THROW(1);
        return locked_;
    }

    __host__ __device__ void unlock() {locked_ = false;}

    __host__ __device__ bool locked() const {return locked_;}
};

__host__ __device__
void with_one_or_two_locks() {
    {
        L0 l0;
        L0 l1;
        cuda::std::lock(l0, l1);
        assert(l0.locked());
        assert(l1.locked());
    }
    {
        L0 l0;
        L1 l1;
        cuda::std::lock(l0, l1);
        assert(l0.locked());
        assert(l1.locked());
    }
    {
        L1 l0;
        L0 l1;
        cuda::std::lock(l0, l1);
        assert(l0.locked());
        assert(l1.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L0 l0;
        L2 l1;
        try
        {
            cuda::std::lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
    {
        L2 l0;
        L0 l1;
        try
        {
            cuda::std::lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
    {
        L1 l0;
        L2 l1;
        try
        {
            cuda::std::lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
    {
        L2 l0;
        L1 l1;
        try
        {
            cuda::std::lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
    {
        L2 l0;
        L2 l1;
        try
        {
            cuda::std::lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
#endif
}

__host__ __device__
void with_three_locks() {
    {
        L0 l0;
        L0 l1;
        L0 l2;
        cuda::std::lock(l0, l1, l2);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L2 l0;
        L2 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
#endif
    {
        L0 l0;
        L0 l1;
        L1 l2;
        cuda::std::lock(l0, l1, l2);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
    }
    {
        L0 l0;
        L1 l1;
        L0 l2;
        cuda::std::lock(l0, l1, l2);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
    }
    {
        L1 l0;
        L0 l1;
        L0 l2;
        cuda::std::lock(l0, l1, l2);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L0 l0;
        L0 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L0 l0;
        L2 l1;
        L0 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L2 l0;
        L0 l1;
        L0 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L2 l0;
        L2 l1;
        L0 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L2 l0;
        L0 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L0 l0;
        L2 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L2 l0;
        L2 l1;
        L1 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L2 l0;
        L1 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
    {
        L1 l0;
        L2 l1;
        L2 l2;
        try
        {
            cuda::std::lock(l0, l1, l2);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
}

__host__ __device__
void with_four_locks() {
{
        L0 l0;
        L0 l1;
        L0 l2;
        L0 l3;
        cuda::std::lock(l0, l1, l2, l3);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
        assert(l3.locked());
    }
    {
        L0 l0;
        L0 l1;
        L0 l2;
        L1 l3;
        cuda::std::lock(l0, l1, l2, l3);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
        assert(l3.locked());
    }
    {
        L0 l0;
        L0 l1;
        L1 l2;
        L0 l3;
        cuda::std::lock(l0, l1, l2, l3);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
        assert(l3.locked());
    }
    {
        L0 l0;
        L1 l1;
        L0 l2;
        L0 l3;
        cuda::std::lock(l0, l1, l2, l3);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
        assert(l3.locked());
    }
    {
        L1 l0;
        L0 l1;
        L0 l2;
        L0 l3;
        cuda::std::lock(l0, l1, l2, l3);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
        assert(l3.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L0 l0;
        L0 l1;
        L0 l2;
        L2 l3;
        try
        {
            cuda::std::lock(l0, l1, l2, l3);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
            assert(!l3.locked());
        }
    }
    {
        L0 l0;
        L0 l1;
        L2 l2;
        L0 l3;
        try
        {
            cuda::std::lock(l0, l1, l2, l3);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
            assert(!l3.locked());
        }
    }
    {
        L0 l0;
        L2 l1;
        L0 l2;
        L0 l3;
        try
        {
            cuda::std::lock(l0, l1, l2, l3);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
            assert(!l3.locked());
        }
    }
    {
        L2 l0;
        L0 l1;
        L0 l2;
        L0 l3;
        try
        {
            cuda::std::lock(l0, l1, l2, l3);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
            assert(!l2.locked());
            assert(!l3.locked());
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**)
{
    with_one_or_two_locks();
    with_three_locks();
#ifndef __CUDA_ARCH__ // explodes stack space
    with_four_locks();
#endif
    return 0;
}
