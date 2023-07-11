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

// <mutex>

// template <class L1, class L2, class... L3>
//   int try_lock(L1&, L2&, L3&...);

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

class L0
{
    bool locked_;

public:
    __host__ __device__ L0() : locked_(false) {}

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

    __host__ __device__ bool try_lock()
    {
        TEST_THROW(1);
        return locked_;
    }

    __host__ __device__ void unlock() {locked_ = false;}

    __host__ __device__ bool locked() const {return locked_;}
};

int main(int, char**)
{
    {
        L0 l0;
        L0 l1;
        assert(cuda::std::try_lock(l0, l1) == -1);
        assert(l0.locked());
        assert(l1.locked());
    }
    {
        L0 l0;
        L1 l1;
        assert(cuda::std::try_lock(l0, l1) == 1);
        assert(!l0.locked());
        assert(!l1.locked());
    }
    {
        L1 l0;
        L0 l1;
        assert(cuda::std::try_lock(l0, l1) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L0 l0;
        L2 l1;
        try
        {
            (void)cuda::std::try_lock(l0, l1);
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
            (void)cuda::std::try_lock(l0, l1);
            assert(false);
        }
        catch (int)
        {
            assert(!l0.locked());
            assert(!l1.locked());
        }
    }
#endif
#if TEST_STD_VER >= 11
    {
        L0 l0;
        L0 l1;
        L0 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == -1);
        assert(l0.locked());
        assert(l1.locked());
        assert(l2.locked());
    }
    {
        L1 l0;
        L1 l1;
        L1 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L2 l0;
        L2 l1;
        L2 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
        L1 l1;
        L2 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 1);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
#endif
    {
        L0 l0;
        L0 l1;
        L1 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 2);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L0 l0;
        L1 l1;
        L0 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 1);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L1 l0;
        L0 l1;
        L0 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L0 l0;
        L0 l1;
        L2 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
        L1 l0;
        L1 l1;
        L0 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L1 l0;
        L0 l1;
        L1 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L0 l0;
        L1 l1;
        L1 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 1);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        L1 l0;
        L1 l1;
        L2 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L1 l0;
        L2 l1;
        L1 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L2 l0;
        L1 l1;
        L1 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
            (void)cuda::std::try_lock(l0, l1, l2);
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
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L0 l0;
        L2 l1;
        L1 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
        L0 l1;
        L2 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L1 l0;
        L2 l1;
        L0 l2;
        assert(cuda::std::try_lock(l0, l1, l2) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
    }
    {
        L2 l0;
        L0 l1;
        L1 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
        L0 l2;
        try
        {
            (void)cuda::std::try_lock(l0, l1, l2);
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
    {
        L0 l0;
        L0 l1;
        L0 l2;
        L0 l3;
        assert(cuda::std::try_lock(l0, l1, l2, l3) == -1);
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
        assert(cuda::std::try_lock(l0, l1, l2, l3) == 0);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
        assert(!l3.locked());
    }
    {
        L0 l0;
        L1 l1;
        L0 l2;
        L0 l3;
        assert(cuda::std::try_lock(l0, l1, l2, l3) == 1);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
        assert(!l3.locked());
    }
    {
        L0 l0;
        L0 l1;
        L1 l2;
        L0 l3;
        assert(cuda::std::try_lock(l0, l1, l2, l3) == 2);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
        assert(!l3.locked());
    }
    {
        L0 l0;
        L0 l1;
        L0 l2;
        L1 l3;
        assert(cuda::std::try_lock(l0, l1, l2, l3) == 3);
        assert(!l0.locked());
        assert(!l1.locked());
        assert(!l2.locked());
        assert(!l3.locked());
    }
#endif // TEST_STD_VER >= 11

  return 0;
}
