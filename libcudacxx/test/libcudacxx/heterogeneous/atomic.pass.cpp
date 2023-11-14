//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include "helpers.h"

#include <cuda/std/atomic>

template<int Operand>
struct store_tester
{
    template<typename A>
    __host__ __device__
    static void initialize(A & a)
    {
        using T = decltype(a.load());
        a.store(static_cast<T>(Operand));
    }

    template<typename A>
    __host__ __device__
    static void validate(A & a)
    {
        using T = decltype(a.load());
        assert(a.load() == static_cast<T>(Operand));
    }
};

template<int PreviousValue, int Operand>
struct exchange_tester
{
    template<typename A>
    __host__ __device__
    static void initialize(A & a)
    {
        using T = decltype(a.load());
        assert(a.exchange(static_cast<T>(Operand)) == static_cast<T>(PreviousValue));
    }

    template<typename A>
    __host__ __device__
    static void validate(A & a)
    {
        using T = decltype(a.load());
        assert(a.load() == static_cast<T>(Operand));
    }
};

template<int PreviousValue, int Expected, int Desired, int Result>
struct strong_cas_tester
{
    enum { ShouldSucceed = (Expected == PreviousValue) };
    template<typename A>
    __host__ __device__
    static void initialize(A & a)
    {
        using T = decltype(a.load());
        T expected = static_cast<T>(Expected);
        assert(a.compare_exchange_strong(expected, static_cast<T>(Desired)) == ShouldSucceed);
        assert(expected == static_cast<T>(PreviousValue));
    }

    template<typename A>
    __host__ __device__
    static void validate(A & a)
    {
        using T = decltype(a.load());
        assert(a.load() == static_cast<T>(Result));
    }
};

template<int PreviousValue, int Expected, int Desired, int Result>
struct weak_cas_tester
{
    enum { ShouldSucceed = (Expected == PreviousValue) };
    template<typename A>
    __host__ __device__
    static void initialize(A & a)
    {
        using T = decltype(a.load());
        T expected = static_cast<T>(Expected);
        if (!ShouldSucceed)
        {
            assert(a.compare_exchange_weak(expected, static_cast<T>(Desired)) == false);
        }
        else
        {
            while (a.compare_exchange_weak(expected, static_cast<T>(Desired)) != ShouldSucceed) ;
        }
        assert(expected == static_cast<T>(PreviousValue));
    }

    template<typename A>
    __host__ __device__
    static void validate(A & a)
    {
        using T = decltype(a.load());
        assert(a.load() == static_cast<T>(Result));
    }
};

#define ATOMIC_TESTER(operation) \
    template<int PreviousValue, int Operand, int ExpectedValue> \
    struct operation ## _tester \
    { \
        template<typename A> \
        __host__ __device__ \
        static void initialize(A & a) \
        { \
            using T = decltype(a.load()); \
            assert(a.operation(Operand) == static_cast<T>(PreviousValue)); \
        } \
        \
        template<typename A> \
        __host__ __device__ \
        static void validate(A & a) \
        { \
            using T = decltype(a.load()); \
            assert(a.load() == static_cast<T>(ExpectedValue)); \
        } \
    };

ATOMIC_TESTER(fetch_add);
ATOMIC_TESTER(fetch_sub);

ATOMIC_TESTER(fetch_and);
ATOMIC_TESTER(fetch_or);
ATOMIC_TESTER(fetch_xor);

using basic_testers = tester_list<
    store_tester<0>,
    store_tester<-1>,
    store_tester<17>,
    exchange_tester<17, 31>,
    /* *_cas_tester<PreviousValue, Expected, Desired, Result> */
    weak_cas_tester<31, 12, 13, 31>,
    weak_cas_tester<31, 31, -6, -6>,
    strong_cas_tester<-6, -6, -12, -12>,
    strong_cas_tester<-12, 31, 17, -12>,
    exchange_tester<-12, 17>
>;

using arithmetic_atomic_testers = extend_tester_list<
    basic_testers,
    fetch_add_tester<17, 13, 30>,
    fetch_sub_tester<30, 21, 9>,
    fetch_sub_tester<9, 17, -8>
>;

using bitwise_atomic_testers = extend_tester_list<
    arithmetic_atomic_testers,
    fetch_add_tester<-8, 10, 2>,
    fetch_or_tester<2, 13, 15>,
    fetch_and_tester<15, 8, 8>,
    fetch_and_tester<8, 13, 8>,
    fetch_xor_tester<8, 12, 4>
>;

class big_not_lockfree_type
{
public:
    __host__ __device__
    big_not_lockfree_type() noexcept : big_not_lockfree_type(0)
    {
    }

    __host__ __device__
    big_not_lockfree_type(int value) noexcept
    {
        for (auto && elem : array)
        {
            elem = value++;
        }
    }

    __host__ __device__
    friend bool operator==(const big_not_lockfree_type & lhs, const big_not_lockfree_type & rhs) noexcept
    {
        for (int i = 0; i < 128; ++i)
        {
            if (lhs.array[i] != rhs.array[i])
            {
                return false;
            }
        }

        return true;
    }

private:
    int array[128];
};

__host__ __device__
void validate_not_lock_free()
{
    cuda::std::atomic<big_not_lockfree_type> test;
    assert(!test.is_lock_free());
}

void kernel_invoker()
{
    validate_not_movable<cuda::std::atomic<signed char>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed short>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed int>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed long>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed long long>, arithmetic_atomic_testers>();

    validate_not_movable<cuda::std::atomic<unsigned char>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned short>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned int>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned long>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned long long>, bitwise_atomic_testers>();

    validate_not_movable<cuda::std::atomic<float>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<double>, arithmetic_atomic_testers>();

    validate_not_movable<cuda::std::atomic<big_not_lockfree_type>, basic_testers>();

    validate_not_movable<cuda::atomic<signed char, cuda::thread_scope_system>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::atomic<signed short, cuda::thread_scope_system>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::atomic<signed int, cuda::thread_scope_system>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::atomic<signed long, cuda::thread_scope_system>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::atomic<signed long long, cuda::thread_scope_system>, arithmetic_atomic_testers>();

    validate_not_movable<cuda::atomic<unsigned char, cuda::thread_scope_system>, bitwise_atomic_testers>();
    validate_not_movable<cuda::atomic<unsigned short, cuda::thread_scope_system>, bitwise_atomic_testers>();
    validate_not_movable<cuda::atomic<unsigned int, cuda::thread_scope_system>, bitwise_atomic_testers>();
    validate_not_movable<cuda::atomic<unsigned long, cuda::thread_scope_system>, bitwise_atomic_testers>();
    validate_not_movable<cuda::atomic<unsigned long long, cuda::thread_scope_system>, bitwise_atomic_testers>();

    validate_not_movable<cuda::atomic<float>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::atomic<double>, arithmetic_atomic_testers>();

    validate_not_movable<cuda::atomic<big_not_lockfree_type, cuda::thread_scope_system>, basic_testers>();
}

int main(int arg, char ** argv)
{
    validate_not_lock_free();

    NV_IF_TARGET(NV_IS_HOST,(
        kernel_invoker();
    ))

    return 0;
}
