//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <cuda/std/optional>

// constexpr optional(T&& v);

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "archetypes.h"


using cuda::std::optional;


class Z
{
public:
    __host__ __device__
    Z(int) {}
    __host__ __device__
    Z(Z&&) {TEST_THROW(6);}
};


int main(int, char**)
{
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    {
        typedef int T;
        constexpr optional<T> opt(T(5));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 5, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            __host__ __device__
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef double T;
        constexpr optional<T> opt(T(3));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            __host__ __device__
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    {
        const int x = 42;
        optional<const int> o(cuda::std::move(x));
        assert(*o == 42);
    }
    {
        typedef TestTypes::TestType T;
        T::reset();
        optional<T> opt = T{3};
        assert(T::alive() == 1);
        assert(T::move_constructed() == 1);
        assert(static_cast<bool>(opt) == true);
        assert(opt.value().value == 3);
    }
    {
        typedef ExplicitTestTypes::TestType T;
        static_assert(!cuda::std::is_convertible<T&&, optional<T>>::value, "");
        T::reset();
        optional<T> opt(T{3});
        assert(T::alive() == 1);
        assert(T::move_constructed() == 1);
        assert(static_cast<bool>(opt) == true);
        assert(opt.value().value == 3);
    }
    {
        typedef TestTypes::TestType T;
        T::reset();
        optional<T> opt = {3};
        assert(T::alive() == 1);
        assert(T::value_constructed() == 1);
        assert(T::copy_constructed() == 0);
        assert(T::move_constructed() == 0);
        assert(static_cast<bool>(opt) == true);
        assert(opt.value().value == 3);
    }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    {
        typedef ConstexprTestTypes::TestType T;
        constexpr optional<T> opt = {T(3)};
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(opt.value().value == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            __host__ __device__
            constexpr test_constexpr_ctor(const T&) {}
        };
    }
    {
        typedef ConstexprTestTypes::TestType T;
        constexpr optional<T> opt = {3};
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(opt.value().value == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            __host__ __device__
            constexpr test_constexpr_ctor(const T&) {}
        };
    }
    {
        typedef ExplicitConstexprTestTypes::TestType T;
        static_assert(!cuda::std::is_convertible<T&&, optional<T>>::value, "");
        constexpr optional<T> opt(T{3});
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(opt.value().value == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            __host__ __device__
            constexpr test_constexpr_ctor(T&&) {}
        };

    }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        try
        {
            Z z(3);
            optional<Z> opt(cuda::std::move(z));
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
#endif

  return 0;
}
