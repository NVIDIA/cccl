//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// ~optional();

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;

struct PODType {
  int value;
  int value2;
};

class X
{
public:
    STATIC_MEMBER_VAR(dtor_called, bool);
    X() = default;
    __host__ __device__
    ~X() {dtor_called() = true;}
};

int main(int, char**)
{
    {
        typedef int T;
        static_assert(cuda::std::is_trivially_destructible<T>::value, "");
        static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
    }
    {
        typedef double T;
        static_assert(cuda::std::is_trivially_destructible<T>::value, "");
        static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
    }
    {
        typedef PODType T;
        static_assert(cuda::std::is_trivially_destructible<T>::value, "");
        static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
    }
    {
        typedef X T;
        static_assert(!cuda::std::is_trivially_destructible<T>::value, "");
        static_assert(!cuda::std::is_trivially_destructible<optional<T>>::value, "");
        {
            X x;
            optional<X> opt{x};
            assert(X::dtor_called() == false);
        }
        assert(X::dtor_called() == true);
    }

  return 0;
}
