//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Arithmetic operators

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

template <class... T>
using box = cuda::std::__compressed_movable_box<T...>;

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  { // single element
    { // trivial empty type
      box<TrivialEmpty> input{};
      box<TrivialEmpty> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty>{cuda::std::move(input)}));
    }

    { // trivial nonempty type
      box<int> input{1337};
      box<int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int>{cuda::std::move(input)}));
    }

    { // non-trivial empty type
      box<NotTriviallyMoveConstructibleEmpty<42>> input{};
      box<NotTriviallyMoveConstructibleEmpty<42>> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyMoveConstructibleEmpty<42>>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyMoveConstructible<42>> input{1337};
      box<NotTriviallyMoveConstructible<42>> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyMoveConstructible<42>>{cuda::std::move(input)}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>> input{};
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>> b{cuda::std::move(input)};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyMoveConstructibleEmpty<MayThrow>>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyMoveConstructible<MayThrow>> input{1337};
      box<NotTriviallyMoveConstructible<MayThrow>> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyMoveConstructible<MayThrow>>{cuda::std::move(input)}));
    }
  }

  { // two elements
    { // trivial empty type
      box<TrivialEmpty, int> input{};
      box<TrivialEmpty, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty, int>{cuda::std::move(input)}));
    }

    { // trivial nonempty type
      box<int, int> input{1337};
      box<int, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int, int>{cuda::std::move(input)}));
    }

    { // non-trivial empty type
      box<NotTriviallyMoveConstructibleEmpty<42>, int> input{};
      box<NotTriviallyMoveConstructibleEmpty<42>, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyMoveConstructibleEmpty<42>, int>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyMoveConstructible<42>, int> input{1337};
      box<NotTriviallyMoveConstructible<42>, int> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyMoveConstructible<42>, int>{cuda::std::move(input)}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int> input{};
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyMoveConstructible<MayThrow>, int> input{1337};
      box<NotTriviallyMoveConstructible<MayThrow>, int> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyMoveConstructible<MayThrow>, int>{cuda::std::move(input)}));
    }
  }

  { // three elements
    { // trivial empty type
      box<TrivialEmpty, int, int> input{};
      box<TrivialEmpty, int, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty, int, int>{cuda::std::move(input)}));
    }

    { // trivial nonempty type
      box<int, int, int> input{1337};
      box<int, int, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int, int, int>{cuda::std::move(input)}));
    }

    { // non-trivial empty type
      box<NotTriviallyMoveConstructibleEmpty<42>, int, int> input{};
      box<NotTriviallyMoveConstructibleEmpty<42>, int, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyMoveConstructibleEmpty<42>, int, int>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyMoveConstructible<42>, int, int> input{1337};
      box<NotTriviallyMoveConstructible<42>, int, int> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyMoveConstructible<42>, int, int>{cuda::std::move(input)}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int, int> input{};
      box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int, int> b{cuda::std::move(input)};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyMoveConstructibleEmpty<MayThrow>, int, int>{cuda::std::move(input)}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyMoveConstructible<MayThrow>, int, int> input{1337};
      box<NotTriviallyMoveConstructible<MayThrow>, int, int> b{cuda::std::move(input)};
      assert(input.__get<0>() == -1);
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyMoveConstructible<MayThrow>, int, int>{cuda::std::move(input)}));
    }
  }

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
