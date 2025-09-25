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
      const box<TrivialEmpty> input{};
      box<TrivialEmpty> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty>{input}));
    }

    { // trivial nonempty type
      const box<int> input{1337};
      box<int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int>{input}));
    }

    { // non-trivial empty type
      box<NotTriviallyCopyConstructibleEmpty<42>> input{};
      box<NotTriviallyCopyConstructibleEmpty<42>> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyCopyConstructibleEmpty<42>>{input}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyCopyConstructible<42>> input{1337};
      box<NotTriviallyCopyConstructible<42>> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyCopyConstructible<42>>{input}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>> input{};
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>> b{input};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyCopyConstructibleEmpty<MayThrow>>{input}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyCopyConstructible<MayThrow>> input{1337};
      box<NotTriviallyCopyConstructible<MayThrow>> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyCopyConstructible<MayThrow>>{input}));
    }

    { // not copy constructible
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructible<42>>>);
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructibleEmpty>>);
    }
  }

  { // two elements
    { // trivial empty type
      const box<TrivialEmpty, int> input{};
      box<TrivialEmpty, int> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty, int>{input}));
    }

    { // trivial nonempty type
      const box<int, int> input{1337};
      box<int, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int, int>{input}));
    }

    { // non-trivial empty type
      box<NotTriviallyCopyConstructibleEmpty<42>, int> input{};
      box<NotTriviallyCopyConstructibleEmpty<42>, int> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyCopyConstructibleEmpty<42>, int>{input}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyCopyConstructible<42>, int> input{1337};
      box<NotTriviallyCopyConstructible<42>, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyCopyConstructible<42>, int>{input}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int> input{};
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int> b{input};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int>{input}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyCopyConstructible<MayThrow>, int> input{1337};
      box<NotTriviallyCopyConstructible<MayThrow>, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyCopyConstructible<MayThrow>, int>{input}));
    }

    { // not copy constructible
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructible<42>, int>>);
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructibleEmpty, int>>);
    }
  }

  { // three elements
    { // trivial empty type
      const box<TrivialEmpty, int, int> input{};
      box<TrivialEmpty, int, int> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<TrivialEmpty, int, int>{input}));
    }

    { // trivial nonempty type
      const box<int, int, int> input{1337};
      box<int, int, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<int, int, int>{input}));
    }

    { // non-trivial empty type
      box<NotTriviallyCopyConstructibleEmpty<42>, int, int> input{};
      box<NotTriviallyCopyConstructibleEmpty<42>, int, int> b{input};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyCopyConstructibleEmpty<42>, int, int>{input}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyCopyConstructible<42>, int, int> input{1337};
      box<NotTriviallyCopyConstructible<42>, int, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(noexcept(box<NotTriviallyCopyConstructible<42>, int, int>{input}));
    }

    { // non-trivial empty type, not noexcept
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int, int> input{};
      box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int, int> b{input};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyCopyConstructibleEmpty<MayThrow>, int, int>{input}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyCopyConstructible<MayThrow>, int, int> input{1337};
      box<NotTriviallyCopyConstructible<MayThrow>, int, int> b{input};
      assert(b.__get<0>() == 1337);
      static_assert(!noexcept(box<NotTriviallyCopyConstructible<MayThrow>, int, int>{input}));
    }

    { // not copy constructible
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructible<42>, int, int>>);
      static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructibleEmpty, int, int>>);
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
