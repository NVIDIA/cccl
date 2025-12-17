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

template <class... Ts>
using box = cuda::std::__compressed_movable_box<Ts...>;

template <class T>
__host__ __device__ constexpr void test(const int expected)
{
  constexpr bool is_nothrow = cuda::std::is_nothrow_copy_constructible_v<T>;
  { // single item
    const box<T> input{1337};
    box<T> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T>> == is_nothrow);
  }
  { // two items
    const box<T, int> input{1337};
    box<T, int> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T, int>> == is_nothrow);
  }
  { // three items
    const box<T, int, int> input{1337};
    box<T, int, int> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T, int, int>> == is_nothrow);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  using cuda::std::__compressed_box_choose;
  using cuda::std::__compressed_box_copy_construct_available;
  using cuda::std::__compressed_box_specialization;
  using cuda::std::__smf_availability;

  { // trivial empty type
    using T = TrivialEmpty;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);

    const box<T> input{};
    box<T> b{input};
    assert(b.__get<0>() == 42);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T>>);
  }

  { // trivial nonempty type
    using T = int;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // non-trivial empty type
    using T = NotTriviallyCopyConstructibleEmpty<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(42);
  }

  { // non-trivial nonempty type
    using T = NotTriviallyCopyConstructible<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // non-trivial empty type, not noexcept
    using T = NotTriviallyCopyConstructibleEmpty<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(MayThrow);
  }

  { // non-trivial nonempty type, not noexcept
    using T = NotTriviallyCopyConstructible<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // not default constructible
    using T = NotDefaultConstructible;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // not copy assignable not default constructible
    using T = NotCopyAssignableNotDefaultConstructible<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // copy constructible but not copyable
    using T = CopyConstructibleWithEngaged<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // copy constructible but not copyable
    using T = CopyConstructibleWithEngaged<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // copy constructible but not copyable and not default constructible
    using T = NotDefaultConstructibleWithEngaged<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // not copy constructible
    {
      using T = NotCopyConstructible<42>;
      static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
      static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__deleted);
      static_assert(!cuda::std::is_copy_constructible_v<box<T>>);
    }

    {
      using T = NotCopyConstructibleEmpty;
      static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
      static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__deleted);
      static_assert(!cuda::std::is_copy_constructible_v<box<T>>);
    }

    {
      using T = NotCopyConstructibleEngaged<42>;
      static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
      static_assert(__compressed_box_copy_construct_available<T> == __smf_availability::__deleted);
      static_assert(!cuda::std::is_copy_constructible_v<box<T>>);
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
