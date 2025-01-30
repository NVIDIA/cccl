//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include <utility>

#include <nv/target>

template <class T, class U>
struct explicitly_constructible
{
  explicit constexpr explicitly_constructible(const T& value) noexcept
      : value_(value)
  {}
  constexpr bool operator==(const explicitly_constructible& other) const noexcept
  {
    return other.value_ == value_;
  }
  constexpr bool operator==(const T& other) const noexcept
  {
    return other == value_;
  }

  T value_;
};

template <class T1, class U1, class T2, class U2>
void test_construction()
{
  { // lvalue overloads
    const ::std::pair<T1, U1> matching_types{T1{42}, U1{1337}};
    const ::std::pair<T2, U1> converting_first{T2{42}, U1{1337}};
    const ::std::pair<T1, U2> converting_second{T1{42}, U2{1337}};
    const ::std::pair<T2, U2> converting_both{T2{42}, U2{1337}};

    const ::cuda::std::pair<T1, U1> from_matching{matching_types};
    const ::cuda::std::pair<T1, U1> from_converting_first{converting_first};
    const ::cuda::std::pair<T1, U1> from_converting_second{converting_second};
    const ::cuda::std::pair<T1, U1> from_converting_both{converting_both};

    assert(from_matching.first == T1{42});
    assert(from_matching.second == U1{1337});
    assert(from_converting_first.first == T1{42});
    assert(from_converting_first.second == U1{1337});
    assert(from_converting_second.first == T1{42});
    assert(from_converting_second.second == U1{1337});
    assert(from_converting_both.first == T1{42});
    assert(from_converting_both.second == U1{1337});
  }

  { // explicit lvalue overloads
    const ::std::pair<T2, U1> converting_first{T2{42}, U1{1337}};
    const ::std::pair<T1, U2> converting_second{T1{42}, U2{1337}};
    const ::std::pair<T2, U2> converting_both{T2{42}, U2{1337}};

    const ::cuda::std::pair<explicitly_constructible<T1, T2>, U1> from_converting_first{converting_first};
    const ::cuda::std::pair<T1, explicitly_constructible<U1, U2>> from_converting_second{converting_second};
    const ::cuda::std::pair<explicitly_constructible<T1, T2>, explicitly_constructible<U1, U2>> from_converting_both{
      converting_both};

    assert(from_converting_first.first == T1{42});
    assert(from_converting_first.second == U1{1337});
    assert(from_converting_second.first == T1{42});
    assert(from_converting_second.second == U1{1337});
    assert(from_converting_both.first == T1{42});
    assert(from_converting_both.second == U1{1337});
  }

  { // rvalue overloads
    const ::cuda::std::pair<T1, U1> from_matching{::std::pair<T1, U1>{T1{42}, U1{1337}}};
    const ::cuda::std::pair<T1, U1> from_converting_first{::std::pair<T2, U1>{T2{42}, U1{1337}}};
    const ::cuda::std::pair<T1, U1> from_converting_second{::std::pair<T1, U2>{T1{42}, U2{1337}}};
    const ::cuda::std::pair<T1, U1> from_converting_both{::std::pair<T2, U2>{T2{42}, U2{1337}}};

    assert(from_matching.first == T1{42});
    assert(from_matching.second == U1{1337});
    assert(from_converting_first.first == T1{42});
    assert(from_converting_first.second == U1{1337});
    assert(from_converting_second.first == T1{42});
    assert(from_converting_second.second == U1{1337});
    assert(from_converting_both.first == T1{42});
    assert(from_converting_both.second == U1{1337});
  }

  { // explicit rvalue overloads
    const ::cuda::std::pair<explicitly_constructible<T1, T2>, U1> from_converting_first{
      ::std::pair<T2, U1>{T2{42}, U1{1337}}};
    const ::cuda::std::pair<T1, explicitly_constructible<U1, U2>> from_converting_second{
      ::std::pair<T1, U2>{T1{42}, U2{1337}}};
    const ::cuda::std::pair<explicitly_constructible<T1, T2>, explicitly_constructible<U1, U2>> from_converting_both{
      ::std::pair<T2, U2>{T2{42}, U2{1337}}};

    assert(from_converting_first.first == T1{42});
    assert(from_converting_first.second == U1{1337});
    assert(from_converting_second.first == T1{42});
    assert(from_converting_second.second == U1{1337});
    assert(from_converting_both.first == T1{42});
    assert(from_converting_both.second == U1{1337});
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test_construction<int, double, short, float>();));

  return 0;
}
