//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <utility>

// class make_integer_sequence

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename AtContainer, typename T, T... I>
__host__ __device__ auto extract(const AtContainer& t, const cuda::std::integer_sequence<T, I...>)
  -> decltype(cuda::std::make_tuple(cuda::std::get<I>(t)...))
{
  return cuda::std::make_tuple(cuda::std::get<I>(t)...);
}

int main(int, char**)
{
  //  Make a couple of sequences
  using intseq3 = cuda::std::make_integer_sequence<int, 3>; // generates int:    0,1,2
  using size7   = cuda::std::make_integer_sequence<size_t, 7>; // generates size_t: 0,1,2,3,4,5,6
  using size4   = cuda::std::make_index_sequence<4>; // generates size_t: 0,1,2,3
  using size2   = cuda::std::index_sequence_for<int, size_t>; // generates size_t: 0,1
  using intmix  = cuda::std::integer_sequence<int, 9, 8, 7, 2>; // generates int:    9,8,7,2
  using sizemix = cuda::std::index_sequence<1, 1, 2, 3, 5>; // generates size_t: 1,1,2,3,5

  //  Make sure they're what we expect
  static_assert(cuda::std::is_same<intseq3::value_type, int>::value, "intseq3 type wrong");
  static_assert(intseq3::size() == 3, "intseq3 size wrong");

  static_assert(cuda::std::is_same<size7::value_type, size_t>::value, "size7 type wrong");
  static_assert(size7::size() == 7, "size7 size wrong");

  static_assert(cuda::std::is_same<size4::value_type, size_t>::value, "size4 type wrong");
  static_assert(size4::size() == 4, "size4 size wrong");

  static_assert(cuda::std::is_same<size2::value_type, size_t>::value, "size2 type wrong");
  static_assert(size2::size() == 2, "size2 size wrong");

  static_assert(cuda::std::is_same<intmix::value_type, int>::value, "intmix type wrong");
  static_assert(intmix::size() == 4, "intmix size wrong");

  static_assert(cuda::std::is_same<sizemix::value_type, size_t>::value, "sizemix type wrong");
  static_assert(sizemix::size() == 5, "sizemix size wrong");

  auto tup = cuda::std::make_tuple(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);

  //  Use them
  auto t3 = extract(tup, intseq3());
  static_assert(cuda::std::tuple_size<decltype(t3)>::value == intseq3::size(), "t3 size wrong");
  assert(t3 == cuda::std::make_tuple(10, 11, 12));

  auto t7 = extract(tup, size7());
  static_assert(cuda::std::tuple_size<decltype(t7)>::value == size7::size(), "t7 size wrong");
  assert(t7 == cuda::std::make_tuple(10, 11, 12, 13, 14, 15, 16));

  auto t4 = extract(tup, size4());
  static_assert(cuda::std::tuple_size<decltype(t4)>::value == size4::size(), "t4 size wrong");
  assert(t4 == cuda::std::make_tuple(10, 11, 12, 13));

  auto t2 = extract(tup, size2());
  static_assert(cuda::std::tuple_size<decltype(t2)>::value == size2::size(), "t2 size wrong");
  assert(t2 == cuda::std::make_tuple(10, 11));

  auto tintmix = extract(tup, intmix());
  static_assert(cuda::std::tuple_size<decltype(tintmix)>::value == intmix::size(), "tintmix size wrong");
  assert(tintmix == cuda::std::make_tuple(19, 18, 17, 12));

  auto tsizemix = extract(tup, sizemix());
  static_assert(cuda::std::tuple_size<decltype(tsizemix)>::value == sizemix::size(), "tsizemix size wrong");
  assert(tsizemix == cuda::std::make_tuple(11, 11, 12, 13, 15));

  return 0;
}
