//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <cuda/std/__simd_>

// [simd.iterator], host STL iterator operations for __simd_iterator

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include <iterator>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename Iterator, int N>
constexpr void test_iterator(Iterator first)
{
  using difference_t = typename Iterator::difference_type;
  auto last          = first + difference_t{N};

  // distance
  {
    auto diff = ::std::distance(first, last);
    static_assert(cuda::std::is_same_v<decltype(diff), difference_t>);
    static_assert(noexcept(::std::distance(first, last)));
    assert(diff == N);
  }
  // next
  {
    auto it = ::std::next(first, difference_t{2});
    static_assert(cuda::std::is_same_v<decltype(it), Iterator>);
    static_assert(noexcept(::std::next(first, difference_t{2})));
    assert(it == first + difference_t{2});
  }
  // prev
  {
    auto it = ::std::prev(last, difference_t{2});
    static_assert(cuda::std::is_same_v<decltype(it), Iterator>);
    static_assert(noexcept(::std::prev(last, difference_t{2})));
    assert(it == first + difference_t{N - 2});
  }
  // advance
  {
    auto it = first;
    ::std::advance(it, difference_t{2});
    static_assert(cuda::std::is_same_v<decltype(::std::advance(it, difference_t{2})), void>);
    static_assert(noexcept(::std::advance(it, difference_t{2})));
    assert(it == first + difference_t{2});
  }
}

template <typename T, int N>
constexpr void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  auto vec = make_iota_vec<T, N>();
  test_iterator<typename Vec::iterator, N>(vec.begin());

  const auto cvec = make_iota_vec<T, N>();
  test_iterator<typename Vec::const_iterator, N>(cvec.begin());

  Mask mask(true);
  test_iterator<typename Mask::iterator, N>(mask.begin());

  const Mask cmask(true);
  test_iterator<typename Mask::const_iterator, N>(cmask.begin());
}

constexpr bool test()
{
  test_type<int, 4>();
  test_type<float, 4>();
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 test();
                 static_assert(test());
               }))
  return 0;
}
