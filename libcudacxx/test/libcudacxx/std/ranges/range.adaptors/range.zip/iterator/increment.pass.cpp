//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires all_forward<Const, Views...>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include "../types.h"

struct InputRange : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr InputRange(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif
  using iterator = cpp20_input_iterator<int*>;
  __host__ __device__ constexpr iterator begin() const
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr sentinel_wrapper<iterator> end() const
  {
    return sentinel_wrapper<iterator>(iterator(buffer_ + size_));
  }
};

__host__ __device__ constexpr bool test()
{
  cuda::std::array<int, 4> a{1, 2, 3, 4};
  cuda::std::array<double, 3> b{4.1, 3.2, 4.3};
  {
    // random/contiguous
    cuda::std::ranges::zip_view v(cuda::std::span<int>{a}, cuda::std::span<double>{b}, cuda::std::views::iota(0, 5));
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
    assert(&(cuda::std::get<1>(*it)) == &(b[0]));
    assert(cuda::std::get<2>(*it) == 0);

    static_assert(cuda::std::is_same_v<decltype(++it), Iter&>);

    auto& it_ref = ++it;
    assert(&it_ref == &it);

    assert(&(cuda::std::get<0>(*it)) == &(a[1]));
    assert(&(cuda::std::get<1>(*it)) == &(b[1]));
    assert(cuda::std::get<2>(*it) == 1);

    static_assert(cuda::std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy     = it++;
    assert(original == copy);
    assert(&(cuda::std::get<0>(*it)) == &(a[2]));
    assert(&(cuda::std::get<1>(*it)) == &(b[2]));
    assert(cuda::std::get<2>(*it) == 2);
  }

  {
    //  bidi
    int buffer[2] = {1, 2};

    cuda::std::ranges::zip_view v(BidiCommonView{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(&(cuda::std::get<0>(*it)) == &(buffer[0]));

    static_assert(cuda::std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(cuda::std::get<0>(*it)) == &(buffer[1]));

    static_assert(cuda::std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy     = it++;
    assert(copy == original);
    assert(&(cuda::std::get<0>(*it)) == &(buffer[2]));
  }

  {
    //  forward
    int buffer[2] = {1, 2};

    cuda::std::ranges::zip_view v(ForwardSizedView{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(&(cuda::std::get<0>(*it)) == &(buffer[0]));

    static_assert(cuda::std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(cuda::std::get<0>(*it)) == &(buffer[1]));

    static_assert(cuda::std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy     = it++;
    assert(copy == original);
    assert(&(cuda::std::get<0>(*it)) == &(buffer[2]));
  }

  {
    // all input+
    int buffer[3] = {4, 5, 6};
    cuda::std::ranges::zip_view v(cuda::std::span<int>{a}, InputRange{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
    assert(&(cuda::std::get<1>(*it)) == &(buffer[0]));

    static_assert(cuda::std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(cuda::std::get<0>(*it)) == &(a[1]));
    assert(&(cuda::std::get<1>(*it)) == &(buffer[1]));

    static_assert(cuda::std::is_same_v<decltype(it++), void>);
    it++;
    assert(&(cuda::std::get<0>(*it)) == &(a[2]));
    assert(&(cuda::std::get<1>(*it)) == &(buffer[2]));
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
