//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// transform_view() requires cuda::std::default_initializable<V> &&
//                           cuda::std::default_initializable<F> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

constexpr int buff[] = {1, 2, 3};

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr DefaultConstructibleView() noexcept
      : begin_(buff)
      , end_(buff + 3)
  {}
  __host__ __device__ constexpr int const* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int const* end() const
  {
    return end_;
  }

private:
  int const* begin_;
  int const* end_;
};

struct DefaultConstructibleFunction
{
  int state_;
  __host__ __device__ constexpr DefaultConstructibleFunction() noexcept
      : state_(100)
  {}
  __host__ __device__ constexpr int operator()(int i) const
  {
    return i + state_;
  }
};

struct NoDefaultCtrView : cuda::std::ranges::view_base
{
  NoDefaultCtrView() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct NoDefaultFunction
{
  NoDefaultFunction() = delete;
  __host__ __device__ constexpr int operator()(int i) const
  {
    return i;
  };
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::transform_view<DefaultConstructibleView, DefaultConstructibleFunction> view{};
    assert(view.size() == 3);
    assert(view[0] == 101);
    assert(view[1] == 102);
    assert(view[2] == 103);
  }

  {
    cuda::std::ranges::transform_view<DefaultConstructibleView, DefaultConstructibleFunction> view = {};
    assert(view.size() == 3);
    assert(view[0] == 101);
    assert(view[1] == 102);
    assert(view[2] == 103);
  }

  static_assert(!cuda::std::is_default_constructible_v<
                cuda::std::ranges::transform_view<NoDefaultCtrView, DefaultConstructibleFunction>>);
  static_assert(!cuda::std::is_default_constructible_v<
                cuda::std::ranges::transform_view<DefaultConstructibleView, NoDefaultFunction>>);
  static_assert(
    !cuda::std::is_default_constructible_v<cuda::std::ranges::transform_view<NoDefaultCtrView, NoDefaultFunction>>);

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
