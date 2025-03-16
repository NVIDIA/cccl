//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

//   template<size_t I, class T>
//     constexpr T& get(complex<T>&) noexcept;
//   template<size_t I, class T>
//     constexpr T&& get(complex<T>&&) noexcept;
//   template<size_t I, class T>
//     constexpr const T& get(const complex<T>&) noexcept;
//   template<size_t I, class T>
//     constexpr const T&& get(const complex<T>&&) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/complex>
// #include <cuda/std/vector>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T>
constexpr __host__ __device__ void test()
{
  // &
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto& r = cuda::std::get<0>(c);
    static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::get<0>(c))>);
    static_assert(noexcept(cuda::std::get<0>(c)), "");
    assert(r == T{27});
    auto& i = cuda::std::get<1>(c);
    static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::get<1>(c))>);
    static_assert(noexcept(cuda::std::get<1>(c)), "");
    assert(i == T{28});
  }
  //  &&
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto&& r = cuda::std::get<0>(cuda::std::move(c));
    static_assert(cuda::std::is_same_v<T&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>);
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    assert(r == T{27});
  }
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto&& i = cuda::std::get<1>(cuda::std::move(c));
    static_assert(cuda::std::is_same_v<T&&, decltype(cuda::std::get<1>(cuda::std::move(c)))>);
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(c))), "");
    assert(i == T{28});
  }
  // const &
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto& r = cuda::std::get<0>(c);
    static_assert(cuda::std::is_same_v<const T&, decltype(cuda::std::get<0>(c))>);
    static_assert(noexcept(cuda::std::get<0>(c)), "");
    assert(r == T{27});
    const auto& i = cuda::std::get<1>(c);
    static_assert(cuda::std::is_same_v<const T&, decltype(cuda::std::get<1>(c))>);
    static_assert(noexcept(cuda::std::get<1>(c)), "");
    assert(i == T{28});
  }
  //  const &&
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto&& r = cuda::std::get<0>(cuda::std::move(c));
    static_assert(cuda::std::is_same_v<const T&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>);
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    assert(r == T{27});
  }
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto&& i = cuda::std::get<1>(cuda::std::move(c));
    static_assert(cuda::std::is_same_v<const T&&, decltype(cuda::std::get<1>(cuda::std::move(c)))>);
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(c))), "");
    assert(i == T{28});
  }

  // `get()` allows using `complex` with structured bindings
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto [r, i]{c};
    static_assert(cuda::std::is_same_v<T, decltype(r)>);
    assert(r == T{27});
    static_assert(cuda::std::is_same_v<T, decltype(i)>);
    assert(i == T{28});
  }
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto& [r, i]{c};
    static_assert(cuda::std::is_same_v<T, decltype(r)>);
    assert(r == T{27});
    static_assert(cuda::std::is_same_v<T, decltype(i)>);
    assert(i == T{28});
  }

  // TODO: Re-enable this test when we have cuda::ranges::views
  //
  //   // `get()` allows using `complex` with ranges
  //   {
  //     cuda::std::complex<T> arr[]{{T{27}, T{28}}, {T{82}, T{94}}};

  //     auto reals = arr | cuda::std::views::elements<0>;
  //     ASSERT_SAME_AS(T, cuda::std::ranges::range_value_t<decltype(reals)>);
  //     assert(cuda::std::ranges::size(reals) == 2);
  //     assert(cuda::std::ranges::equal(reals, std::array<T, 2>{27, 82}));

  //     auto imags = arr | cuda::std::views::elements<0>;
  //     ASSERT_SAME_AS(T, cuda::std::ranges::range_value_t<decltype(imags)>);
  //     assert(cuda::std::ranges::size(imags) == 2);
  //     assert(cuda::std::ranges::equal(imags, std::array<T, 2>{28, 94}));
  //   }
  //
}

__host__ __device__ bool test()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return true;
}

constexpr __host__ __device__ bool test_constexpr()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test_constexpr(), "");

  return 0;
}
