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
TEST_CONSTEXPR_CXX14 __host__ __device__ void test()
{
  // &
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto& r = cuda::std::get<0>(c);
    ASSERT_SAME_TYPE(T&, decltype(cuda::std::get<0>(c)));
    static_assert(noexcept(cuda::std::get<0>(c)), "");
    assert(r == T{27});
    auto& i = cuda::std::get<1>(c);
    ASSERT_SAME_TYPE(T&, decltype(cuda::std::get<1>(c)));
    static_assert(noexcept(cuda::std::get<1>(c)), "");
    assert(i == T{28});
  }
  //  &&
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto&& r = cuda::std::get<0>(cuda::std::move(c));
    ASSERT_SAME_TYPE(T&&, decltype(cuda::std::get<0>(cuda::std::move(c))));
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    assert(r == T{27});
  }
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto&& i = cuda::std::get<1>(cuda::std::move(c));
    ASSERT_SAME_TYPE(T&&, decltype(cuda::std::get<1>(cuda::std::move(c))));
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(c))), "");
    assert(i == T{28});
  }
  // const &
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto& r = cuda::std::get<0>(c);
    ASSERT_SAME_TYPE(const T&, decltype(cuda::std::get<0>(c)));
    static_assert(noexcept(cuda::std::get<0>(c)), "");
    assert(r == T{27});
    const auto& i = cuda::std::get<1>(c);
    ASSERT_SAME_TYPE(const T&, decltype(cuda::std::get<1>(c)));
    static_assert(noexcept(cuda::std::get<1>(c)), "");
    assert(i == T{28});
  }
  //  const &&
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto&& r = cuda::std::get<0>(cuda::std::move(c));
    ASSERT_SAME_TYPE(const T&&, decltype(cuda::std::get<0>(cuda::std::move(c))));
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    assert(r == T{27});
  }
  {
    const cuda::std::complex<T> c{T{27}, T{28}};

    const auto&& i = cuda::std::get<1>(cuda::std::move(c));
    ASSERT_SAME_TYPE(const T&&, decltype(cuda::std::get<1>(cuda::std::move(c))));
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(c))), "");
    assert(i == T{28});
  }

#if TEST_STD_VER >= 2017
  // `get()` allows using `complex` with structured bindings
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto [r, i]{c};
    ASSERT_SAME_TYPE(T, decltype(r));
    assert(r == T{27});
    ASSERT_SAME_TYPE(T, decltype(i));
    assert(i == T{28});
  }
  {
    cuda::std::complex<T> c{T{27}, T{28}};

    auto& [r, i]{c};
    ASSERT_SAME_TYPE(T, decltype(r));
    assert(r == T{27});
    ASSERT_SAME_TYPE(T, decltype(i));
    assert(i == T{28});
  }
#endif // TEST_STD_VER >= 2017

  // `get()` allows using `complex` with ranges
  // {
  //   cuda::std::complex<T> arr[]{{T{27}, T{28}}, {T{82}, T{94}}};

  //   std::same_as<cuda::std::vector<T>> decltype(auto) reals{
  //       arr | std::views::elements<0> | cuda::std::ranges::to<cuda::std::vector<T>>()};
  //   assert(reals.size() == 2);
  //   assert(reals[0] == T{27});
  //   assert(reals[1] == T{82});

  //   std::same_as<std::vector<T>> decltype(auto) imags{
  //       arr | std::views::elements<1> | cuda::std::ranges::to<cuda::std::vector<T>>()};
  //   assert(imags.size() == 2);
  //   assert(imags[0] == T{28});
  //   assert(imags[1] == T{94});
  // }
}

__host__ __device__ bool test()
{
  test<float>();
  test<double>();

  // CUDA treats long double as double
  // test<long double>();

#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif

  return true;
}

TEST_CONSTEXPR_CXX14 __host__ __device__ bool test_constexpr()
{
  test<float>();
  test<double>();

  // CUDA treats long double as double
  // test<long double>();

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2014
  static_assert(test_constexpr(), "");
#endif

  return 0;
}
