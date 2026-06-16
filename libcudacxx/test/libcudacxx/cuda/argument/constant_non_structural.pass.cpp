//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Non-structural types (e.g. the extended floating-point types __half and __nv_bfloat16) cannot be non-type template
// parameters. The split spellings `constant<_Value, _Tp>` and `__constant_seq<_Tp, _Values...>` make them usable by
// carrying a structural value plus a separate target type. See NVIDIA/cccl#9274.

#include <cuda/argument>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#if _LIBCUDACXX_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _LIBCUDACXX_HAS_NVBF16()

#include "test_macros.h"

enum class scaled_size : int
{
  tile = 256
};

// A user-defined type that is structural but not constexpr-constructible from an integer, mirroring how the extended
// floating-point types behave. Always available, so the mechanism is exercised even without NVFP16/NVBF16 support.
struct non_constexpr_value
{
  int __payload;

  non_constexpr_value() = default;

  // Intentionally not constexpr.
  _CCCL_HOST_DEVICE non_constexpr_value(int __v)
      : __payload(__v)
  {}

  _CCCL_HOST_DEVICE friend bool operator==(non_constexpr_value __lhs, non_constexpr_value __rhs)
  {
    return __lhs.__payload == __rhs.__payload;
  }
  _CCCL_HOST_DEVICE friend bool operator<(non_constexpr_value __lhs, non_constexpr_value __rhs)
  {
    return __lhs.__payload < __rhs.__payload;
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE void test_single_value()
{
  using C      = cuda::args::constant<128, _Tp>;
  using traits = cuda::args::__traits<C>;

  static_assert(traits::is_constant);
  static_assert(traits::is_single_value);
  static_assert(!traits::is_deferred);
  static_assert(cuda::std::is_same_v<typename traits::value_type, _Tp>);
  static_assert(cuda::std::is_same_v<typename traits::element_type, _Tp>);

  // The value and the free-function bounds are produced at run time (no constexpr ctor required).
  assert(cuda::args::__unwrap(C{}) == _Tp(128));
  assert(cuda::args::__lowest_(C{}) == _Tp(128));
  assert(cuda::args::__highest_(C{}) == _Tp(128));
}

template <class _Tp>
_CCCL_HOST_DEVICE void test_sequence()
{
  using S      = cuda::args::__constant_seq<_Tp, 4, 8, 2>;
  using traits = cuda::args::__traits<S>;

  static_assert(traits::is_constant);
  static_assert(!traits::is_single_value);
  static_assert(!traits::is_deferred);
  static_assert(cuda::std::is_same_v<typename traits::value_type, cuda::std::array<_Tp, 3>>);
  static_assert(cuda::std::is_same_v<typename traits::element_type, _Tp>);

  const auto __arr = cuda::args::__unwrap(S{});
  assert(__arr[0] == _Tp(4));
  assert(__arr[1] == _Tp(8));
  assert(__arr[2] == _Tp(2));
  assert(cuda::args::__lowest_(S{}) == _Tp(2));
  assert(cuda::args::__highest_(S{}) == _Tp(8));

  // make_constant_seq: structural source array, non-structural target element type.
  static constexpr int __src[]{5, 1, 9};
  const auto __made = cuda::args::__make_constant_seq<_Tp, __src>();
  static_assert(cuda::std::is_same_v<typename decltype(__made)::value_type, cuda::std::array<_Tp, 3>>);
  const auto __made_arr = cuda::args::__unwrap(__made);
  assert(__made_arr[0] == _Tp(5));
  assert(__made_arr[1] == _Tp(1));
  assert(__made_arr[2] == _Tp(9));
  assert(cuda::args::__lowest_(__made) == _Tp(1));
  assert(cuda::args::__highest_(__made) == _Tp(9));
}

TEST_FUNC void test()
{
  // Arithmetic element types: the split sequence is fully usable at compile time.
  {
    using S = cuda::args::__constant_seq<int, 3, 1, 2>;
    static_assert(cuda::std::is_same_v<S::value_type, cuda::std::array<int, 3>>);
    static_assert(cuda::args::__unwrap(S{}) == cuda::std::array<int, 3>{3, 1, 2});
    static_assert(cuda::args::__traits<S>::lowest == 1);
    static_assert(cuda::args::__traits<S>::highest == 3);
    static_assert(cuda::args::__lowest_(S{}) == 1);
    static_assert(cuda::args::__highest_(S{}) == 3);
  }
  {
    using S = cuda::args::__constant_seq<float, 1, 2, 3>;
    static_assert(cuda::args::__unwrap(S{}) == cuda::std::array<float, 3>{1.f, 2.f, 3.f});
    static_assert(cuda::args::__traits<S>::highest == 3.f);
  }

  // make_constant_seq with arithmetic types.
  {
    static constexpr int __src[]{10, 30, 20};
    constexpr auto __s = cuda::args::__make_constant_seq<__src>();
    static_assert(cuda::args::__unwrap(__s) == cuda::std::array<int, 3>{10, 30, 20});
    static_assert(cuda::args::__traits<decltype(__s)>::lowest == 10);
    static_assert(cuda::args::__traits<decltype(__s)>::highest == 30);

    constexpr auto __sf = cuda::args::__make_constant_seq<float, __src>();
    static_assert(cuda::args::__unwrap(__sf) == cuda::std::array<float, 3>{10.f, 30.f, 20.f});
  }

  // Regression guard: a non-arithmetic but constexpr-constructible element type (enum) keeps its static bounds. A
  // coarser `is_arithmetic`-based predicate would wrongly drop them.
  {
    using C = cuda::args::constant<scaled_size::tile>;
    static_assert(cuda::std::is_same_v<C::value_type, scaled_size>);
    static_assert(cuda::args::__unwrap(C{}) == scaled_size::tile);
    static_assert(cuda::args::__traits<C>::lowest == scaled_size::tile);
    static_assert(cuda::args::__traits<C>::highest == scaled_size::tile);

    using S = cuda::args::__constant_seq<scaled_size, 1, 256, 64>;
    static_assert(cuda::args::__traits<S>::lowest == scaled_size{1});
    static_assert(cuda::args::__traits<S>::highest == scaled_size::tile);
  }

  // Non-structural, non-constexpr-constructible element type.
  test_single_value<non_constexpr_value>();
  test_sequence<non_constexpr_value>();

#if _LIBCUDACXX_HAS_NVFP16()
  test_single_value<__half>();
  test_sequence<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  test_single_value<__nv_bfloat16>();
  test_sequence<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  test();
  return 0;
}
