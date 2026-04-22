//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.ctor], basic_vec constructors
//
// constexpr explicit basic_vec(Up&&) noexcept;                                     // broadcast (explicit)
// constexpr basic_vec(Up&&) noexcept;                                              // broadcast (implicit)
// constexpr explicit basic_vec(const basic_vec<U,UAbi>&) noexcept;                 // converting (explicit)
// constexpr basic_vec(const basic_vec<U,UAbi>&) noexcept;                          // converting (implicit)
// constexpr explicit basic_vec(Generator&&);                                       // generator
// constexpr basic_vec(Range&&, flags<> = {});                                      // range
// constexpr basic_vec(Range&&, const mask_type&, flags<> = {});                    // masked range

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// member types and size

template <typename T, int N>
TEST_FUNC constexpr void test_member_types()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<typename Vec::value_type, T>);
  static_assert(cuda::std::is_same_v<typename Vec::abi_type, simd::fixed_size<N>>);
  static_assert(Vec::size() == N);
  static_assert(cuda::std::is_trivially_copyable_v<Vec>);
}

//----------------------------------------------------------------------------------------------------------------------
// default construction: value-initialize all elements

template <typename T, int N>
TEST_FUNC constexpr void test_default_ctor()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec{};
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == T{});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// copy construction and copy assignment

template <typename T, int N>
TEST_FUNC constexpr void test_copy()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec original(T{42});

  Vec copied(original);
  for (int i = 0; i < N; ++i)
  {
    assert(copied[i] == T{42});
  }

  Vec assigned{};
  assigned = original;
  for (int i = 0; i < N; ++i)
  {
    assert(assigned[i] == T{42});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor

template <typename T, int N>
TEST_FUNC constexpr void test_broadcast()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  static_assert(noexcept(Vec(cuda::std::declval<T>()))); // declval<T>() is needed for __half and __nv_bfloat16

  Vec vec(T{42});
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == T{42});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// generator constructor

template <typename T, int N>
TEST_FUNC constexpr void test_generator()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  const Vec vec(iota_generator<T>{});
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// converting constructor

template <typename T, typename U, int N>
TEST_FUNC constexpr void test_converting()
{
  using Src = simd::basic_vec<U, simd::fixed_size<N>>;
  using Dst = simd::basic_vec<T, simd::fixed_size<N>>;
  Src src(U{3});
  static_assert(noexcept(Dst(src)));

  Dst dst(src);
  for (int i = 0; i < N; ++i)
  {
    assert(dst[i] == static_cast<T>(U{3}));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor

template <typename T, int N>
TEST_FUNC constexpr void test_range()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  static_assert(!noexcept(Vec(arr)));
  static_assert(!noexcept(Vec(arr, simd::flag_default)));

  Vec vec(arr);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i + 1));
  }

  Vec vec2(arr, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    assert(vec2[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor with fixed-extent span

template <typename T, int N>
TEST_FUNC constexpr void test_range_span()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  const cuda::std::span<T, N> values(arr);
  const Vec vec(values);
  const Vec vec2(values, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i + 1));
    assert(vec2[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor with alignment flags

template <typename T, int N>
TEST_FUNC constexpr void test_range_alignment_flags()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  alignas(64) cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  const Vec aligned_vec(arr, simd::flag_aligned);
  const Vec overaligned_vec(arr, simd::flag_overaligned<32>);
  for (int i = 0; i < N; ++i)
  {
    assert(aligned_vec[i] == static_cast<T>(i + 1));
    assert(overaligned_vec[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// masked range constructor

template <typename T, int N>
TEST_FUNC constexpr void test_masked_range()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  Mask even_mask(is_even{});
  static_assert(!noexcept(Vec(arr, even_mask)));
  static_assert(!noexcept(Vec(arr, even_mask, simd::flag_default)));

  Vec vec(arr, even_mask);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<T>(i + 1));
    }
    else
    {
      assert(vec[i] == T{0});
    }
  }

  Vec vec2(arr, even_mask, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec2[i] == static_cast<T>(i + 1));
    }
    else
    {
      assert(vec2[i] == T{0});
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor with flag_convert
// constructs a basic_vec<T> from an array<U> with simd::flag_convert, where U is wider than T (not value-preserving)

template <typename T, typename U, int N>
TEST_FUNC constexpr void test_range_convert_lossy()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<U, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<U>(i + 1);
  }

  static_assert(!noexcept(Vec(arr, simd::flag_convert)));

  Vec vec(arr, simd::flag_convert);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(static_cast<U>(i + 1)));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// masked range constructor with flag_convert
// constructs a basic_vec<T> from an array<U> with simd::flag_convert, where U is wider than T (not value-preserving)

template <typename T, typename U, int N>
TEST_FUNC constexpr void test_masked_range_convert_lossy()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<U, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<U>(i + 1);
  }

  Mask even_mask(is_even{});
  static_assert(!noexcept(Vec(arr, even_mask, simd::flag_convert)));

  Vec vec(arr, even_mask, simd::flag_convert);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<T>(static_cast<U>(i + 1)));
    }
    else
    {
      assert(vec[i] == T{0});
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor with constexpr-wrapper-like types
// [simd.ctor] p4.3: implicit when From::value is representable by value_type

template <typename T, int N>
TEST_FUNC constexpr void test_broadcast_constexpr_wrapper()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  // integral_constant<T, V> where V fits in T: implicit
  static_assert(cuda::std::is_convertible_v<cuda::std::integral_constant<T, T{1}>, Vec>);

  // integral_constant from a wider type, but the specific value fits: implicit
  if constexpr (sizeof(T) < sizeof(int64_t) && cuda::std::is_integral_v<T>)
  {
    using IC = cuda::std::integral_constant<int64_t, 5>;
    static_assert(cuda::std::is_constructible_v<Vec, IC>);
    static_assert(cuda::std::is_convertible_v<IC, Vec>);
    Vec vec = IC{};
    for (int i = 0; i < N; ++i)
    {
      assert(vec[i] == static_cast<T>(5));
    }
  }

  // integral_constant from a wider type with a value that does NOT fit: explicit
  if constexpr (cuda::std::is_same_v<T, int8_t>)
  {
    using IC = cuda::std::integral_constant<int64_t, 200>;
    static_assert(cuda::std::is_constructible_v<Vec, IC>);
    static_assert(!cuda::std::is_convertible_v<IC, Vec>);
  }

  // unsigned value in a signed target that fits: implicit
  if constexpr (cuda::std::is_same_v<T, int>)
  {
    using IC = cuda::std::integral_constant<uint16_t, 100>;
    static_assert(cuda::std::is_convertible_v<IC, Vec>);
  }

  // negative value in an unsigned target: explicit
  if constexpr (cuda::std::is_same_v<T, uint32_t>)
  {
    using IC_neg = cuda::std::integral_constant<int64_t, -1>;
    static_assert(cuda::std::is_constructible_v<Vec, IC_neg>);
    static_assert(!cuda::std::is_convertible_v<IC_neg, Vec>);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor explicit/implicit boundary for arithmetic types
// [simd.ctor] p4: implicit iff convertible_to and value-preserving

template <typename T, int N>
TEST_FUNC constexpr void test_broadcast_explicit_implicit()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  // (1) same type is implicit
  static_assert(cuda::std::is_convertible_v<T, Vec>);

  // (2) value-preserving and wider type is implicit
  if constexpr (cuda::std::is_same_v<T, int>)
  {
    static_assert(cuda::std::is_convertible_v<int16_t, Vec>);
  }
  else if constexpr (cuda::std::is_same_v<T, double>)
  {
    static_assert(cuda::std::is_convertible_v<float, Vec>);
  }

  // (3) narrow conversion is explicit
  else if constexpr (cuda::std::is_same_v<T, int16_t>)
  {
    static_assert(cuda::std::is_constructible_v<Vec, int>);
    static_assert(!cuda::std::is_convertible_v<int, Vec>);
  }
  else if constexpr (cuda::std::is_same_v<T, float>)
  {
    static_assert(cuda::std::is_constructible_v<Vec, double>);
    static_assert(!cuda::std::is_convertible_v<double, Vec>);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// SFINAE constraints

template <typename T, int N>
TEST_FUNC constexpr void test_sfinae()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  static_assert(cuda::std::is_constructible_v<Vec, T>);

  using VecDifferentSize = simd::basic_vec<T, simd::fixed_size<N + 1>>;
  static_assert(!cuda::std::is_constructible_v<Vec, const VecDifferentSize&>);

  static_assert(!cuda::std::is_constructible_v<Vec, wrong_generator>);
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_member_types<T, N>();
  test_default_ctor<T, N>();
  test_copy<T, N>();
  test_broadcast<T, N>();
  test_broadcast_explicit_implicit<T, N>();
  test_generator<T, N>();
  test_range<T, N>();
  test_range_span<T, N>();
  test_range_alignment_flags<T, N>();
  test_masked_range<T, N>();
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_broadcast_constexpr_wrapper<T, N>();
  }
  test_sfinae<T, N>();
  if constexpr (sizeof(T) >= 2 && cuda::std::is_integral_v<T>)
  {
    using Smaller = cuda::std::conditional_t<cuda::std::is_signed_v<T>, int8_t, uint8_t>;
    test_converting<T, Smaller, N>();
  }
  if constexpr (sizeof(T) < 8 && cuda::std::is_integral_v<T>)
  {
    using Wider = cuda::std::conditional_t<cuda::std::is_signed_v<T>, int64_t, uint64_t>;
    test_range_convert_lossy<T, Wider, N>();
    test_masked_range_convert_lossy<T, Wider, N>();
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    test_range_convert_lossy<T, double, N>();
    test_masked_range_convert_lossy<T, double, N>();
  }
}

//----------------------------------------------------------------------------------------------------------------------
// enable/disable boundary: basic_vec<T, fixed_size<N>> is enabled iff T is vectorizable and N in [1, 64]

TEST_FUNC constexpr void test_enable_abi_boundary()
{
  using T = int;

  // enabled at the range boundaries
  static_assert(cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<1>>>);
  static_assert(cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<64>>>);

  // disabled outside the [1, 64] range
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<0>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<65>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<100>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_vec<T, simd::fixed_size<-1>>>);

  // the disabled specialization has all special members deleted
  using DisabledVec = simd::basic_vec<T, simd::fixed_size<65>>;
  static_assert(!cuda::std::is_default_constructible_v<DisabledVec>);
  static_assert(!cuda::std::is_copy_constructible_v<DisabledVec>);
  static_assert(!cuda::std::is_copy_assignable_v<DisabledVec>);
  static_assert(!cuda::std::is_destructible_v<DisabledVec>);

  // the disabled specialization still exposes value_type / abi_type / mask_type
  static_assert(cuda::std::is_same_v<DisabledVec::value_type, T>);
  static_assert(cuda::std::is_same_v<DisabledVec::abi_type, simd::fixed_size<65>>);
  static_assert(cuda::std::is_same_v<DisabledVec::mask_type, simd::basic_mask<sizeof(T), simd::fixed_size<65>>>);
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  test_enable_abi_boundary();
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
