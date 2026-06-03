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

// [simd.permute.mask]
//
// template<simd-vec-type V>
// constexpr V compress(const V& v, const typename V::mask_type& selector);
// template<simd-mask-type V>
// constexpr V compress(const V& v, const type_identity_t<V>& selector);
//
// template<simd-vec-type V>
// constexpr V compress(const V& v, const typename V::mask_type& selector,
//                      const typename V::value_type& fill_value);
// template<simd-mask-type V>
// constexpr V compress(const V& v, const type_identity_t<V>& selector,
//                      const typename V::value_type& fill_value);
//
// template<simd-vec-type V>
// constexpr V expand(const V& v, const typename V::mask_type& selector, const V& original = {});
// template<simd-mask-type V>
// constexpr V expand(const V& v, const type_identity_t<V>& selector, const V& original = {});

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// compress: basic_vec

template <typename T, int N>
TEST_FUNC constexpr void test_compress_vec()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec src(iota_generator<T>{});

  // all-true selector: output == source
  {
    Mask all_true(true);
    Vec compress = simd::compress(src, all_true);
    static_assert(cuda::std::is_same_v<decltype(compress), Vec>);
    for (int i = 0; i < N; ++i)
    {
      assert(compress[i] == src[i]);
    }
  }
  // all-false selector with fill_value: every component == fill
  {
    Mask all_false(false);
    T fill       = static_cast<T>(42);
    Vec compress = simd::compress(src, all_false, fill);
    for (int i = 0; i < N; ++i)
    {
      assert(compress[i] == fill);
    }
  }
  // alternating pattern: components 0, 2, 4, ... are true.
  // Selected elements compact to the front; trailing lanes take the fill value.
  if constexpr (N >= 2)
  {
    Mask sel(is_even{});
    T fill             = static_cast<T>(99);
    Vec compress       = simd::compress(src, sel, fill);
    constexpr int half = (N + 1) / 2;
    for (int i = 0; i < N; ++i)
    {
      auto expected = i < half ? src[2 * i] : fill;
      assert(compress[i] == expected);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// expand: basic_vec

template <typename T, int N>
TEST_FUNC constexpr void test_expand_vec()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec src(iota_generator<T>{});
  Vec original(iota_generator<T, 100>{});

  // all-true selector: output == src (src[0..N-1] populates every component in order)
  {
    Mask all_true(true);
    Vec expand = simd::expand(src, all_true, original);
    static_assert(cuda::std::is_same_v<decltype(expand), Vec>);
    for (int i = 0; i < N; ++i)
    {
      assert(expand[i] == src[i]);
    }
  }
  // all-false selector: output == original
  {
    Mask all_false(false);
    Vec expand = simd::expand(src, all_false, original);
    for (int i = 0; i < N; ++i)
    {
      assert(expand[i] == original[i]);
    }
  }
  // default original = {}: all false-selector components become T{}
  {
    Mask all_false(false);
    Vec r = simd::expand(src, all_false);
    for (int i = 0; i < N; ++i)
    {
      assert(r[i] == T{});
    }
  }
  // alternating: components 0, 2, 4, ... true.
  // True components pull from the front of src, false components from original.
  if constexpr (N >= 2)
  {
    Mask even_mask(is_even{});
    Vec expand = simd::expand(src, even_mask, original);
    int count  = 0;
    for (int i = 0; i < N; ++i)
    {
      if (i % 2 == 0)
      {
        assert(expand[i] == src[count]);
        ++count;
      }
      else
      {
        assert(expand[i] == original[i]);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// compress: basic_mask

template <int Bytes, int N>
TEST_FUNC constexpr void test_compress_mask()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask src(is_even{});

  // all-true selector: output == source
  {
    Mask all_true(true);
    Mask compress = simd::compress(src, all_true);
    static_assert(cuda::std::is_same_v<decltype(compress), Mask>);
    for (int i = 0; i < N; ++i)
    {
      assert(compress[i] == src[i]);
    }
  }
  // all-false selector with fill=true: every component == true
  {
    Mask all_false(false);
    Mask r = simd::compress(src, all_false, true);
    for (int i = 0; i < N; ++i)
    {
      assert(r[i] == true);
    }
  }
  // alternating pattern: selector == src == [T, F, T, F, ...].
  // Selected src values at positions 0, 2, 4, ... are all true and compact to the front;
  // trailing lanes take the fill value (false).
  if constexpr (N >= 2)
  {
    Mask even_mask(is_even{});
    Mask compress      = simd::compress(src, even_mask, false);
    constexpr int half = (N + 1) / 2;
    for (int i = 0; i < N; ++i)
    {
      assert(compress[i] == (i < half));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// expand: basic_mask

template <int Bytes, int N>
TEST_FUNC constexpr void test_expand_mask()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  Mask src(is_even{});
  Mask original(true);

  // all-true selector: output == src
  {
    Mask all_true(true);
    Mask expand = simd::expand(src, all_true, original);
    for (int i = 0; i < N; ++i)
    {
      assert(expand[i] == src[i]);
    }
  }

  // all-false selector: output == original
  {
    Mask all_false(false);
    Mask expand = simd::expand(src, all_false, original);
    for (int i = 0; i < N; ++i)
    {
      assert(expand[i] == original[i]);
    }
  }

  // default original = {}: all components default (false)
  {
    Mask all_false(false);
    Mask expand = simd::expand(src, all_false);
    for (int i = 0; i < N; ++i)
    {
      assert(expand[i] == false);
    }
  }

  // alternating: sel = [T, F, T, F]; src = [T, F, T, F]
  // True components pull from the front of src, false components from original.
  if constexpr (N >= 2)
  {
    Mask even_mask(is_even{});
    Mask expand = simd::expand(src, even_mask, original);
    int count   = 0;
    for (int i = 0; i < N; ++i)
    {
      if (i % 2 == 0)
      {
        assert(expand[i] == src[count]);
        ++count;
      }
      else
      {
        assert(expand[i] == original[i]);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept

TEST_FUNC constexpr void test_noexcept()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = simd::basic_mask<4, simd::fixed_size<4>>;
  Vec v{};
  Mask m{};
  int fill{};
  bool bfill{};
  unused(v, m, fill, bfill);

  static_assert(!noexcept(simd::compress(v, m)));
  static_assert(!noexcept(simd::compress(v, m, fill)));
  static_assert(!noexcept(simd::expand(v, m, v)));

  static_assert(!noexcept(simd::compress(m, m)));
  static_assert(!noexcept(simd::compress(m, m, bfill)));
  static_assert(!noexcept(simd::expand(m, m, m)));
}

//----------------------------------------------------------------------------------------------------------------------
// Return type

TEST_FUNC constexpr void test_return_type()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = simd::basic_mask<4, simd::fixed_size<4>>;
  Vec v{};
  Mask m{};
  int fill{};
  unused(v, m, fill);

  static_assert(cuda::std::is_same_v<decltype(simd::compress(v, m)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::compress(v, m, fill)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::expand(v, m, v)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(simd::compress(m, m)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(simd::expand(m, m)), Mask>);
}

//----------------------------------------------------------------------------------------------------------------------

// do not depend on element types
TEST_FUNC constexpr bool test_fixed_type()
{
  test_noexcept();
  test_return_type();
  test_compress_mask<1, 1>();
  test_compress_mask<1, 4>();
  test_compress_mask<4, 1>();
  test_compress_mask<4, 4>();
  test_compress_mask<8, 4>();
  test_expand_mask<1, 1>();
  test_expand_mask<1, 4>();
  test_expand_mask<4, 1>();
  test_expand_mask<4, 4>();
  test_expand_mask<8, 4>();
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Test drivers

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_compress_vec<T, N>();
  test_expand_vec<T, N>();
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_fixed_type());
  static_assert(test());
  static_assert(test_fixed_type());
  assert(test_runtime());
  return 0;
}
