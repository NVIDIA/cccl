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

// [simd.mask.ctor], basic_mask constructors
//
// constexpr explicit basic_mask(value_type) noexcept;                          // broadcast
// constexpr explicit basic_mask(const basic_mask<UBytes, UAbi>&) noexcept;     // converting
// constexpr explicit basic_mask(Generator&&);                                  // generator
// constexpr basic_mask(const bitset<size()>&) noexcept;                        // bitset
// constexpr explicit basic_mask(unsigned-integer) noexcept;                    // unsigned integer

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__simd_>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// member types and size

template <int Bytes, int N>
TEST_FUNC constexpr void test_member_types()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<typename Mask::value_type, bool>);
  static_assert(cuda::std::is_same_v<typename Mask::abi_type, simd::fixed_size<N>>);
  static_assert(Mask::size() == N);
  static_assert(cuda::std::is_trivially_copyable_v<Mask>);
}

//----------------------------------------------------------------------------------------------------------------------
// default construction: value-initializes all elements to false

template <int Bytes, int N>
TEST_FUNC constexpr void test_default_ctor()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask{};
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// copy construction and copy assignment

template <int Bytes, int N>
TEST_FUNC constexpr void test_copy()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask original(is_even{});

  Mask copied(original);
  for (int i = 0; i < N; ++i)
  {
    assert(copied[i] == original[i]);
  }

  Mask assigned(false);
  assigned = original;
  for (int i = 0; i < N; ++i)
  {
    assert(assigned[i] == original[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor

template <int Bytes, int N>
TEST_FUNC constexpr void test_broadcast()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  static_assert(noexcept(Mask(true)));

  Mask all_true(true);
  Mask all_false(false);
  for (int i = 0; i < N; ++i)
  {
    assert(all_true[i] == true);
    assert(all_false[i] == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// converting constructor

template <int Bytes, int UBytes, int N>
TEST_FUNC constexpr void test_converting()
{
  using Src = simd::basic_mask<UBytes, simd::fixed_size<N>>;
  using Dst = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Src src(is_even{});
  static_assert(noexcept(Dst(src)));

  Dst dst(src);
  for (int i = 0; i < N; ++i)
  {
    assert(dst[i] == src[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// generator constructor

template <int Bytes, int N>
TEST_FUNC constexpr void test_generator()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
#if _CCCL_COMPILER(GCC, !=, 7)
  static_assert(!noexcept(Mask(is_even{})));
#endif

  Mask mask(is_even{});
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// bitset constructor

template <int Bytes, int N>
TEST_FUNC constexpr void test_bitset()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  cuda::std::bitset<N> bitset;
  static_assert(noexcept(Mask(bitset)));

  for (int i = 0; i < N; ++i)
  {
    bitset.set(i, (i % 2 == 0));
  }
  Mask mask(bitset);
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unsigned integer constructor

template <int Bytes, int N, typename U>
TEST_FUNC constexpr void test_unsigned_int()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  static_assert(noexcept(Mask(U{0})));

  Mask mask(U{0});
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == false);
  }

  constexpr int num_bits  = cuda::std::__num_bits_v<U>;
  constexpr int mask_bits = cuda::std::min(N, num_bits);
  Mask all_one(static_cast<U>(~U{0}));
  for (int i = 0; i < mask_bits; ++i)
  {
    assert(all_one[i] == true);
  }

  if constexpr (N >= 4)
  {
    Mask m_pat(U{0b101});
    assert(m_pat[0] == true);
    assert(m_pat[1] == false);
    assert(m_pat[2] == true);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// SFINAE and explicit constraints

template <int Bytes, int N>
TEST_FUNC constexpr void test_sfinae()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  // broadcast: only accepts bool
  static_assert(!cuda::std::is_constructible_v<Mask, int>);
  // broadcast: must be explicit
  static_assert(!cuda::std::is_convertible_v<bool, Mask>);

  // converting: requires matching element count
  using MaskDifferentSize = simd::basic_mask<Bytes, simd::fixed_size<N + 1>>;
  static_assert(!cuda::std::is_constructible_v<Mask, const MaskDifferentSize&>);
  // converting: must be explicit
  using MaskOtherBytes = simd::basic_mask<(Bytes == 1 ? 2 : 1), simd::fixed_size<N>>;
  static_assert(!cuda::std::is_convertible_v<const MaskOtherBytes&, Mask>);

  // generator: rejects non-callable types
  static_assert(!cuda::std::is_constructible_v<Mask, wrong_generator>);
  // generator: must be explicit
  static_assert(!cuda::std::is_convertible_v<is_even, Mask>);

  // bitset: requires matching size
  static_assert(!cuda::std::is_constructible_v<Mask, cuda::std::bitset<N + 1>>);
  // bitset: is implicit
  static_assert(cuda::std::is_convertible_v<const cuda::std::bitset<N>&, Mask>);

  // unsigned integer: must be explicit
  static_assert(!cuda::std::is_convertible_v<uint32_t, Mask>);
  // unsigned integer: signed integers are rejected
  static_assert(!cuda::std::is_constructible_v<Mask, int>);
  static_assert(!cuda::std::is_constructible_v<Mask, long long>);
  // unsigned integer: bool is rejected by the unsigned-integer overload (explicit is allowed with the bool overload)
  static_assert(!cuda::std::is_convertible_v<bool, Mask>);
  // unsigned integer: unsigned character types are accepted
  static_assert(cuda::std::is_constructible_v<Mask, char16_t>);
  static_assert(cuda::std::is_constructible_v<Mask, char32_t>);
#if _CCCL_HAS_CHAR8_T()
  static_assert(cuda::std::is_constructible_v<Mask, char8_t>);
#endif // _CCCL_HAS_CHAR8_T()
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes, int N>
TEST_FUNC constexpr void test_size()
{
  test_member_types<Bytes, N>();
  test_default_ctor<Bytes, N>();
  test_copy<Bytes, N>();
  test_broadcast<Bytes, N>();
  test_generator<Bytes, N>();
  test_bitset<Bytes, N>();
  test_unsigned_int<Bytes, N, uint8_t>();
  test_unsigned_int<Bytes, N, uint16_t>();
  test_unsigned_int<Bytes, N, uint32_t>();
  test_unsigned_int<Bytes, N, uint64_t>();
  test_sfinae<Bytes, N>();
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_size<Bytes, 1>();
  test_size<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif

  // test_converting  N1: Destination type size, N2: Source type size, N3: Mask number of elements
  test_converting<4, 2, 4>(); // 4 -> 2, 4 elements
  test_converting<2, 4, 4>(); // 2 -> 4, 4 elements
  test_converting<1, 8, 4>(); // 1 -> 8, 4 elements
  test_converting<8, 1, 4>(); // 8 -> 1, 4 elements
  test_converting<4, 4, 4>(); // 4 -> 4, 4 elements

  test_converting<1, 2, 1>(); // 1 -> 2, 1 element
  test_converting<1, 2, 2>(); // 1 -> 2, 2 elements
  test_converting<1, 2, 8>(); // 1 -> 2, 8 elements
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.mask.overview] enable/disable boundary: basic_mask<Bytes, fixed_size<N>> is enabled iff Bytes is a
// vectorizable byte size and N in [1, 64]

TEST_FUNC constexpr void test_enable_abi_boundary()
{
  // enabled at the range boundaries
  static_assert(cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<1>>>);
  static_assert(cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<64>>>);

  // disabled outside the [1, 64] range
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<0>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<65>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<100>>>);
  static_assert(!cuda::std::is_default_constructible_v<simd::basic_mask<4, simd::fixed_size<-1>>>);

  // the disabled specialization has all special members deleted
  using DisabledMask = simd::basic_mask<4, simd::fixed_size<65>>;
  static_assert(!cuda::std::is_default_constructible_v<DisabledMask>);
  static_assert(!cuda::std::is_copy_constructible_v<DisabledMask>);
  static_assert(!cuda::std::is_copy_assignable_v<DisabledMask>);
  static_assert(!cuda::std::is_destructible_v<DisabledMask>);

  // the disabled specialization still exposes value_type / abi_type
  static_assert(cuda::std::is_same_v<DisabledMask::value_type, bool>);
  static_assert(cuda::std::is_same_v<DisabledMask::abi_type, simd::fixed_size<65>>);
}

int main(int, char**)
{
  test_enable_abi_boundary();
  assert(test());
  static_assert(test());
  return 0;
}
