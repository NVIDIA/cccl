//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <cuda/experimental/simd.cuh>

#include <testing.cuh>

namespace dp = cuda::experimental::datapar;

namespace
{
struct identity_index_generator
{
  template <class Index>
  __host__ __device__ constexpr int operator()(Index idx) const
  {
    return static_cast<int>(idx);
  }
};

struct double_index_generator
{
  template <class Index>
  __host__ __device__ constexpr int operator()(Index idx) const
  {
    return static_cast<int>(idx * 2);
  }
};

struct alternating_mask_generator
{
  template <class Index>
  __host__ __device__ constexpr bool operator()(Index idx) const
  {
    return (idx % 2) == 0;
  }
};
} // namespace

template <class Simd, class T, ::cuda::std::size_t N>
__host__ __device__ void expect_equal(const Simd& actual, const ::cuda::std::array<T, N>& expected)
{
  static_assert(N == Simd::size(), "Mismatch between expected values and simd width");
  for (::cuda::std::size_t i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(actual[i] == expected[i]);
  }
}

C2H_CCCLRT_TEST("simd.traits", "[simd][traits]")
{
  using abi_t    = dp::simd_abi::fixed_size<4>;
  using simd_t   = dp::simd<int, 4>;
  using mask_t   = dp::simd_mask<int, 4>;
  using other_t  = dp::simd<float, 4>;
  using rebind_t = dp::rebind_simd_t<float, simd_t>;

  STATIC_REQUIRE(dp::is_abi_tag_v<abi_t>);
  STATIC_REQUIRE(!dp::is_abi_tag_v<int>);
  STATIC_REQUIRE(dp::simd_size_v<int, abi_t> == 4);
  STATIC_REQUIRE(dp::simd_size_v<float, abi_t> == 4);
  STATIC_REQUIRE(dp::simd_size_v<void, abi_t> == 0);

  STATIC_REQUIRE(dp::is_simd_v<simd_t>);
  STATIC_REQUIRE(!dp::is_simd_v<int>);
  STATIC_REQUIRE(dp::is_simd_mask_v<mask_t>);
  STATIC_REQUIRE(!dp::is_simd_mask_v<simd_t>);

  STATIC_REQUIRE(dp::is_simd_flag_type_v<dp::element_aligned_tag>);
  STATIC_REQUIRE(dp::is_simd_flag_type_v<dp::vector_aligned_tag>);
  STATIC_REQUIRE(dp::is_simd_flag_type_v<dp::overaligned_tag<64>>);

  STATIC_REQUIRE(simd_t::size() == 4);
  STATIC_REQUIRE(mask_t::size() == simd_t::size());
  STATIC_REQUIRE(dp::memory_alignment_v<simd_t> == alignof(int));
  STATIC_REQUIRE(dp::memory_alignment_v<simd_t, dp::vector_aligned_tag> == alignof(int) * simd_t::size());
  STATIC_REQUIRE(dp::memory_alignment_v<simd_t, dp::overaligned_tag<128>> == 128);
  STATIC_REQUIRE(dp::memory_alignment_v<mask_t> == alignof(bool));
  STATIC_REQUIRE(dp::memory_alignment_v<mask_t, dp::vector_aligned_tag> == alignof(bool) * mask_t::size());

  STATIC_REQUIRE(::cuda::std::is_same_v<rebind_t, other_t>);
}

C2H_CCCLRT_TEST("simd.construction_and_memory", "[simd][construction]")
{
  using simd_t = dp::simd<int, 4>;

  simd_t broadcast(7);
  expect_equal(broadcast, ::cuda::std::array<int, simd_t::size()>{7, 7, 7, 7});

  simd_t generated(double_index_generator{});
  expect_equal(generated, ::cuda::std::array<int, simd_t::size()>{0, 2, 4, 6});

  alignas(64) int storage[simd_t::size()] = {0, 1, 2, 3};
  simd_t from_ptr(storage, dp::overaligned<64>);
  expect_equal(from_ptr, ::cuda::std::array<int, simd_t::size()>{0, 1, 2, 3});

  int roundtrip[simd_t::size()] = {};
  generated.copy_to(roundtrip, dp::overaligned<64>);

  simd_t loaded;
  loaded.copy_from(roundtrip, dp::overaligned<64>);
  expect_equal(loaded, ::cuda::std::array<int, simd_t::size()>{0, 2, 4, 6});

  dp::simd<float, 4> widened(generated);
  expect_equal(widened, ::cuda::std::array<float, simd_t::size()>{0.0F, 2.0F, 4.0F, 6.0F});

  dp::simd<int, 4> assigned = simd_t(identity_index_generator{});
  assigned                  = generated;
  expect_equal(assigned, ::cuda::std::array<int, simd_t::size()>{0, 2, 4, 6});

  auto incremented = generated;
  ++incremented;
  expect_equal(incremented, ::cuda::std::array<int, simd_t::size()>{1, 3, 5, 7});

  auto decremented = incremented;
  decremented--;
  expect_equal(decremented, ::cuda::std::array<int, simd_t::size()>{0, 2, 4, 6});
}

C2H_CCCLRT_TEST("simd.arithmetic_and_comparisons", "[simd][arithmetic]")
{
  using simd_t = dp::simd<int, 4>;
  using mask_t = simd_t::mask_type;

  simd_t lhs(identity_index_generator{});
  simd_t rhs(2);

  auto sum = lhs + rhs;
  expect_equal(sum, ::cuda::std::array<int, simd_t::size()>{2, 3, 4, 5});

  auto difference = sum - 1;
  expect_equal(difference, ::cuda::std::array<int, simd_t::size()>{1, 2, 3, 4});

  auto product = lhs * rhs;
  expect_equal(product, ::cuda::std::array<int, simd_t::size()>{0, 2, 4, 6});

  auto quotient = product / rhs;
  expect_equal(quotient, ::cuda::std::array<int, simd_t::size()>{0, 1, 2, 3});

  auto modulo = product % rhs;
  expect_equal(modulo, ::cuda::std::array<int, simd_t::size()>{0, 0, 0, 0});

  auto bit_and = product & simd_t(3);
  expect_equal(bit_and, ::cuda::std::array<int, simd_t::size()>{0, 2, 0, 2});

  auto bit_or = bit_and | simd_t(4);
  expect_equal(bit_or, ::cuda::std::array<int, simd_t::size()>{4, 6, 4, 6});

  auto bit_xor = bit_and ^ simd_t(1);
  expect_equal(bit_xor, ::cuda::std::array<int, simd_t::size()>{1, 3, 1, 3});

  auto shift_left = simd_t(1) << lhs;
  expect_equal(shift_left, ::cuda::std::array<int, simd_t::size()>{1, 2, 4, 8});

  auto shift_right = shift_left >> simd_t(1);
  expect_equal(shift_right, ::cuda::std::array<int, simd_t::size()>{0, 1, 2, 4});

  mask_t eq_mask = (lhs == lhs);
  CUDAX_REQUIRE(eq_mask.all());
  mask_t lt_mask = (lhs < simd_t(2));
  CUDAX_REQUIRE(lt_mask.count() == 2);
  mask_t ge_mask = (lhs >= simd_t(1));
  CUDAX_REQUIRE(ge_mask.any());
  CUDAX_REQUIRE(!ge_mask.none());

  auto negated = -lhs;
  expect_equal(negated, ::cuda::std::array<int, simd_t::size()>{0, -1, -2, -3});
}

C2H_CCCLRT_TEST("simd.mask", "[simd][mask]")
{
  using mask_t = dp::simd_mask<int, 4>;
  using simd_t = dp::simd<int, 4>;

  mask_t alternating(alternating_mask_generator{});
  expect_equal(alternating, ::cuda::std::array<bool, mask_t::size()>{true, false, true, false});
  CUDAX_REQUIRE(alternating.count() == 2);
  CUDAX_REQUIRE(alternating.any());
  CUDAX_REQUIRE(!alternating.all());
  CUDAX_REQUIRE(!alternating.none());

  mask_t inverted = !alternating;
  expect_equal(inverted, ::cuda::std::array<bool, mask_t::size()>{false, true, false, true});

  mask_t zero = alternating & inverted;
  CUDAX_REQUIRE(zero.none());

  mask_t combined = alternating | inverted;
  CUDAX_REQUIRE(combined.all());

  bool buffer[mask_t::size()] = {};
  alternating.copy_to(buffer);
  mask_t loaded(buffer);
  CUDAX_REQUIRE(loaded == alternating);

  auto vec_from_mask = static_cast<simd_t>(alternating);
  expect_equal(vec_from_mask, ::cuda::std::array<int, simd_t::size()>{1, 0, 1, 0});

  mask_t mutated = alternating;
  mutated[1]     = true;
  mutated[3]     = true;
  CUDAX_REQUIRE(mutated.all());

  mask_t broadcast_true(true);
  CUDAX_REQUIRE(broadcast_true.all());
}
