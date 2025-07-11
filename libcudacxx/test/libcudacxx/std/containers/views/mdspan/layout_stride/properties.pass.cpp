//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// namespace std {
//   template<class Extents>
//   class layout_stride::mapping {
//
//     ...
//     static constexpr bool is_always_unique() noexcept { return true; }
//     static constexpr bool is_always_exhaustive() noexcept { return false; }
//     static constexpr bool is_always_strided() noexcept { return true; }
//
//     static constexpr bool is_unique() noexcept { return true; }
//     static constexpr bool is_exhaustive() noexcept;
//     static constexpr bool is_strided() noexcept { return true; }
//     ...
//   };
// }
//
//
// layout_stride::mapping<E> is a trivially copyable type that models regular for each E.
//
// constexpr bool is_exhaustive() const noexcept;
//
// Returns:
//   - true if rank_ is 0.
//   - Otherwise, true if there is a permutation P of the integers in the range [0, rank_) such that
//     stride(p0) equals 1, and stride(pi) equals stride(pi_1) * extents().extent(pi_1) for i in the
//     range [1, rank_), where pi is the ith element of P.
//   - Otherwise, false.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class E, class M, cuda::std::enable_if_t<(E::rank() > 0), int> = 0>
__host__ __device__ constexpr void
test_strides(E ext, M& m, const M& c_m, cuda::std::array<typename E::index_type, E::rank()> strides)
{
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    assert(m.stride(r) == strides[r]);
    assert(c_m.stride(r) == strides[r]);
    static_assert(noexcept(m.stride(r)));
    static_assert(noexcept(c_m.stride(r)));
  }

  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    if (ext.extent(r) == 0)
    {
      expected_size = 0;
      break;
    }
    expected_size += (ext.extent(r) - 1) * static_cast<typename E::index_type>(strides[r]);
  }
  assert(m.required_span_size() == expected_size);
  assert(c_m.required_span_size() == expected_size);
  static_assert(noexcept(m.required_span_size()));
  static_assert(noexcept(c_m.required_span_size()));
}
template <class E, class M, cuda::std::enable_if_t<(E::rank() == 0), int> = 0>
__host__ __device__ constexpr void
test_strides(E, M& m, const M& c_m, cuda::std::array<typename E::index_type, E::rank()> strides)
{
  typename E::index_type expected_size = 1;
  assert(m.required_span_size() == expected_size);
  assert(c_m.required_span_size() == expected_size);
  static_assert(noexcept(m.required_span_size()));
  static_assert(noexcept(c_m.required_span_size()));
}

template <class E>
__host__ __device__ constexpr void
test_layout_mapping_stride(E ext, cuda::std::array<typename E::index_type, E::rank()> strides, bool exhaustive)
{
  using M = cuda::std::layout_stride::template mapping<E>;
  M m(ext, strides);
  const M c_m = m;
  assert(m.strides() == strides);
  assert(c_m.strides() == strides);
  assert(m.extents() == ext);
  assert(c_m.extents() == ext);
  assert(M::is_unique() == true);
  assert(m.is_exhaustive() == exhaustive);
  assert(c_m.is_exhaustive() == exhaustive);
  assert(M::is_strided() == true);
  assert(M::is_always_unique() == true);
  assert(M::is_always_exhaustive() == false);
  assert(M::is_always_strided() == true);

  static_assert(noexcept(m.strides()));
  static_assert(noexcept(c_m.strides()));
  static_assert(noexcept(m.extents()));
  static_assert(noexcept(c_m.extents()));
  static_assert(noexcept(M::is_unique()));
  static_assert(noexcept(m.is_exhaustive()));
  static_assert(noexcept(c_m.is_exhaustive()));
  static_assert(noexcept(M::is_strided()));
  static_assert(noexcept(M::is_always_unique()));
  static_assert(noexcept(M::is_always_exhaustive()));
  static_assert(noexcept(M::is_always_strided()));

  test_strides(ext, m, c_m, strides);

  static_assert(cuda::std::is_trivially_copyable_v<M>);
  static_assert(cuda::std::regular<M>);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_layout_mapping_stride(cuda::std::extents<int>(), cuda::std::array<int, 0>{}, true);
  test_layout_mapping_stride(cuda::std::extents<char, 4, 5>(), cuda::std::array<char, 2>{1, 4}, true);
  test_layout_mapping_stride(cuda::std::extents<char, 4, 5>(), cuda::std::array<char, 2>{1, 5}, false);
  test_layout_mapping_stride(cuda::std::extents<unsigned, D, 4>(7), cuda::std::array<unsigned, 2>{20, 2}, false);
  test_layout_mapping_stride(
    cuda::std::extents<size_t, D, D, D, D>(3, 3, 3, 3), cuda::std::array<size_t, 4>{3, 1, 9, 27}, true);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
