//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/mdspan>

// constexpr index_type stride(rank_type i) const noexcept;
//
//   Constraints: extents_type::rank() > 0 is true.
//
//   Preconditions: i < extents_type::rank() is true.
//
//   Returns: strides_[i].
//
// constexpr array<intptr_t, rank_> strides() const noexcept;
//
//   Returns: strides_.

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <class E, class... Args>
__host__ __device__ constexpr void
test_stride(cuda::std::array<cuda::std::intptr_t, E::rank()> input_strides, cuda::std::intptr_t offset, Args... args)
{
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;
  cuda::std::array<offset_type, E::rank()> strides{};
  for (size_t r = 0; r < E::rank(); r++)
  {
    strides[r] = static_cast<offset_type>(input_strides[r]);
  }
  M m(E(args...), strides, static_cast<offset_type>(offset));

  static_assert(noexcept(m.stride(0)));
  for (size_t r = 0; r < E::rank(); r++)
  {
    assert(strides[r] == m.stride(r));
  }

  static_assert(noexcept(m.strides()));
  auto strides_out = m.strides();
  static_assert(cuda::std::is_same<decltype(strides_out), cuda::dstrides<offset_type, E::rank()>>::value, "");
  for (size_t r = 0; r < E::rank(); r++)
  {
    assert(strides[r] == strides_out.stride(r));
  }
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Basic cases with positive strides
  test_stride<cuda::std::extents<unsigned, D>>(cuda::std::array<cuda::std::intptr_t, 1>{1}, 0, 7);
  test_stride<cuda::std::extents<unsigned, 7>>(cuda::std::array<cuda::std::intptr_t, 1>{1}, 0);
  test_stride<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{8, 1}, 0);
  test_stride<cuda::std::extents<int64_t, D, 8, D, D>>(
    cuda::std::array<cuda::std::intptr_t, 4>{720, 90, 10, 1}, 0, 7, 9, 10);

  // Cases with non-zero offset
  test_stride<cuda::std::extents<unsigned, D>>(cuda::std::array<cuda::std::intptr_t, 1>{1}, 10, 7);
  test_stride<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{8, 1}, 50);

  // Cases with negative strides
  test_stride<cuda::std::extents<int, D>>(cuda::std::array<cuda::std::intptr_t, 1>{-1}, 6, 7);
  test_stride<cuda::std::extents<int, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{-8, 1}, 48);
  test_stride<cuda::std::extents<int, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{8, -1}, 7);
  test_stride<cuda::std::extents<int, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{-8, -1}, 55);

  // Cases with zero strides (broadcasting)
  test_stride<cuda::std::extents<int, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{0, 1}, 0);
  test_stride<cuda::std::extents<int, 7, 8>>(cuda::std::array<cuda::std::intptr_t, 2>{8, 0}, 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
