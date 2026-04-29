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

// [simd.iterator], spec-mandated nested typedefs and iterator_traits
//
// value_type, iterator_category, iterator_concept, difference_type,
// iterator_traits::reference and iterator_traits::pointer for
// basic_vec::iterator, basic_vec::const_iterator, basic_mask::iterator,
// basic_mask::const_iterator.

#include <cuda/std/__simd_>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename Iter, typename ValueType>
TEST_FUNC void check_iter_traits()
{
  using Traits = cuda::std::iterator_traits<Iter>;

  static_assert(cuda::std::is_same_v<typename Iter::value_type, ValueType>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_category, cuda::std::input_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::difference_type, cuda::std::ptrdiff_t>);

  static_assert(cuda::std::is_same_v<typename Traits::value_type, ValueType>);
  static_assert(cuda::std::is_same_v<typename Traits::iterator_category, cuda::std::input_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Traits::iterator_concept, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Traits::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::is_same_v<typename Traits::reference, ValueType>);
  static_assert(cuda::std::is_same_v<typename Traits::pointer, void>);
  static_assert(cuda::std::is_same_v<cuda::std::iter_reference_t<Iter>, ValueType>);
}

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  check_iter_traits<typename Vec::iterator, T>();
  check_iter_traits<typename Vec::const_iterator, T>();
  check_iter_traits<typename Mask::iterator, bool>();
  check_iter_traits<typename Mask::const_iterator, bool>();
}

// The iterator nested typedefs do not depend on N, so a couple of representative
// element types are enough.
TEST_FUNC void test()
{
  test_type<int8_t, 4>();
  test_type<uint64_t, 4>();
  test_type<float, 1>();
}

int main(int, char**)
{
  return 0;
}
