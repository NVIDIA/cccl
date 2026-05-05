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

// [simd.iterator], begin/end on basic_vec and basic_mask
//
// constexpr iterator begin() noexcept;
// constexpr const_iterator begin() const noexcept;
// constexpr const_iterator cbegin() const noexcept;
// constexpr default_sentinel_t end() const noexcept;
// constexpr default_sentinel_t cend() const noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// begin/end member return types, noexcept, cuda::std::begin/end round-trips,
// begin() to end() count is N elements

template <int N, typename Type>
TEST_FUNC constexpr void test_begin_end(Type& obj, const Type& const_obj)
{
  using Iter      = typename Type::iterator;
  using ConstIter = typename Type::const_iterator;
  using Sentinel  = cuda::std::default_sentinel_t;

  // member return types
  static_assert(cuda::std::is_same_v<decltype(obj.begin()), Iter>);
  static_assert(cuda::std::is_same_v<decltype(const_obj.begin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(const_obj.cbegin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(obj.end()), Sentinel>);
  static_assert(cuda::std::is_same_v<decltype(const_obj.end()), Sentinel>);
  static_assert(cuda::std::is_same_v<decltype(const_obj.cend()), Sentinel>);

  // noexcept
  static_assert(noexcept(obj.begin()));
  static_assert(noexcept(const_obj.begin()));
  static_assert(noexcept(const_obj.cbegin()));
  static_assert(noexcept(obj.end()));
  static_assert(noexcept(const_obj.end()));
  static_assert(noexcept(const_obj.cend()));

  // cuda::std::begin / end return types
  static_assert(cuda::std::is_same_v<decltype(cuda::std::begin(obj)), Iter>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::begin(const_obj)), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cbegin(const_obj)), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::end(obj)), Sentinel>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::end(const_obj)), Sentinel>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cend(const_obj)), Sentinel>);

  // cuda::std::begin / end round-trip
  assert(obj.begin() == cuda::std::begin(obj));
  assert(const_obj.begin() == cuda::std::begin(const_obj));
  assert(const_obj.cbegin() == cuda::std::cbegin(const_obj));
  assert(cuda::std::begin(obj) != cuda::std::end(obj));
  assert(cuda::std::begin(const_obj) != cuda::std::end(const_obj));
  assert(cuda::std::cbegin(const_obj) != cuda::std::cend(const_obj));

  const auto end = obj.begin() + N;
  assert(cuda::std::distance(obj.begin(), end) == N);
  assert(end == obj.end());
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec vec{};
  const Vec const_vec{};
  test_begin_end<N>(vec, const_vec);

  Mask mask(true);
  const Mask const_mask(true);
  test_begin_end<N>(mask, const_mask);
}

// The begin/end surface does not depend on T, so a couple of representative
// (type, N) combinations are enough.
TEST_FUNC constexpr bool test()
{
  test_type<int32_t, 1>();
  test_type<float, 4>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
