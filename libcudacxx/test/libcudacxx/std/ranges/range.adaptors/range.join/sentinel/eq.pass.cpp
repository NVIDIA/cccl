//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<bool OtherConst>
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "../types.h"

template <class Iter, class Sent>
inline constexpr bool EqualityComparable = cuda::std::invocable<cuda::std::equal_to<>, const Iter&, const Sent&>;

using Iterator      = random_access_iterator<BufferView<int*>*>;
using ConstIterator = random_access_iterator<const BufferView<int*>*>;

template <bool Const>
struct ConstComparableSentinel
{
  using Iter = cuda::std::conditional_t<Const, ConstIterator, Iterator>;
  Iter iter_;

  explicit ConstComparableSentinel() = default;
  __host__ __device__ constexpr explicit ConstComparableSentinel(const Iter& it)
      : iter_(it)
  {}

  __host__ __device__ constexpr friend bool operator==(const Iterator& i, const ConstComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const ConstComparableSentinel& s, const Iterator& i)
  {
    return base(i) == base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const Iterator& i, const ConstComparableSentinel& s)
  {
    return base(i) != base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ConstComparableSentinel& s, const Iterator& i)
  {
    return base(i) != base(s.iter_);
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ constexpr friend bool operator==(const ConstIterator& i, const ConstComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const ConstComparableSentinel& s, const ConstIterator& i)
  {
    return base(i) == base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ConstIterator& i, const ConstComparableSentinel& s)
  {
    return base(i) != base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ConstComparableSentinel& s, const ConstIterator& i)
  {
    return base(i) != base(s.iter_);
  }
#endif // TEST_STD_VER <= 2017
};

struct ConstComparableView : BufferView<BufferView<int*>*>
{
#if defined(TEST_COMPILER_NVRTC)
  ConstComparableView() noexcept = default;

  template <class T>
  __host__ __device__ constexpr ConstComparableView(T&& arr) noexcept(
    noexcept(BufferView<BufferView<int*>*>(cuda::std::declval<T>())))
      : BufferView<BufferView<int*>*>(_CUDA_VSTD::forward<T>(arr))
  {}
#else
  using BufferView<BufferView<int*>*>::BufferView;
#endif

  __host__ __device__ constexpr auto begin()
  {
    return Iterator(data_);
  }
  __host__ __device__ constexpr auto begin() const
  {
    return ConstIterator(data_);
  }
  __host__ __device__ constexpr auto end()
  {
    return ConstComparableSentinel<false>(Iterator(data_ + size_));
  }
  __host__ __device__ constexpr auto end() const
  {
    return ConstComparableSentinel<true>(ConstIterator(data_ + size_));
  }
};

static_assert(EqualityComparable<cuda::std::ranges::iterator_t<ConstComparableView>,
                                 cuda::std::ranges::sentinel_t<const ConstComparableView>>);
static_assert(EqualityComparable<cuda::std::ranges::iterator_t<const ConstComparableView>,
                                 cuda::std::ranges::sentinel_t<ConstComparableView>>);

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  // test iterator<false> == sentinel<false>
  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv               = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));
  }

  // test iterator<false> == sentinel<true>
  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    using ParentT         = cuda::std::remove_all_extents_t<decltype(children)>;
    auto jv               = cuda::std::ranges::join_view(ForwardParentView<ParentT>(children));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(jv.begin(), 16));
  }

  // test iterator<true> == sentinel<true>
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
    using ParentT = cuda::std::remove_all_extents_t<decltype(children)>;
    const auto jv = cuda::std::ranges::join_view(ForwardParentView<ParentT>(children));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));
  }

  // test iterator<Const> == sentinel<!Const>
  {
    BufferView<int*> inners[] = {buffer[0], buffer[1]};
    ConstComparableView outer(inners);
    auto jv = cuda::std::ranges::join_view(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 8));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(jv.begin(), 8));
    assert(jv.end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 8));
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
