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

// constexpr sentinel(sentinel<!Const> s);
//             requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

template <class T>
struct convertible_sentinel_wrapper
{
  explicit convertible_sentinel_wrapper() = default;
  __host__ __device__ constexpr convertible_sentinel_wrapper(const T& it)
      : it_(it)
  {}

  template <class U, cuda::std::enable_if_t<cuda::std::convertible_to<const U&, T>, int> = 0>
  __host__ __device__ constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other)
      : it_(other.it_)
  {}

  __host__ __device__ constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other)
  {
    return self.it_ == other;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const T& other, convertible_sentinel_wrapper const& self)
  {
    return self.it_ == other;
  }
  __host__ __device__ constexpr friend bool operator!=(convertible_sentinel_wrapper const& self, const T& other)
  {
    return self.it_ != other;
  }
  __host__ __device__ constexpr friend bool operator!=(const T& other, convertible_sentinel_wrapper const& self)
  {
    return self.it_ != other;
  }
#endif // TEST_STD_VER <= 2017

  T it_;
};

struct ConstConveritbleView : BufferView<BufferView<int*>*>
{
#if defined(TEST_COMPILER_NVRTC)
  ConstConveritbleView() noexcept = default;

  template <class T>
  __host__ __device__ constexpr ConstConveritbleView(T&& arr) noexcept(
    noexcept(BufferView<BufferView<int*>*>(cuda::std::declval<T>())))
      : BufferView<BufferView<int*>*>(_CUDA_VSTD::forward<T>(arr))
  {}
#else
  using BufferView<BufferView<int*>*>::BufferView;
#endif

  using sentinel       = convertible_sentinel_wrapper<BufferView<int*>*>;
  using const_sentinel = convertible_sentinel_wrapper<const BufferView<int*>*>;

  __host__ __device__ constexpr BufferView<int*>* begin()
  {
    return data_;
  }
  __host__ __device__ constexpr const BufferView<int*>* begin() const
  {
    return data_;
  }
  __host__ __device__ constexpr sentinel end()
  {
    return sentinel(data_ + size_);
  }
  __host__ __device__ constexpr const_sentinel end() const
  {
    return const_sentinel(data_ + size_);
  }
};
static_assert(!cuda::std::ranges::common_range<ConstConveritbleView>);
static_assert(cuda::std::convertible_to<cuda::std::ranges::sentinel_t<ConstConveritbleView>,
                                        cuda::std::ranges::sentinel_t<ConstConveritbleView const>>);
static_assert(!simple_view<ConstConveritbleView>);

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    BufferView<int*> inners[] = {buffer[0], buffer[1], buffer[2]};
    ConstConveritbleView outer(inners);
    cuda::std::ranges::join_view jv(outer);
    auto sent1                                              = jv.end();
    cuda::std::ranges::sentinel_t<const decltype(jv)> sent2 = sent1;
    assert(cuda::std::as_const(jv).begin() != sent2);
    assert(cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 12) == sent2);

    // We cannot create a non-const sentinel from a const sentinel.
    static_assert(!cuda::std::constructible_from<decltype(sent1), decltype(sent2)>);
  }

  {
    // cannot create a const sentinel from a non-const sentinel if the underlying
    // const sentinel cannot be created from the underlying non-const sentinel
    using Inner = BufferView<int*>;
    using ConstInconvertibleOuter =
      BufferView<forward_iterator<const Inner*>,
                 sentinel_wrapper<forward_iterator<const Inner*>>,
                 bidirectional_iterator<Inner*>,
                 sentinel_wrapper<bidirectional_iterator<Inner*>>>;
    using JoinView       = cuda::std::ranges::join_view<ConstInconvertibleOuter>;
    using sentinel       = cuda::std::ranges::sentinel_t<JoinView>;
    using const_sentinel = cuda::std::ranges::sentinel_t<const JoinView>;
    static_assert(!cuda::std::constructible_from<sentinel, const_sentinel>);
    static_assert(!cuda::std::constructible_from<const_sentinel, sentinel>);
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
