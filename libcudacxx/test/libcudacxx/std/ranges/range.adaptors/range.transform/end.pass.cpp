//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr sentinel<false> end();
// constexpr iterator<false> end() requires common_range<V>;
// constexpr sentinel<true> end() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;
// constexpr iterator<true> end() const
//   requires common_range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT HasConstQualifiedEnd = _CCCL_REQUIRES_EXPR((T), const T& t)((t.end()));

__host__ __device__ constexpr bool test()
{
  {
    using TransformView = cuda::std::ranges::transform_view<ForwardView, PlusOneMutable>;
    static_assert(cuda::std::ranges::common_range<TransformView>);
    TransformView tv{};
    auto it  = tv.end();
    using It = decltype(it);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&>(it).base()), const forward_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&&>(it).base()), forward_iterator<int*>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&>(it).base()), const forward_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&&>(it).base()), const forward_iterator<int*>&>);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(cuda::std::move(it).base()) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<InputView, PlusOneMutable>;
    static_assert(!cuda::std::ranges::common_range<TransformView>);
    TransformView tv{};
    auto sent  = tv.end();
    using Sent = decltype(sent);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const Sent&>(sent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const Sent&&>(sent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    assert(base(base(sent.base())) == globalBuff + 8);
    assert(base(base(cuda::std::move(sent).base())) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<InputView, PlusOne>;
    static_assert(!cuda::std::ranges::common_range<TransformView>);
    TransformView tv{};
    auto sent  = tv.end();
    using Sent = decltype(sent);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const Sent&>(sent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const Sent&&>(sent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    assert(base(base(sent.base())) == globalBuff + 8);
    assert(base(base(cuda::std::move(sent).base())) == globalBuff + 8);

    auto csent  = cuda::std::as_const(tv).end();
    using CSent = decltype(csent);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<CSent&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<CSent&&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const CSent&>(csent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const CSent&&>(csent).base()),
                                       sentinel_wrapper<cpp20_input_iterator<int*>>>);
    assert(base(base(csent.base())) == globalBuff + 8);
    assert(base(base(cuda::std::move(csent).base())) == globalBuff + 8);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    static_assert(cuda::std::ranges::common_range<TransformView>);
    TransformView tv{};
    auto it  = tv.end();
    using It = decltype(it);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&&>(it).base()), int*>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&&>(it).base()), int* const&>);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(cuda::std::move(it).base()) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
    static_assert(cuda::std::ranges::common_range<TransformView>);
    TransformView tv{};
    auto it  = tv.end();
    using It = decltype(it);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&&>(it).base()), int*>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&&>(it).base()), int* const&>);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(cuda::std::move(it).base()) == globalBuff + 8);

    auto csent  = cuda::std::as_const(tv).end();
    using CSent = decltype(csent);
    static_assert(cuda::std::is_same_v<decltype(static_cast<CSent&>(csent).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<CSent&&>(csent).base()), int*>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const CSent&>(csent).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const CSent&&>(csent).base()), int* const&>);
    assert(base(base(csent.base())) == globalBuff + 8);
    assert(base(base(cuda::std::move(csent).base())) == globalBuff + 8);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
