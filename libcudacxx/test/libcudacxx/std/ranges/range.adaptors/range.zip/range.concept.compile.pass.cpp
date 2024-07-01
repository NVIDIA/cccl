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

// test if zip_view models input_range, forward_range, bidirectional_range,
//                         random_access_range, contiguous_range, common_range
//                         sized_range

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "types.h"

__host__ __device__ void testConceptPair()
{
  int buffer1[2] = {1, 2};
  int buffer2[3] = {1, 2, 3};
  {
    cuda::std::ranges::zip_view v{ContiguousCommonView{buffer1}, ContiguousCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ContiguousNonCommonView{buffer1}, ContiguousNonCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ContiguousNonCommonSized{buffer1}, ContiguousNonCommonSized{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{SizedRandomAccessView{buffer1}, ContiguousCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{SizedRandomAccessView{buffer1}, SizedRandomAccessView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{NonSizedRandomAccessView{buffer1}, NonSizedRandomAccessView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiCommonView{buffer1}, SizedRandomAccessView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiCommonView{buffer1}, BidiCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiCommonView{buffer1}, ForwardSizedView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiNonCommonView{buffer1}, ForwardSizedView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ForwardSizedView{buffer1}, ForwardSizedView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ForwardSizedNonCommon{buffer1}, ForwardSizedView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputCommonView{buffer1}, ForwardSizedView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputCommonView{buffer1}, InputCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputNonCommonView{buffer1}, InputCommonView{buffer2}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }
}

__host__ __device__ void testConceptTuple()
{
  int buffer1[2] = {1, 2};
  int buffer2[3] = {1, 2, 3};
  int buffer3[4] = {1, 2, 3, 4};

  {
    cuda::std::ranges::zip_view v{
      ContiguousCommonView{buffer1}, ContiguousCommonView{buffer2}, ContiguousCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      ContiguousNonCommonView{buffer1}, ContiguousNonCommonView{buffer2}, ContiguousNonCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      ContiguousNonCommonSized{buffer1}, ContiguousNonCommonSized{buffer2}, ContiguousNonCommonSized{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      SizedRandomAccessView{buffer1}, ContiguousCommonView{buffer2}, ContiguousCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      SizedRandomAccessView{buffer1}, SizedRandomAccessView{buffer2}, SizedRandomAccessView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      NonSizedRandomAccessView{buffer1}, NonSizedRandomAccessView{buffer2}, NonSizedRandomAccessView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::contiguous_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{
      BidiCommonView{buffer1}, SizedRandomAccessView{buffer2}, SizedRandomAccessView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiCommonView{buffer1}, BidiCommonView{buffer2}, BidiCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::random_access_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiCommonView{buffer1}, ForwardSizedView{buffer2}, ForwardSizedView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{BidiNonCommonView{buffer1}, ForwardSizedView{buffer2}, ForwardSizedView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ForwardSizedView{buffer1}, ForwardSizedView{buffer2}, ForwardSizedView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{ForwardSizedNonCommon{buffer1}, ForwardSizedView{buffer2}, ForwardSizedView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputCommonView{buffer1}, ForwardSizedView{buffer2}, ForwardSizedView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputCommonView{buffer1}, InputCommonView{buffer2}, InputCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }

  {
    cuda::std::ranges::zip_view v{InputNonCommonView{buffer1}, InputCommonView{buffer2}, InputCommonView{buffer3}};
    using View = decltype(v);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(!cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::sized_range<View>);
  }
}

using OutputIter = cpp17_output_iterator<int*>;
static_assert(cuda::std::output_iterator<OutputIter, int>);

struct OutputView : cuda::std::ranges::view_base
{
  __host__ __device__ OutputIter begin() const;
  __host__ __device__ sentinel_wrapper<OutputIter> end() const;
};
static_assert(cuda::std::ranges::output_range<OutputView, int>);
static_assert(!cuda::std::ranges::input_range<OutputView>);

#if TEST_STD_VER >= 2020
template <class... Ts>
concept zippable = requires { typename cuda::std::ranges::zip_view<Ts...>; };
#else
template <class... Ts>
inline constexpr bool zippable = cuda::std::invocable<cuda::std::ranges::views::__zip::__fn, Ts...>;
#endif // TEST_STD_VER <= 2017

// output_range is not supported
static_assert(!zippable<OutputView>);
static_assert(!zippable<SimpleCommon, OutputView>);
static_assert(zippable<SimpleCommon>);

int main(int, char**)
{
  return 0;
}
