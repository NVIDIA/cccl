// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Internal output-padding types shared by DeviceBatchedTopK dispatch and agents.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
template <typename KeyT, typename ValueT, bool HasValue>
struct output_padding_values
{
  static constexpr bool has_value = HasValue;

  KeyT key;
  ValueT value;
};

// Query tag for the optional, stateful padding property. Padding is deliberately not a cuda::execution requirement:
// `require` accepts only stateless requirement types, while these values must survive in the environment until launch.
struct get_output_padding_t
{};

// Carries an assignment-domain padding value alongside an output iterator-of-iterators while forwarding ordinary
// segment access unchanged. Opt-in calls therefore instantiate a distinct kernel through the already-existing output
// iterator template parameter; default calls retain their original kernel type and compile the padding path out.
template <typename OutputSegmentsIteratorT, typename PaddingT>
struct padded_output_segments_iterator
{
  OutputSegmentsIteratorT output_segments;
  PaddingT padding_value;

  template <typename SegmentIndexT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE decltype(auto) operator[](SegmentIndexT segment_id)
  {
    return output_segments[segment_id];
  }
};

template <typename OutputSegmentsIteratorT>
struct output_segments_iterator_traits
{
  static constexpr bool is_padded = false;
  using iterator_type             = OutputSegmentsIteratorT;
};

template <typename OutputSegmentsIteratorT, typename PaddingT>
struct output_segments_iterator_traits<padded_output_segments_iterator<OutputSegmentsIteratorT, PaddingT>>
{
  static constexpr bool is_padded = true;
  using iterator_type             = OutputSegmentsIteratorT;
};

template <typename OutputSegmentsIteratorT>
inline constexpr bool is_padded_output_segments_iterator_v =
  output_segments_iterator_traits<OutputSegmentsIteratorT>::is_padded;

template <typename OutputSegmentsIteratorT, typename PaddingT>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
make_padded_output_segments_iterator(OutputSegmentsIteratorT output_segments, PaddingT padding_value)
{
  return padded_output_segments_iterator<OutputSegmentsIteratorT, PaddingT>{output_segments, padding_value};
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
