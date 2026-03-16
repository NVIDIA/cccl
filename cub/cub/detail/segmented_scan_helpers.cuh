// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/upper_bound.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
namespace multi_segment_helpers
{
template <typename ValueT, typename FlagT = bool>
using augmented_value_t = ::cuda::std::tuple<ValueT, FlagT>;

template <typename ComputeT, int MaxSegmentsPerWorker>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<MaxSegmentsPerWorker == 1, ComputeT, augmented_value_t<ComputeT>>;

template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr FlagT get_flag(augmented_value_t<ValueT, FlagT> fv) noexcept
{
  return ::cuda::std::get<1>(fv);
}

template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr ValueT get_value(augmented_value_t<ValueT, FlagT> fv) noexcept
{
  return ::cuda::std::get<0>(fv);
}

template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr augmented_value_t<ValueT, FlagT> make_value_flag(ValueT v, FlagT f) noexcept
{
  return {v, f};
}

template <typename BinaryOpT, typename ValueT, typename FlagT = bool>
struct schwarz_scan_op
{
  using fv_t = augmented_value_t<ValueT, FlagT>;
  BinaryOpT& scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2)
  {
    if (get_flag(o2))
    {
      return o2;
    }
    const auto o2_value    = get_value(o2);
    const auto o1_value    = get_value(o1);
    const ValueT res_value = scan_op(o1_value, o2_value);

    return make_value_flag(res_value, get_flag(o1));
  }
};

template <typename V, typename F = bool>
struct packer
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return make_value_flag(v, f);
  }
};

template <typename ScanOp, typename V, typename F = bool>
struct packer_iv
{
  ScanOp& op;
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    V res = v;
    if (f)
    {
      res = op(init_v, v);
    }
    return make_value_flag(res, f);
  }
};

template <typename V, typename F = bool>
struct projector
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F) const
  {
    return v;
  }
};

template <typename V, typename F = bool>
struct projector_iv
{
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return (f) ? init_v : v;
  }
};

// Given a sequence of segments, specified by cumulative sum of its sizes
// and iterator of offsets to beginning of each segment in some allocation
// the bag_of_segments struct maps a logical identifier of an element,
// 0 <= elem_id < m_offsets[m_offsets.size()-1], to segment id and relative
// offset within the segment and produces offset of the corresponding element
// in the underlying allocation.
template <typename SpanT, unsigned int LinearBinarySearchThreshold = 20>
struct bag_of_segments
{
private:
  SpanT m_offsets;

public:
  using logical_offset_t = typename SpanT::value_type;
  using segment_id_t     = typename SpanT::size_type;
  using search_data_t    = ::cuda::std::tuple<segment_id_t, logical_offset_t>;

  _CCCL_DEVICE _CCCL_FORCEINLINE bag_of_segments(SpanT cum_sizes)
      : m_offsets(cum_sizes)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t find(logical_offset_t elem_id) const
  {
    const bool is_small = (m_offsets.size() < LinearBinarySearchThreshold);
    const auto pos      = (is_small) ? locate_linear_search(elem_id) : locate_binary_search(elem_id);
    return pos;
  }

private:
  // Given ordinal logical position in the sequence of input segments comprising several segments,
  // searcher returns the segment the element is a part of, and its relative position within that segment.

  // This comment applies to both linear_search and binary search functions below:
  //    m_offsets views into array of non-negative non-decreasing values, obtained as
  //    prefix sum of segment sizes. Expectation: 0 <= pos < last element of m_offsets

  // Linear search
  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t locate_linear_search(logical_offset_t pos) const
  {
    const auto offset_size = m_offsets.size();

    segment_id_t segment_id = 0;
    logical_offset_t offset_c{0};

    logical_offset_t shifted_offset = pos;

    _CCCL_PRAGMA_UNROLL(4)
    for (segment_id_t i = 0; i < offset_size; ++i)
    {
      const auto offset_n = m_offsets[i];
      const bool cond     = ((offset_c <= pos) && (pos < offset_n));
      segment_id          = (cond) ? i : segment_id;
      shifted_offset      = (cond) ? pos - offset_c : shifted_offset;
      offset_c            = offset_n;
    }
    return {segment_id, shifted_offset};
  }

  // Binary search
  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t locate_binary_search(logical_offset_t pos) const
  {
    const auto offset_size = m_offsets.size();
    const auto beg_it      = m_offsets.data();
    const auto end_it      = beg_it + offset_size;

    const auto ub = ::cuda::std::upper_bound(beg_it, end_it, pos);

    const segment_id_t segment_id = ::cuda::std::distance(beg_it, ub);

    const logical_offset_t shifted_offset = (segment_id == 0) ? pos : pos - m_offsets[segment_id - 1];

    return {segment_id, shifted_offset};
  }
};

template <typename SpanT, unsigned int MaxBagSize, unsigned int LinearBinarySearchThreshold = 16>
struct statically_bound_bag_of_segments
{
private:
  SpanT m_offsets;

public:
  using logical_offset_t = typename SpanT::value_type;
  using segment_id_t     = typename SpanT::size_type;
  using search_data_t    = ::cuda::std::tuple<segment_id_t, logical_offset_t>;

  static constexpr segment_id_t max_offset_size = MaxBagSize;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  statically_bound_bag_of_segments(SpanT cum_sizes, ::cuda::std::integral_constant<unsigned int, MaxBagSize>)
      : m_offsets(cum_sizes)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t find(logical_offset_t elem_id) const
  {
    if constexpr (max_offset_size < LinearBinarySearchThreshold)
    {
      return locate_linear_search(elem_id);
    }
    else
    {
      return locate_binary_search(elem_id);
    }
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  is_it_past_this_segment(logical_offset_t offset, segment_id_t segment_id, segment_id_t size) const
  {
    return ((segment_id < size) && (m_offsets[segment_id] <= offset));
  }

  // Given ordinal logical position in the sequence of input segments comprising several segments,
  // searcher returns the segment the element is a part of, and its relative position within that segment.

  // This comment applies to both linear_search and binary search functions below:
  //    m_offsets views into array of non-negative non-decreasing values, obtained as
  //    prefix sum of segment sizes. Expectation: 0 <= pos < last element of m_offsets

  // Linear search
  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t locate_linear_search(logical_offset_t pos) const
  {
    const auto n_offsets = m_offsets.size();

    segment_id_t segment_id = 0;

    _CCCL_PRAGMA_UNROLL()
    for (segment_id_t i = 0; i < max_offset_size; ++i)
    {
      segment_id += is_it_past_this_segment(pos, i, n_offsets);
    }

    const logical_offset_t relative_offset = (segment_id == 0) ? pos : pos - m_offsets[segment_id - 1];
    return {segment_id, relative_offset};
  }

  // Branchless binary search
  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t locate_binary_search(logical_offset_t pos) const
  {
    constexpr segment_id_t start = ::cuda::std::bit_ceil(max_offset_size) >> 1;

    const auto offset_size = m_offsets.size();

    segment_id_t segment_id{0};

    _CCCL_PRAGMA_UNROLL_FULL()
    for (segment_id_t step = start; step > 0; step >>= 1)
    {
      const segment_id_t new_segment_id = segment_id + (step - 1);
      const segment_id_t cond           = is_it_past_this_segment(pos, new_segment_id, offset_size);

      segment_id += cond * step;
    }

    const logical_offset_t relative_offset = (segment_id == 0) ? pos : pos - m_offsets[segment_id - 1];

    return {segment_id, relative_offset};
  }
};

template <unsigned int MaxBagSize, typename SpanT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto make_statically_bound_bag_of_segments(SpanT span)
{
  return statically_bound_bag_of_segments<SpanT, MaxBagSize>{span, {}};
}

template <typename SizeT>
struct bag_of_fixed_size_segments
{
private:
  SizeT m_segment_size;

public:
  using logical_offset_t = SizeT;
  using segment_id_t     = SizeT;
  using search_data_t    = ::cuda::std::tuple<SizeT, SizeT>;

  _CCCL_DEVICE _CCCL_FORCEINLINE bag_of_fixed_size_segments(SizeT segment_size)
      : m_segment_size(segment_size)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE search_data_t find(logical_offset_t elem_id) const
  {
    const SizeT segment_id      = elem_id / m_segment_size;
    const SizeT relative_offset = elem_id - segment_id * m_segment_size;

    return {segment_id, relative_offset};
  }
};

template <typename IterT,
          typename OffsetT,
          typename SearcherT,
          typename BeginOffsetIterT,
          typename ReadTransformT,
          typename WriteTransformT,
          unsigned int LinearBinarySearchThreshold = 18>
struct multi_segmented_iterator
{
  IterT m_it;
  OffsetT m_start;
  SearcherT m_searcher;
  BeginOffsetIterT m_it_idx_begin;
  ReadTransformT m_read_transform_fn;
  WriteTransformT m_write_transform_fn;

  using iterator_concept      = ::cuda::std::random_access_iterator_tag;
  using iterator_category     = ::cuda::std::random_access_iterator_tag;
  using underlying_value_type = ::cuda::std::iter_value_t<IterT>;
  using value_type            = ::cuda::std::invoke_result_t<ReadTransformT, underlying_value_type, bool>;
  using difference_type       = ::cuda::std::remove_cv_t<OffsetT>;
  using reference             = void;
  using pointer               = void;

  static_assert(::cuda::std::is_convertible_v<::cuda::std::invoke_result_t<WriteTransformT, underlying_value_type, bool>,
                                              underlying_value_type>,
                "Write transform function return value must be convertible to underlying iterator value type");
  static_assert(::cuda::std::is_same_v<difference_type, typename SearcherT::logical_offset_t>,
                "offset types are inconsistent");

  struct __mapping_proxy
  {
    IterT m_it;
    OffsetT m_offset;
    bool m_head_flag;
    ReadTransformT m_read_fn;
    WriteTransformT m_write_fn;

    _CCCL_DEVICE _CCCL_FORCEINLINE explicit __mapping_proxy(
      IterT it, OffsetT offset, bool head_flag, ReadTransformT read_fn, WriteTransformT write_fn)
        : m_it(it)
        , m_offset(offset)
        , m_head_flag(head_flag)
        , m_read_fn(read_fn)
        , m_write_fn(write_fn)
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE operator value_type() const
    {
      return m_read_fn(m_it[m_offset], m_head_flag);
    }

    template <typename V, typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(::cuda::std::tuple<V, F> new_value)
    {
      m_it[m_offset] = m_write_fn(get_value(new_value), m_head_flag);
      return *this;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(const __mapping_proxy& other)
    {
      return (*this = static_cast<value_type>(other));
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE multi_segmented_iterator(
    IterT it,
    OffsetT start,
    SearcherT searcher,
    BeginOffsetIterT it_idx_begin,
    ReadTransformT read_fn,
    WriteTransformT write_fn)
      : m_it{it}
      , m_start{start}
      , m_searcher{searcher}
      , m_it_idx_begin{it_idx_begin}
      , m_read_transform_fn{read_fn}
      , m_write_transform_fn{write_fn}
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator*() const
  {
    return make_proxy(0);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator[](difference_type n) const
  {
    return make_proxy(n);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE friend multi_segmented_iterator
  operator+(const multi_segmented_iterator& iter, difference_type n)
  {
    return {iter.m_it,
            iter.m_start + n,
            iter.m_searcher,
            iter.m_it_idx_begin,
            iter.m_read_transform_fn,
            iter.m_write_transform_fn};
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy make_proxy(difference_type n) const
  {
    const auto [segment_id, rel_offset] = m_searcher.find(m_start + n);

    const auto offset    = m_it_idx_begin[segment_id] + rel_offset;
    const bool head_flag = (rel_offset == 0);

    return __mapping_proxy(m_it, offset, head_flag, m_read_transform_fn, m_write_transform_fn);
  }
};
} // namespace multi_segment_helpers

template <typename PrefixT, typename BinaryOpT>
struct worker_prefix_callback_t
{
  PrefixT& m_exclusive_prefix;
  BinaryOpT& m_scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE worker_prefix_callback_t(PrefixT& prefix, BinaryOpT& op)
      : m_exclusive_prefix(prefix)
      , m_scan_op(op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixT operator()(PrefixT block_aggregate)
  {
    const PrefixT previous_prefix = m_exclusive_prefix;
    m_exclusive_prefix            = m_scan_op(m_exclusive_prefix, block_aggregate);
    return previous_prefix;
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
