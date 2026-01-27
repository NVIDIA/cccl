// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/visibility.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
namespace multi_segment_helpers
{
template <typename ValueT, typename FlagT>
using augmented_value_t = ::cuda::std::tuple<ValueT, FlagT>;

template <typename ComputeT, int MaxSegmentsPerWorker>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<MaxSegmentsPerWorker == 1, ComputeT, augmented_value_t<ComputeT, bool>>;

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

template <typename ValueT, typename FlagT, typename BinaryOpT>
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
    const auto o2_value    = get_value(::cuda::std::move(o2));
    const auto o1_value    = get_value(o1);
    const ValueT res_value = scan_op(o1_value, o2_value);

    return make_value_flag(res_value, get_flag(::cuda::std::move(o1)));
  }
};

template <typename V, typename F>
struct packer
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return make_value_flag(v, f);
  }
};

template <typename V, typename F, typename ScanOp>
struct packer_iv
{
  V init_v;
  ScanOp& op;

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

template <typename V, typename F>
struct projector
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F) const
  {
    return v;
  }
};

template <typename V, typename F>
struct projector_iv
{
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return (f) ? init_v : v;
  }
};

template <typename IterT,
          typename OffsetT,
          typename SpanT,
          typename BeginOffsetIterT,
          typename ReadTransformT,
          typename WriteTransformT>
struct multi_segmented_iterator
{
  IterT m_it;
  OffsetT m_start;
  SpanT m_offsets;
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

  static_assert(::cuda::std::is_same_v<difference_type, typename SpanT::value_type>, "types are inconsistent");

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
        , m_read_fn(::cuda::std::move(read_fn))
        , m_write_fn(::cuda::std::move(write_fn))
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE operator value_type() const
    {
      return m_read_fn(m_it[m_offset], m_head_flag);
    }

    template <typename V, typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(::cuda::std::tuple<V, F> new_value)
    {
      m_it[m_offset] = m_write_fn(get_value(::cuda::std::move(new_value)), m_head_flag);
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
    SpanT cum_sizes,
    BeginOffsetIterT it_idx_begin,
    ReadTransformT read_fn,
    WriteTransformT write_fn)
      : m_it{it}
      , m_start{start}
      , m_offsets{cum_sizes}
      , m_it_idx_begin{it_idx_begin}
      , m_read_transform_fn{::cuda::std::move(read_fn)}
      , m_write_transform_fn{::cuda::std::move(write_fn)}
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
            iter.m_offsets,
            iter.m_it_idx_begin,
            iter.m_read_transform_fn,
            iter.m_write_transform_fn};
  }

private:
  // Given ordinal logical position in the sequence of input segments comprising several segments,
  // return the segment elements is a part of, and relative position within that segment

  // Linear search
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::tuple<int, difference_type> locate_linear_search(difference_type n) const
  {
    const int offset_size     = static_cast<int>(m_offsets.size());
    const difference_type pos = m_start + n;

    int segment_id = 0;
    auto offset_c  = static_cast<typename SpanT::value_type>(0);

    difference_type shifted_offset = pos;

    _CCCL_PRAGMA_UNROLL(4)
    for (int i = 0; i < offset_size; ++i)
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
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::tuple<int, difference_type> locate_binary_search(difference_type n) const
  {
    const int offset_size     = static_cast<int>(m_offsets.size());
    const difference_type pos = m_start + n;

    const auto beg_it    = m_offsets.data();
    const auto end_it    = beg_it + offset_size;
    const auto ub        = ::cuda::std::upper_bound(beg_it, end_it, pos);
    const int segment_id = ::cuda::std::distance(ub, beg_it);

    const difference_type shifted_offset = (segment_id == 0) ? 0 : m_offsets[segment_id - 1];

    return {segment_id, shifted_offset};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy make_proxy(difference_type n) const
  {
    const auto& [segment_id, rel_offset] = locate_linear_search(n);
    const auto offset                    = m_it_idx_begin[segment_id] + rel_offset;
    const bool head_flag                 = (rel_offset == 0);
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
