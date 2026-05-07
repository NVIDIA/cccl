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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/minimum.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/span>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename ValueT, typename FlagT = bool>
struct augmented_value_t
{
  ValueT value;
  FlagT flag;
};

template <typename ValueT, typename FlagT = bool>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES augmented_value_t(ValueT, FlagT) -> augmented_value_t<ValueT, FlagT>;

template <typename ComputeT, int MaxSegmentsPerWorker>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<MaxSegmentsPerWorker == 1, ComputeT, augmented_value_t<ComputeT>>;

template <typename ToT>
struct initial_value_converter
{
  template <typename FromT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static constexpr ToT cast(FromT v) noexcept
  {
    return static_cast<ToT>(v);
  }
};

template <typename ToValueT, typename ToFlagT>
struct initial_value_converter<augmented_value_t<ToValueT, ToFlagT>>
{
  template <typename FromValueT, typename FromFlagT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static constexpr augmented_value_t<ToValueT, ToFlagT>
  cast(augmented_value_t<FromValueT, FromFlagT> fv) noexcept
  {
    return {static_cast<ToValueT>(fv.value), static_cast<ToFlagT>(fv.flag)};
  }
};

template <typename ToT, typename FromT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr ToT convert_initial_value(FromT v) noexcept
{
  return initial_value_converter<ToT>::cast(v);
}

template <typename BinaryOpT, typename ValueT, typename FlagT = bool>
struct schwarz_scan_op
{
  using fv_t = augmented_value_t<ValueT, FlagT>;
  mutable BinaryOpT scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2) const
  {
    if (o2.flag)
    {
      return o2;
    }
    const ValueT res_value = scan_op(o1.value, o2.value);

    return fv_t{res_value, o1.flag};
  }
};

template <typename V, typename F = bool>
struct packer
{
  _CCCL_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return augmented_value_t<V, F>{v, f};
  }
};

template <typename ScanOp, typename V, typename F = bool>
struct packer_iv
{
  mutable ScanOp op;
  V init_v;

  _CCCL_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    V res = v;
    if (f)
    {
      res = op(init_v, v);
    }
    return augmented_value_t<V, F>{res, f};
  }
};

template <typename V, typename F = bool>
struct projector
{
  _CCCL_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F) const
  {
    return v;
  }
};

template <typename V, typename F = bool>
struct projector_iv
{
  V init_v;

  _CCCL_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
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
template <typename ValueT, unsigned int LinearBinarySearchThreshold = 20>
struct bag_of_segments
{
private:
  ::cuda::std::span<ValueT> m_offsets;

public:
  using logical_offset_t = ValueT;
  using segment_id_t     = ::cuda::std::size_t;

  struct search_data_t
  {
    segment_id_t segment_id;
    logical_offset_t logical_offset;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE bag_of_segments(::cuda::std::span<ValueT> cum_sizes)
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

template <typename ValueT, unsigned int MaxBagSize, unsigned int LinearBinarySearchThreshold = 16>
struct statically_bound_bag_of_segments
{
private:
  ::cuda::std::span<ValueT> m_offsets;

public:
  using logical_offset_t = ValueT;
  using segment_id_t     = ::cuda::std::size_t;

  struct search_data_t
  {
    segment_id_t segment_id;
    logical_offset_t logical_offset;
  };

  static constexpr segment_id_t max_offset_size = MaxBagSize;

  _CCCL_DEVICE _CCCL_FORCEINLINE statically_bound_bag_of_segments(
    ::cuda::std::span<ValueT> cum_sizes, ::cuda::std::integral_constant<unsigned int, MaxBagSize>)
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

template <unsigned int MaxBagSize, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto make_statically_bound_bag_of_segments(::cuda::std::span<ValueT> span)
{
  return statically_bound_bag_of_segments<ValueT, MaxBagSize>{span, {}};
}

template <typename SizeT>
struct bag_of_fixed_size_segments
{
private:
  SizeT m_segment_size;

public:
  using logical_offset_t = SizeT;
  using segment_id_t     = SizeT;

  struct search_data_t
  {
    SizeT segment_id;
    SizeT logical_offset;
  };

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
          unsigned int LinearBinarySearchThreshold = 18>
struct multi_segmented_input_iterator
{
  IterT m_it;
  OffsetT m_start;
  SearcherT m_searcher;
  BeginOffsetIterT m_it_idx_begin;
  mutable ReadTransformT m_read_transform_fn;

  using iterator_concept      = ::cuda::std::random_access_iterator_tag;
  using iterator_category     = ::cuda::std::random_access_iterator_tag;
  using underlying_value_type = ::cuda::std::iter_value_t<IterT>;
  using value_type            = ::cuda::std::invoke_result_t<ReadTransformT, underlying_value_type, bool>;
  using difference_type       = ::cuda::std::remove_cv_t<OffsetT>;
  using reference             = void;
  using pointer               = void;

  static_assert(::cuda::std::is_same_v<difference_type, typename SearcherT::logical_offset_t>,
                "offset types are inconsistent");

  struct __mapping_proxy
  {
    IterT m_it;
    OffsetT m_offset;
    bool m_head_flag;
    ReadTransformT m_read_fn;

    _CCCL_DEVICE _CCCL_FORCEINLINE operator value_type() const
    {
      return m_read_fn(m_it[m_offset], m_head_flag);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(const __mapping_proxy& other)
    {
      return (*this = static_cast<value_type>(other));
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator*() const
  {
    return make_proxy(0);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator[](difference_type n) const
  {
    return make_proxy(n);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE friend multi_segmented_input_iterator
  operator+(const multi_segmented_input_iterator& iter, difference_type n)
  {
    return {iter.m_it, iter.m_start + n, iter.m_searcher, iter.m_it_idx_begin, iter.m_read_transform_fn};
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy make_proxy(difference_type n) const
  {
    const auto [segment_id, rel_offset] = m_searcher.find(m_start + n);

    const auto offset    = m_it_idx_begin[segment_id] + rel_offset;
    const bool head_flag = (rel_offset == 0);

    return {m_it, offset, head_flag, m_read_transform_fn};
  }
};

template <typename IterT, typename OffsetT, typename SearcherT, typename BeginOffsetIterT, typename ReadTransformT>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES
multi_segmented_input_iterator(IterT, OffsetT, SearcherT, BeginOffsetIterT, ReadTransformT)
  -> multi_segmented_input_iterator<IterT,
                                    ::cuda::std::remove_cv_t<OffsetT>,
                                    SearcherT,
                                    BeginOffsetIterT,
                                    ::cuda::std::remove_cv_t<ReadTransformT>>;

template <typename IterT, typename OffsetT, typename SearcherT, typename BeginOffsetIterT, typename WriteTransformT>
struct multi_segmented_output_iterator
{
  IterT m_it;
  OffsetT m_start;
  SearcherT m_searcher;
  BeginOffsetIterT m_it_idx_begin;
  mutable WriteTransformT m_write_transform_fn;

  using iterator_concept  = ::cuda::std::random_access_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using difference_type   = ::cuda::std::remove_cv_t<OffsetT>;
  using reference         = void;
  using pointer           = void;

  static_assert(::cuda::std::is_same_v<difference_type, typename SearcherT::logical_offset_t>,
                "offset types are inconsistent");

  struct __mapping_proxy
  {
    IterT m_it;
    OffsetT m_offset;
    bool m_head_flag;
    WriteTransformT m_write_fn;

    template <typename V, typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(augmented_value_t<V, F> new_value)
    {
      m_it[m_offset] = m_write_fn(new_value.value, m_head_flag);
      return *this;
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator*() const
  {
    return make_proxy(0);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator[](difference_type n) const
  {
    return make_proxy(n);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE friend multi_segmented_output_iterator
  operator+(const multi_segmented_output_iterator& iter, difference_type n)
  {
    return {iter.m_it, iter.m_start + n, iter.m_searcher, iter.m_it_idx_begin, iter.m_write_transform_fn};
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy make_proxy(difference_type n) const
  {
    const auto [segment_id, rel_offset] = m_searcher.find(m_start + n);

    const auto offset    = m_it_idx_begin[segment_id] + rel_offset;
    const bool head_flag = (rel_offset == 0);

    return {m_it, offset, head_flag, m_write_transform_fn};
  }
};

template <typename IterT, typename OffsetT, typename SearcherT, typename BeginOffsetIterT, typename WriteTransformT>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES
multi_segmented_output_iterator(IterT, OffsetT, SearcherT, BeginOffsetIterT, WriteTransformT)
  -> multi_segmented_output_iterator<IterT,
                                     ::cuda::std::remove_cv_t<OffsetT>,
                                     SearcherT,
                                     BeginOffsetIterT,
                                     ::cuda::std::remove_cv_t<WriteTransformT>>;

template <typename PrefixT, typename BinaryOpT>
struct worker_prefix_callback_t
{
  PrefixT& m_exclusive_prefix;
  BinaryOpT m_scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixT current_prefix() const
  {
    return m_exclusive_prefix;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixT operator()(PrefixT block_aggregate)
  {
    const PrefixT previous_prefix = m_exclusive_prefix;
    m_exclusive_prefix            = m_scan_op(m_exclusive_prefix, block_aggregate);
    return previous_prefix;
  }
};

template <typename PrefixT, typename OpT>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES worker_prefix_callback_t(PrefixT&, OpT) -> worker_prefix_callback_t<PrefixT, OpT>;

// ScopeT should implement uniform thread participation,
// valid worker_id, collective scan/reduce, and synchronization behavior
template <::cuda::std::size_t MaxNumSegments,
          typename OffsetT,
          typename ScopeT,
          typename InputBeginOffsetIteratorT,
          typename InputEndOffsetIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE int preprocess_segment_sizes(
  ScopeT scope, InputBeginOffsetIteratorT input_begin_idx_it, InputEndOffsetIteratorT input_end_idx_it, int n_segments)
{
  scope.init_fixed_size_mask();

  static_assert(MaxNumSegments < ::cuda::std::numeric_limits<int>::max(), "MaxNumSegments is too large to cast to int");

  n_segments        = ::cuda::std::min(n_segments, static_cast<int>(MaxNumSegments));
  unsigned n_chunks = ::cuda::ceil_div(n_segments, ScopeT::worker_thread_count);

  OffsetT exclusive_prefix = 0;
  using plus_t             = ::cuda::std::plus<>;
  const plus_t offsets_scan_op{};
  worker_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

  for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
  {
    const unsigned work_id = chunk_id * ScopeT::worker_thread_count + scope.worker_id();

    const OffsetT input_segment_begin = (work_id < n_segments) ? input_begin_idx_it[work_id] : 0;
    const OffsetT input_segment_end   = (work_id < n_segments) ? input_end_idx_it[work_id] : 0;

    // Treat end_offset < begin_offset as an empty segment
    const OffsetT segment_size = (::cuda::std::max) (input_segment_end, input_segment_begin) - input_segment_begin;

    const OffsetT prefix = scope.inclusive_scan_segment_size(segment_size, prefix_callback_op);

    if (work_id < n_segments)
    {
      scope.store_logical_segment_offset(work_id, prefix);
    }
    scope.sync();

    ::cuda::minimum<unsigned int> min_op;

    const unsigned int fixed_size_check =
      ((work_id >= n_segments) || (prefix == (work_id + 1) * scope.first_logical_segment_offset())) ? 1u : 0u;
    const unsigned int worker_fixed_size_check = scope.reduce_fixed_size_check(fixed_size_check, min_op);
    scope.update_fixed_size_mask(worker_fixed_size_check, min_op);
    scope.sync();
  }

  return n_segments;
}

template <int ItemsPerThread,
          int TileItems,
          typename OffsetT,
          typename AccumT,
          typename ScopeT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void single_segment_scan_chunked(
  ScopeT scope,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT input_begin_idx,
  OffsetT input_end_idx,
  OffsetT output_begin_idx,
  ScanOpT scan_op,
  InitValueT initial_value)
{
  const OffsetT segment_items = (::cuda::std::max) (input_end_idx, input_begin_idx) - input_begin_idx;
  const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, TileItems);

  AccumT exclusive_prefix{};
  worker_prefix_callback_t prefix_op{exclusive_prefix, scan_op};

  AccumT thread_values[ItemsPerThread];
  for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
  {
    const OffsetT chunk_begin = input_begin_idx + chunk_id * TileItems;
    const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + TileItems, input_end_idx);

    // chunk_size <= TileItems, casting to int is safe
    const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

    scope.load_single(d_in + chunk_begin, thread_values, chunk_size, AccumT{});
    scope.sync();

    if (chunk_id == 0)
    {
      // Initialize exclusive_prefix, referenced from prefix_op
      scope.scan_first_single(thread_values, initial_value, scan_op, exclusive_prefix);
    }
    else
    {
      scope.scan_later_single(thread_values, scan_op, prefix_op);
    }
    scope.sync();

    scope.store_single(d_out + output_begin_idx + chunk_id * TileItems, thread_values, chunk_size);

    // Avoiding synchronization at the end of last chunk could save up to 10% of performance for very short segments
    if (++chunk_id < n_chunks)
    {
      scope.sync();
    }
  }
}

template <::cuda::std::size_t MaxNumSegments,
          typename OffsetT,
          typename ScopeT,
          typename CumSizesT,
          typename InputBeginOffsetIteratorT,
          typename OutputBeginOffsetIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void select_segment_scan_searcher(
  ScopeT scope,
  CumSizesT cum_sizes,
  int n_segments,
  bool has_fixed_size_segments,
  InputBeginOffsetIteratorT input_begin_idx_it,
  OutputBeginOffsetIteratorT output_begin_idx_it)
{
  const OffsetT items_per_worker = cum_sizes[n_segments - 1];

  if (has_fixed_size_segments && (items_per_worker > 0))
  {
    const auto segment_size = cum_sizes[0];
    _CCCL_ASSERT((segment_size > 0) && ((items_per_worker % segment_size) == 0),
                 "Fixed-size segment invariant violated: segment_size must be positive and divide items_per_worker "
                 "evenly");

    // fast path: all segments have identical size (checked via fixed_size_mask)
    // fixed-size searcher can cheaply identify which segment an element belongs to
    // using segment_id = elem_id / segment_size;
    const bag_of_fixed_size_segments searcher{segment_size};
    scope.scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_worker);
  }
  else
  {
    constexpr bool use_branchless = true;

    if constexpr (use_branchless)
    {
      /*
Benchmark result when use_branchless is true

## varying_size_segments

### [0] NVIDIA RTX A6000

| T{ct} | OffsetT{ct} | Elements{io} | #Seg{io} | #Seg/Worker{io} | Worker{io} | Samples | BWUtil |
|-------|-------------|--------------|----------|-----------------|------------|---------|--------|
|   I32 |         I32 |    2^27      |       57 |             216 |      block |    672x | 51.97% |
|   I32 |         I32 |    2^27      |       57 |              18 |       warp |   2832x | 86.45% |
|   I64 |         I32 |    2^27      |       57 |             216 |      block |    720x | 76.07% |
|   I64 |         I32 |    2^27      |       57 |              18 |       warp |    784x | 86.95% |
       */

      // searcher locates segment_id using branchless (unrolled) linear/binary search in cum_sizes
      const auto searcher = make_statically_bound_bag_of_segments<MaxNumSegments>(cum_sizes);
      scope.scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_worker);
    }
    else
    {
      /*
Result for the same benchmark when use_branchless is false

## varying_size_segments

### [0] NVIDIA RTX A6000

| T{ct} | OffsetT{ct} | Elements{io} | #Seg{io} | #Seg/Worker{io} | Worker{io} | Samples | BWUtil |
|-------|-------------|--------------|----------|-----------------|------------|---------|--------|
|   I32 |         I32 |     2^27     |       57 |             117 |      block |    944x | 39.85% |
|   I32 |         I32 |     2^27     |       57 |              18 |       warp |   2821x | 86.41% |
|   I64 |         I32 |     2^27     |       57 |             117 |      block |    544x | 53.11% |
|   I64 |         I32 |     2^27     |       57 |              18 |       warp |    752x | 86.90% |
       */

      // searcher locates segment_id using linear/binary search in cum_sizes
      // binary search is using while-loop based cub::std::upper_bound
      bag_of_segments searcher{cum_sizes};
      scope.scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_worker);
    }
  }
}

template <bool HasInit,
          bool IsInclusive,
          int ItemsPerThread,
          int TileItems,
          typename OffsetT,
          typename AccumT,
          typename ScopeT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename SearcherT,
          typename InputBeginOffsetIteratorT,
          typename OutputBeginOffsetIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void multi_segment_scan_chunked(
  ScopeT scope,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ScanOpT scan_op,
  InitValueT initial_value,
  const SearcherT& searcher,
  InputBeginOffsetIteratorT input_begin_idx_it,
  OutputBeginOffsetIteratorT output_begin_idx_it,
  OffsetT items_per_worker)
{
  using augmented_accum_t   = augmented_value_t<AccumT>;
  using augmented_scan_op_t = schwarz_scan_op<ScanOpT, AccumT>;

  const OffsetT n_chunks = ::cuda::ceil_div(items_per_worker, TileItems);

  augmented_scan_op_t augmented_scan_op{scan_op};

  augmented_accum_t exclusive_prefix{};
  worker_prefix_callback_t prefix_op{exclusive_prefix, augmented_scan_op};

  augmented_accum_t thread_flag_values[ItemsPerThread];
  for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
  {
    const OffsetT chunk_begin = chunk_id * TileItems;
    const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + TileItems, items_per_worker);

    // chunk_size <= TileItems, casting to int is safe
    const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

    // load values, and pack them into head_flag-value pairs
    {
      constexpr auto oob_default = augmented_accum_t{AccumT{}, false};

      if constexpr (HasInit)
      {
        const packer_iv<ScanOpT, AccumT> packer_op{scan_op, initial_value};
        multi_segmented_input_iterator it_in{d_in, chunk_begin, searcher, input_begin_idx_it, packer_op};

        scope.load_multi(it_in, thread_flag_values, chunk_size, oob_default);
      }
      else
      {
        constexpr packer<AccumT> packer_op{};
        multi_segmented_input_iterator it_in{d_in, chunk_begin, searcher, input_begin_idx_it, packer_op};

        scope.load_multi(it_in, thread_flag_values, chunk_size, oob_default);
      }
    }
    scope.sync();

    {
      if (chunk_id == 0)
      {
        augmented_accum_t augmented_init_value{};
        if constexpr (HasInit)
        {
          augmented_init_value = augmented_accum_t{initial_value, false};
        }
        else
        {
          augmented_init_value = augmented_accum_t{AccumT{}, false};
        }
        // Initialize exclusive_prefix, referenced from prefix_op
        scope.scan_first_multi(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scope.scan_later_multi(thread_flag_values, augmented_scan_op, prefix_op);
      }
    }
    scope.sync();

    // store prefix-scan values, discarding head flags
    {
      const OffsetT output_offset = chunk_id * TileItems;

      if constexpr (IsInclusive)
      {
        constexpr projector<AccumT> projector_op{};
        multi_segmented_output_iterator it_out{d_out, output_offset, searcher, output_begin_idx_it, projector_op};

        scope.store_multi(it_out, thread_flag_values, chunk_size);
      }
      else
      {
        const projector_iv<AccumT> projector_op{initial_value};
        multi_segmented_output_iterator it_out{d_out, output_offset, searcher, output_begin_idx_it, projector_op};

        scope.store_multi(it_out, thread_flag_values, chunk_size);
      }
    }

    // Avoiding synchronization at the end of last chunk could save up to 10% of performance for very short segments
    if (++chunk_id < n_chunks)
    {
      scope.sync();
    }
  }
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
