// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_SORT_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_SORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_copy.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/detail/vector_base.h>

#include <cuda/__container/buffer.h>
#include <cuda/__container/make_buffer_with_pool.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__launch/launch.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__algorithm/copy_if.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__random/bernoulli_distribution.h>
#include <cuda/std/__random/philox_engine.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/repeat_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm_utils.h>
#include <cuda/experimental/__multi_gpu/communicator.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail
{
// TODO(jfaibussowit):
//
// Parallelize with multiple threads (but not too many!). __I_j is O(p-1), so in
// practice at the absolute max a few thousand if you are running on the worlds
// largest supercomputers.
template <class _Iter, class _Sent, class _Tp, class _BinaryOp>
_CCCL_KERNEL_ATTRIBUTES void __sample_probes_kernel(
  ::cuda::std::bernoulli_distribution __keep,
  _Iter __begin,
  _Sent __end,
  ::cuda::std::span<const ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>> __I_j,
  _BinaryOp __cmp,
  _Tp* __samples_iter,
  ::cuda::std::size_t* __samples_size_bytes)
{
  if (threadIdx.x != 0)
  {
    // Just in case
    return;
  }

  // Only 1 thread for now, but I guess it cannot hurt to seed?
  auto __gen                       = ::cuda::std::philox4x64{::clock()};
  auto* const __samples_iter_begin = __samples_iter;

  // By value so that load from global memory happens only once
  for (const auto [__lo, __hi] : __I_j)
  {
    // Sample from the union of splitter intervals. Splitter intervals are
    // disjoint or identical; lo_it skips an identical interval already covered
    // by an earlier splitter.
    const auto __last  = __hi.has_value() ? ::cuda::std::lower_bound(__begin, __end, *__hi, __cmp) : __end;
    const auto __first = __lo.has_value() ? ::cuda::std::lower_bound(__begin, __last, *__lo, __cmp) : __begin;

    _CCCL_ASSERT(__first <= __last, "Inputs are not sorted for binary search");
    __samples_iter = ::cuda::std::copy_if(__first, __last, __samples_iter, [&](const auto&) {
      return __keep(__gen);
    });

    __begin = __last;
  }

  *__samples_size_bytes =
    static_cast<::cuda::std::size_t>(__samples_iter - __samples_iter_begin) * sizeof(*__samples_iter);
}

template <class _Integral>
struct __movable_integral
{
  _Integral __value{};

  _CCCL_HIDE_FROM_ABI constexpr __movable_integral() = default;

  _CCCL_HOST_API constexpr explicit __movable_integral(_Integral __v) noexcept
      : __value{__v}
  {}

  _CCCL_HOST_API operator _Integral() const noexcept
  {
    return __value;
  }

  __movable_integral(const __movable_integral&)            = default;
  __movable_integral& operator=(const __movable_integral&) = default;
  _CCCL_HOST_API __movable_integral(__movable_integral&& __other) noexcept
      : __value{__other.__value}
  {
    __other.__value = _Integral{};
  }

  _CCCL_HOST_API __movable_integral& operator=(__movable_integral&& __other) noexcept
  {
    __value         = __other.__value;
    __other.__value = _Integral{};
    return *this;
  }
};

// The following functors replace what would naturally be written as
// function-local lambdas at their use sites inside _Sorter. They are hoisted to
// named, namespace-scope types on purpose: NVCC cannot name a function-local
// closure as a `__global__` kernel template argument when it generates the host
// registration stub, emitting "insufficient contextual information to determine
// type" in *.cudafe1.stub.c. A named functor is nameable there. They live
// outside _Sorter, and depend only on the types they actually use, so they are
// not re-instantiated per _Env/_BinaryOp.

// Maps a splitter index to its ideal global rank Ni/p. Used as the source of a
// transform_iterator.
struct __ideal_rank_fn
{
  ::cuda::std::uint64_t __N;
  ::cuda::std::uint64_t __comm_size;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint64_t operator()(::cuda::std::uint64_t __i) const noexcept
  {
    return ((__i + 1) * __N) / __comm_size;
  }
};

// Interval-narrowing functor for __update_intervals. Given (target_rank, L, U),
// walks the local probe histogram and tightens the [L, U] bracket, returning
// the updated interval and brackets.
template <class _Tp, class _Bracket>
struct __update_intervals_fn
{
  const _Tp* __probes_begin;
  const ::cuda::std::uint64_t* __hist_begin;
  ::cuda::std::size_t __num_probes;

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::
    tuple<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>, _Bracket, _Bracket>
    operator()(const _Tup& __tup) const noexcept
  {
    auto [__target, __L_i, __U_i]       = __tup;
    ::cuda::std::uint64_t __global_rank = 0;

    for (::cuda::std::uint64_t __j = 0; __j < __num_probes; ++__j)
    {
      const auto __rank = __global_rank;

      __global_rank += __hist_begin[__j];
      if ((__rank < __target) && (__rank > __L_i.__rank))
      {
        __L_i = _Bracket{__rank, __probes_begin[__j]};
      }
      else if ((__rank > __target) && (__rank < __U_i.__rank))
      {
        __U_i = _Bracket{__rank, __probes_begin[__j]};
      }
      else if (__rank == __target)
      {
        __L_i = __U_i = _Bracket{__rank, __probes_begin[__j]};
        break;
      }
    }

    return ::cuda::std::make_tuple(::cuda::std::pair{__L_i.__key, __U_i.__key}, __L_i, __U_i);
  }
};

// Splitter-selection functor for __finalize_splitters. Given (target_rank, L,
// U), returns the bracket endpoint key closest to the target rank, realizing an
// unset (unbounded) endpoint from the probe extrema.
template <class _Tp, class _Probe>
struct __finalize_splitters_fn
{
  _Probe __first_probe;
  _Probe __last_probe;

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp operator()(const _Tup& __tup) const noexcept
  {
    const auto [__target_rank, __L_i, __U_i] = __tup;

    // Pick the bracket endpoint closest to the target rank. The chosen key
    // becomes the splitter used by data exchange.
    const bool __use_L = (__target_rank - __L_i.__rank) <= (__U_i.__rank - __target_rank);
    if (__use_L)
    {
      return __L_i.__key.has_value() ? *__L_i.__key : *__first_probe;
    }
    return __U_i.__key.has_value() ? *__U_i.__key : *__last_probe;
  }
};

template <class _Range>
_CCCL_CONCEPT __can_thrust_no_init_resize = _CCCL_REQUIRES_EXPR((_Range), _Range& __range)(
  requires(::cuda::std::is_trivially_constructible_v<::cuda::std::ranges::range_value_t<_Range>>),
  __range.resize(::cuda::std::size_t{}, ::thrust::no_init));

template <class _Range>
_CCCL_CONCEPT __can_cuda_no_init_resize = _CCCL_REQUIRES_EXPR((_Range), _Range& __range)(
  requires(::cuda::std::is_trivially_constructible_v<::cuda::std::ranges::range_value_t<_Range>>),
  __range.resize(::cuda::std::size_t{}, ::cuda::no_init));

template <class _Range>
_CCCL_HOST_API void __resize_for_overwrite(_Range& __range, ::cuda::std::size_t __size)
{
  if constexpr (__can_thrust_no_init_resize<_Range>)
  {
    __range.resize(__size, ::thrust::no_init);
  }
  else if constexpr (__can_cuda_no_init_resize<_Range>)
  {
    __range.resize(__size, ::cuda::no_init);
  }
  else
  {
    __range.resize(__size);
  }
}

template <class _Tp, class _Env, class _BinaryOp>
struct _Sorter
{
  using __pool_type     = decltype(::cuda::device_default_memory_pool(::cuda::std::declval<::cuda::device_ref>()));
  using __resource_type = ::cuda::std::remove_cvref_t<
    ::cuda::__lazy_call_result_or_t<::cuda::mr::get_memory_resource_t, __pool_type(void), _Env>>;

  // cuda::buffer has no concept of size vs capacity. We need both below because
  // we want to be able to shrink a buffer without reallocating a new one.
  template <class _Up>
  struct __buffer
  {
    template <class _Up2>
    using __buffer_type_for = typename ::cuda::__buffer_type_for_props<_Up2, typename __resource_type::default_queries>;

    using __buff_type = __buffer_type_for<_Up>;

    __buffer(__buffer&&) noexcept            = default;
    __buffer& operator=(__buffer&&) noexcept = default;
    __buffer(const __buffer&)                = default;
    __buffer& operator=(const __buffer& __other)
    {
      __buf_         = __buff_type{__other.__buf_};
      __actual_size_ = __other.__actual_size_;
      return *this;
    }
    // Recreate the most common constructors for buffer that we use below to
    // avoid needing to call make_buffer() over and over again.
    template <class _Resource, class _EnvT>
    _CCCL_HOST_API explicit __buffer(::cuda::stream_ref __stream, _Resource&& __resource, _EnvT&& __env)
        : __buffer{
            __buff_type{__stream, ::cuda::std::forward<_Resource>(__resource), ::cuda::std::forward<_EnvT>(__env)}}
    {}

    template <class _Resource, class _EnvT>
    _CCCL_HOST_API explicit __buffer(
      ::cuda::stream_ref __stream, _Resource&& __resource, ::cuda::std::size_t __size, ::cuda::no_init_t, _EnvT&& __env)
        : __buffer{__buff_type{
            __stream,
            ::cuda::std::forward<_Resource>(__resource),
            __size,
            ::cuda::no_init,
            ::cuda::std::forward<_EnvT>(__env)}}
    {}

    template <class _Resource, class _EnvT>
    _CCCL_HOST_API explicit __buffer(
      ::cuda::stream_ref __stream, _Resource&& __resource, ::cuda::std::size_t __size, const _Up& __value, _EnvT&& __env)
        : __buffer{::cuda::make_buffer<_Up>(
            __stream, ::cuda::std::forward<_Resource>(__resource), __size, __value, ::cuda::std::forward<_EnvT>(__env))}
    {}

    _CCCL_HOST_API explicit __buffer(__buff_type __buf)
        : __buf_{::cuda::std::move(__buf)}
        , __actual_size_{__buf_.size()}
    {}

    _CCCL_HOST_API void resize(::cuda::std::size_t __new_size)
    {
      // TODO(jfaibussowit):
      //
      // This is WRONG
      resize(__new_size, ::cuda::no_init);
    }

    // Grow or shrink the buffer to __new_size. This is effectively
    // std::vector::resize() except that it never touches original values. On
    // growth it will allocate a new buffer with uninitialized values, while on
    // shrinkage it will leave the original values as-is.
    _CCCL_HOST_API void resize(::cuda::std::size_t __new_size, ::cuda::no_init_t)
    {
      if (__new_size > capacity())
      {
        // Don't use __make_empty_like() here. Even if the current buffer
        // doesn't say it can hold the new size, it's possible that the
        // underlying allocation actually *is* big enough. The only thing that
        // knows this is the memory resource, so we first return the existing
        // buffer to the memory resource before creating the new one.
        const auto __stream = __get().stream();
        auto __mr           = __get().memory_resource();
        const auto __align  = __get().alignment();

        __get().destroy();

        __get() = __buff_type{
          __stream,
          ::cuda::std::move(__mr),
          __new_size,
          ::cuda::no_init,
          ::cuda::std::execution::prop{::cuda::allocation_alignment, __align}};
      }
      __actual_size_ = __movable_integral<::cuda::std::size_t>{__new_size};
    }

    template <class _Up2 = _Up>
    [[nodiscard]] _CCCL_HOST_API __buffer_type_for<_Up2> __make_empty_like(::cuda::std::size_t __new_size) const
    {
      // TODO(jfaibussowit):
      //
      // buffer ideally should have a make_buffer_like(source_buffer) helper
      // that does this for us, similar to e.g. numpy.empty_like()
      return __buffer_type_for<_Up2>{
        __get().stream(),
        __get().memory_resource(),
        __new_size,
        ::cuda::no_init,
        ::cuda::std::execution::prop{::cuda::allocation_alignment, __get().alignment()}};
    }

    template <class _Up2 = _Up>
    [[nodiscard]] _CCCL_HOST_API __buffer_type_for<_Up2> __make_empty_like() const
    {
      return __make_empty_like<_Up2>(size());
    }

    [[nodiscard]] _CCCL_HOST_API __buff_type& __get() noexcept
    {
      return __buf_;
    }

    [[nodiscard]] _CCCL_HOST_API const __buff_type& __get() const noexcept
    {
      return __buf_;
    }

    [[nodiscard]] _CCCL_HOST_API auto begin() noexcept
    {
      return __buf_.begin();
    }

    [[nodiscard]] _CCCL_HOST_API auto end() noexcept
    {
      return __buf_.end();
    }

    [[nodiscard]] _CCCL_HOST_API auto begin() const noexcept
    {
      return __buf_.begin();
    }

    [[nodiscard]] _CCCL_HOST_API auto end() const noexcept
    {
      return __buf_.end();
    }

    [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t size() const noexcept
    {
      return __actual_size_;
    }

    [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t capacity() const noexcept
    {
      return __get().size();
    }

  private:
    __buff_type __buf_;
    __movable_integral<::cuda::std::size_t> __actual_size_{};
  };

  // Persistent per-splitter bracket. rank = global rank of `key`.
  // L: largest key proven to sit BELOW the ideal rank Ni/p.
  // U: smallest key proven to sit ABOVE  the ideal rank Ni/p.
  struct _Bracket
  {
    ::cuda::std::uint64_t __rank; // < global rank of the key
    ::cuda::std::optional<_Tp> __key; // < the key, if found. If nullopt means either +/- inf
  };

  static constexpr double __EPS     = 0.02; // 2% tolerance
  static constexpr auto __ROOT_RANK = 0;

  template <class _InputRange>
  _CCCL_HOST_API static void __sample_probes(
    ::cuda::stream_ref __stream,
    _InputRange&& __input,
    const __buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>& __I_j,
    double __sampling_probability,
    _BinaryOp __cmp,
    __buffer<_Tp>* __samples,
    __buffer<::cuda::std::size_t>* __samples_size_bytes)
  {
    constexpr auto __config =
      ::cuda::make_config(::cuda::make_hierarchy(::cuda::block_dims<1>(), ::cuda::grid_dims<1>()));

    ::cuda::launch(
      __stream,
      __config,
      __sample_probes_kernel<::cuda::std::ranges::iterator_t<_InputRange>,
                             ::cuda::std::ranges::sentinel_t<_InputRange>,
                             _Tp,
                             _BinaryOp>,
      ::cuda::std::bernoulli_distribution{__sampling_probability},
      ::cuda::std::ranges::begin(__input),
      ::cuda::std::ranges::end(__input),
      __I_j.__get(),
      ::cuda::std::move(__cmp),
      __samples->__get().data(),
      __samples_size_bytes->__get().data());
  }

  // TODO(jfaibussowit):
  //
  // Horrifically inefficient!
  _CCCL_HOST_API static void __merge_k_way(
    const communicator& __comm,
    const _Env& __env,
    const __buffer<_Tp>& __data,
    const ::std::vector<::cuda::std::size_t>& __counts_bytes,
    const ::std::vector<::cuda::std::size_t>& __displs_bytes,
    _BinaryOp __cmp,
    __buffer<_Tp>* __ret)
  {
    if (__counts_bytes.size() < 2)
    {
      // TODO(jfaibussowit):
      //
      // Handle properly
      _CCCL_VERIFY(__displs_bytes.empty() || __displs_bytes.front() == 0, "Nonzero displacement for first entry");
      // 0 or 1 inputs, we just copy directly, nothing to merge
      *__ret = __data;
      return;
    }

    const auto __total = (__counts_bytes.back() + __displs_bytes.back()) / sizeof(_Tp);

    __resize_for_overwrite(*__ret, __total);

    auto __tmp_buf                    = __ret->__make_empty_like(__total);
    auto& __ret_buf                   = __ret->__get();
    auto& __data_buf                  = __data.__get();
    ::cuda::std::size_t __merged_size = 0;

    {
      const auto __num_keys1 = __counts_bytes[0] / sizeof(_Tp);
      const auto __num_keys2 = __counts_bytes[1] / sizeof(_Tp);
      const auto __off1      = __displs_bytes[0] / sizeof(_Tp);
      const auto __off2      = __displs_bytes[1] / sizeof(_Tp);

      __CUDAX_MULTI_GPU_DOUBLE_DISPATCH(
        __comm.device(),
        __num_keys1,
        __num_keys2,
        ::cub::DeviceMerge::MergeKeys,
        (__data_buf.begin() + __off1,
         __num_keys1_fixed,
         __data_buf.begin() + __off2,
         __num_keys2_fixed,
         __ret_buf.begin(),
         __cmp,
         __env));

      __merged_size = __num_keys1 + __num_keys2;
    }

    for (::cuda::std::size_t __i = 2; __i < __displs_bytes.size(); ++__i)
    {
      const auto __num_keys1 = __merged_size;
      const auto __num_keys2 = __counts_bytes[__i] / sizeof(_Tp);
      const auto __off       = __displs_bytes[__i] / sizeof(_Tp);

      __CUDAX_MULTI_GPU_DOUBLE_DISPATCH(
        __comm.device(),
        __num_keys1,
        __num_keys2,
        ::cub::DeviceMerge::MergeKeys,
        (__ret_buf.begin(),
         __num_keys1_fixed,
         __data_buf.begin() + __off,
         __num_keys2_fixed,
         __tmp_buf.begin(),
         __cmp,
         __env));

      __ret_buf.swap(__tmp_buf);
      __merged_size = __num_keys1 + __num_keys2;
    }
  }

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __gather_merge_broadcast(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__buffer<_Tp>>& __local_samples,
    _BinaryOp __cmp,
    ::std::vector<__buffer<::cuda::std::size_t>>* __local_samples_size_bytes,
    ::std::vector<__buffer<_Tp>>* __local_probes)
  {
    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __samples_size_bytes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, *__local_samples_size_bytes))
      {
        auto* const __ptr = __samples_size_bytes.__get().data();

        __nccl::__ncclGather(__ptr, __ptr, sizeof(*__ptr), __nccl::__ncclChar, __ROOT_RANK, __comm.comm(), __stream);
      }
    }

    ::std::vector<::cuda::std::size_t> __sendcounts_bytes(::cuda::std::ranges::begin(__comms)->size());
    ::std::vector<::cuda::std::size_t> __root_recvcounts_bytes;
    ::std::vector<::cuda::std::size_t> __root_displs_bytes;
    ::cuda::std::optional<__buffer<_Tp>> __root_all_samples;

    for (auto&& [__comm, __stream, __env, __samples_size_bytes, __sendcount] :
         ::cuda::std::ranges::views::zip(__comms, __streams, __envs, *__local_samples_size_bytes, __sendcounts_bytes))
    {
      if (__comm.rank() == __ROOT_RANK)
      {
        __root_recvcounts_bytes.resize(__samples_size_bytes.size());

        ::cuda::__driver::__memcpyAsync(
          __root_recvcounts_bytes.data(),
          __samples_size_bytes.__get().data(),
          __samples_size_bytes.size() * sizeof(*__samples_size_bytes.__get().data()),
          __stream.get());

        __root_displs_bytes.reserve(__root_recvcounts_bytes.size() + 1);

        // Defer until the last possible moment
        __stream.sync();

        // __root_recv_counts[__ROOT_RANK] is exactly the root's send counts as
        // well
        __sendcount = __root_recvcounts_bytes[__ROOT_RANK];

        // recvcounts is likely relatively small (it is on the order of
        // O(ranks)). It may be faster to just do the exclusive scan on the host
        // than to copy to device, scan, and copy back
        ::cuda::std::exclusive_scan(
          __root_recvcounts_bytes.begin(),
          __root_recvcounts_bytes.end(),
          ::cuda::std::back_inserter(__root_displs_bytes),
          ::cuda::std::size_t{0});
        // displs.back() is the LAST rank's offset (exclusive scan), so total
        // bytes = displs.back() + __root_local_samples_size.back(). all_samples
        // is vector<T>, so size it in ELEMENTS, not bytes.
        const auto __all_recv = (__root_displs_bytes.back() + __root_recvcounts_bytes.back()) / sizeof(_Tp);

        __root_all_samples.emplace(
          __stream, __samples_size_bytes.__get().memory_resource(), __all_recv, ::cuda::no_init, __env);
      }
      else
      {
        // Non-root, __samples_size_bytes.size() should be == 1, but just pass
        // sizeof(dest) to be safe
        ::cuda::__driver::__memcpyAsync(
          &__sendcount, __samples_size_bytes.__get().data(), sizeof(__sendcount), __stream.get());
        __stream.sync();
      }
    }

    // ---- 3. Variable-count gather of the actual sample keys ----
    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __samples, __sendcount_bytes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_samples, __sendcounts_bytes))
      {
        const auto __rank = __comm.rank();

        __nccl::__ncclGatherv(
          __samples.__get().data(),
          __sendcount_bytes,
          __rank == __ROOT_RANK ? __root_all_samples.value().__get().data() : nullptr,
          __nccl::__ncclChar,
          __root_recvcounts_bytes.data(),
          __root_displs_bytes.data(),
          __ROOT_RANK,
          __comm.comm(),
          __stream);
      }
    }

    // ---- 4. Root merges the p sorted runs into one sorted probe set ----
    auto __probe_count = ::cuda::make_pinned_buffer<::cuda::std::uint64_t>(
      ::cuda::stream_ref{::cudaStream_t{}}, /*__size=*/::cuda::std::size_t{1}, ::cuda::no_init);

    for (auto&& [__comm, __env, __probes] : ::cuda::std::ranges::views::zip(__comms, __envs, *__local_probes))
    {
      if (__comm.rank() == __ROOT_RANK)
      {
        __merge_k_way(
          __comm,
          __env,
          __root_all_samples.value(),
          __root_recvcounts_bytes,
          __root_displs_bytes,
          ::cuda::std::move(__cmp),
          &__probes);

        __probe_count.front() = __probes.size();
        break;
      }
    }

    // Extremely painful stuff here. We need to send the probe count, but we can
    // only use NCCL, and NCCL only provides device transport (even though they
    // definitely have host-host transport available internally).
    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream] : ::cuda::std::ranges::views::zip(__comms, __streams))
      {
        auto* const __ptr = __probe_count.data();

        __nccl::__ncclBroadcast(__ptr, __ptr, sizeof(*__ptr), __nccl::__ncclChar, __ROOT_RANK, __comm.comm(), __stream);
      }
    }

    for (auto&& [__comm, __stream, __probes] : ::cuda::std::ranges::views::zip(__comms, __streams, *__local_probes))
    {
      if (__comm.rank() != __ROOT_RANK)
      {
        // wait for comm
        __stream.sync();
        __resize_for_overwrite(__probes, __probe_count.front());
      }
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __probes] : ::cuda::std::ranges::views::zip(__comms, __streams, *__local_probes))
      {
        auto* const __ptr = __probes.__get().data();

        __nccl::__ncclBroadcast(
          __ptr,
          __ptr,
          __probe_count.front() * sizeof(*__ptr),
          __nccl::__ncclChar,
          __ROOT_RANK,
          __comm.comm(),
          __stream);
      }
    }
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __local_histogram(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __range_of_local_keys,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    const _BinaryOp& __cmp,
    ::std::vector<__buffer<::cuda::std::uint64_t>>* __local_hist)
  {
    for (auto&& [__comm, __env, __keys, __probes, __hist] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_local_keys, __local_probes, *__local_hist))
    {
      const auto __num_probes  = __probes.size();
      const auto __num_buckets = __num_probes + 1;

      const auto __keys_first    = ::cuda::std::ranges::begin(__keys);
      const auto __keys_last     = ::cuda::std::ranges::end(__keys);
      const auto* __probes_first = __probes.__get().data();

      __resize_for_overwrite(__hist, __num_buckets);

      constexpr auto __in = ::cuda::counting_iterator<::cuda::std::uint64_t>{};
      auto __op           = [__lo = __keys_first, __keys_last, __probes_first, __cmp, __num_probes]
        _CCCL_HOST_DEVICE(::cuda::std::uint64_t __bucket) mutable {
          auto __hi = __keys_last;
          // This reordering takes advantage of the fact that we cached __lo
          // previously. In that case, __lo will already have raised the lower
          // bound of the search. We now just need to raise the *upper* bound of
          // the search. We cannot do that generally by caching (because the
          // operators are called from left to right), but we *can* bound the
          // search for __lo.
          if (__bucket != __num_probes)
          {
            __hi = ::cuda::std::lower_bound(__lo, __keys_last, __probes_first[__bucket], __cmp);
          }

          if (__bucket != 0)
          {
            __lo = ::cuda::std::lower_bound(__lo, __hi, __probes_first[__bucket - 1], __cmp);
          }

          const auto __ret = __hi - __lo;
          // This caching of __lo relies on the fact that CUB calls these
          // operators with monotonically increasing bucket count (i.e. the stride
          // is always adding blockIdx.x). If that doesn't happen, then this
          // caching is wrong.
          __lo = __hi;
          return __ret;
        };

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __num_buckets,
        ::cub::DeviceTransform::Transform,
        (__in, __hist.begin(), __num_buckets_fixed, ::cuda::std::move(__op), __env));
    }
  }

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __update_intervals(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    const ::std::vector<__buffer<::cuda::std::uint64_t>>& __local_hist,
    ::cuda::std::uint64_t __N,
    ::std::vector<__buffer<_Bracket>>* __local_Ls,
    ::std::vector<__buffer<_Bracket>>* __local_Us,
    ::std::vector<__buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>>* __local_I_js)
  {
    for (auto&& [__comm, __env, __probes, __histogram, __Ls, __Us, __I_j] : ::cuda::std::ranges::views::zip(
           __comms, __envs, __local_probes, __local_hist, *__local_Ls, *__local_Us, *__local_I_js))
    {
      const auto __comm_size     = __comm.size();
      const auto __num_splitters = __I_j.size();
      const auto __I_j_begin     = __I_j.begin();
      const auto __Ls_begin      = __Ls.begin();
      const auto __Us_begin      = __Us.begin();

      auto __in = ::cuda::make_zip_iterator(
        ::cuda::make_transform_iterator(
          ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
        __Ls_begin,
        __Us_begin);
      auto __out = ::cuda::make_zip_iterator(__I_j_begin, __Ls_begin, __Us_begin);
      auto __op =
        __update_intervals_fn<_Tp, _Bracket>{__probes.__get().data(), __histogram.__get().data(), __probes.size()};

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __num_splitters,
        ::cub::DeviceTransform::Transform,
        (::cuda::std::move(__in), ::cuda::std::move(__out), __num_splitters_fixed, ::cuda::std::move(__op), __env));
    }
  }

  // Returns the p-1 partition keys per local communicator. Data exchange uses
  // them to route records; the post-exchange distribution is measured later in
  // __rebalance_to_original_counts (one size all-gather), mirroring the
  // reference, rather than predicted here.
  template <class _CommRange, class _EnvRange>
  [[nodiscard]] static ::std::vector<__buffer<_Tp>> __finalize_splitters(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<__buffer<_Bracket>>& __local_Ls,
    const ::std::vector<__buffer<_Bracket>>& __local_Us,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    ::cuda::std::uint64_t __N)
  {
    ::std::vector<__buffer<_Tp>> __ret;

    __ret.reserve(__local_Ls.size());
    for (auto&& [__comm, __env, __Ls, __Us, __probes] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __local_Ls, __local_Us, __local_probes))
    {
      auto& __splitters = __ret.emplace_back(__Ls.template __make_empty_like<_Tp>());

      const auto __comm_size = __comm.size();
      const auto __num_items = __Ls.size();
      auto __in              = ::cuda::make_zip_iterator(
        ::cuda::make_transform_iterator(
          ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
        __Ls.begin(),
        __Us.begin());
      auto __out           = __splitters.begin();
      using __probe_iter_t = ::cuda::std::remove_cvref_t<decltype(__probes.begin())>;
      auto __op            = __finalize_splitters_fn<_Tp, __probe_iter_t>{__probes.begin(), __probes.end() - 1};

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __num_items,
        ::cub::DeviceTransform::Transform,
        (::cuda::std::move(__in), ::cuda::std::move(__out), __num_items_fixed, ::cuda::std::move(__op), __env));
    }

    return __ret;
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __data_exchange(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__resource_type>& __resources,
    ::std::vector<__buffer<_Tp>> __local_splitters,
    _BinaryOp __cmp,
    _InputRange&& __local_inputs)
  {
    ::std::vector<__buffer<::cuda::std::size_t>> __local_send_counts_bytes;
    ::std::vector<__buffer<::cuda::std::size_t>> __local_send_displs_bytes;
    ::std::vector<__buffer<::cuda::std::size_t>> __local_recv_counts_bytes;

    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs_bytes;

    ::std::vector<__buffer<_Tp>> __local_recvd;

    const auto __comm_size        = ::cuda::std::ranges::begin(__comms)->size();
    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    __local_send_counts_bytes.reserve(__num_local_inputs);
    __local_send_displs_bytes.reserve(__num_local_inputs);
    __local_recv_counts_bytes.reserve(__num_local_inputs);
    __local_h_send_counts_bytes.reserve(__num_local_inputs);
    __local_h_send_displs_bytes.reserve(__num_local_inputs);
    __local_h_recv_counts_bytes.reserve(__num_local_inputs);
    __local_h_recv_displs_bytes.reserve(__num_local_inputs);
    __local_recvd.reserve(__num_local_inputs);

    for (auto&& [__comm, __env, __stream, __resource, __input, __splitters] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __streams, __resources, __local_inputs, __local_splitters))
    {
      auto& __send_counts_bytes =
        __local_send_counts_bytes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      auto& __send_displs_bytes =
        __local_send_displs_bytes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

      auto __out = ::cuda::make_zip_iterator(__send_counts_bytes.begin(), __send_displs_bytes.begin());

      const auto __input_begin = ::cuda::std::ranges::begin(__input);

      // TODO(jfaibussowit):
      //
      // This kernel could actually be fused with the splitters finalization one
      // I am fairly sure.
      auto __op =
        [__input_begin,
         __input_end       = ::cuda::std::ranges::end(__input),
         __splitters_begin = __splitters.__get().data(),
         __num_splitters   = __splitters.size(),
         __lo              = __input_begin,
         __cmp] _CCCL_HOST_DEVICE(::cuda::std::size_t __d) mutable {
          auto __hi = __input_end;

          if (__d != __num_splitters)
          {
            __hi = ::cuda::std::lower_bound(__lo, __input_end, __splitters_begin[__d], __cmp);
          }

          if (__d != 0)
          {
            __lo = ::cuda::std::lower_bound(__lo, __hi, __splitters_begin[__d - 1], __cmp);
          }

          const auto __displ_bytes = (__lo - __input_begin) * sizeof(_Tp);
          const auto __count_bytes = (__hi - __lo) * sizeof(_Tp);

          __lo = __hi;
          return ::cuda::std::tuple{__count_bytes, __displ_bytes};
        };

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __comm_size,
        ::cub::DeviceTransform::Transform,
        (::cuda::counting_iterator<::cuda::std::size_t>{},
         ::cuda::std::move(__out),
         __comm_size_fixed,
         ::cuda::std::move(__op),
         __env));
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __send_counts_bytes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_send_counts_bytes))
      {
        auto& __recv_counts_bytes = __local_recv_counts_bytes.emplace_back(__send_counts_bytes.__make_empty_like());
        auto* const __send_ptr    = __send_counts_bytes.__get().data();
        auto* const __recv_ptr    = __recv_counts_bytes.__get().data();

        __nccl::__ncclAlltoAll(__send_ptr, __recv_ptr, sizeof(*__send_ptr), __nccl::__ncclChar, __comm.comm(), __stream);
      }
    }

    for (auto&& [__comm, __stream, __resource, __env, __send_counts_bytes, __send_displs_bytes, __recv_counts_bytes] :
         ::cuda::std::ranges::views::zip(
           __comms,
           __streams,
           __resources,
           __envs,
           __local_send_counts_bytes,
           __local_send_displs_bytes,
           __local_recv_counts_bytes))
    {
      const auto __num_items   = __recv_counts_bytes.size();
      auto __recv_displs_bytes = __recv_counts_bytes.__make_empty_like();

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __num_items,
        ::cub::DeviceScan::ExclusiveSum,
        (__recv_counts_bytes.begin(), __recv_displs_bytes.begin(), __num_items_fixed, __env));

      auto& __h_send_counts_bytes = __local_h_send_counts_bytes.emplace_back(__send_counts_bytes.size());
      auto& __h_send_displs_bytes = __local_h_send_displs_bytes.emplace_back(__send_displs_bytes.size());
      auto& __h_recv_counts_bytes = __local_h_recv_counts_bytes.emplace_back(__recv_counts_bytes.size());
      auto& __h_recv_displs_bytes = __local_h_recv_displs_bytes.emplace_back(__recv_displs_bytes.size());

      ::cuda::__driver::__memcpyAsync(
        __h_send_counts_bytes.data(),
        __send_counts_bytes.__get().data(),
        __h_send_counts_bytes.size() * sizeof(*__h_send_counts_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_send_displs_bytes.data(),
        __send_displs_bytes.__get().data(),
        __h_send_displs_bytes.size() * sizeof(*__h_send_displs_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_recv_counts_bytes.data(),
        __recv_counts_bytes.__get().data(),
        __h_recv_counts_bytes.size() * sizeof(*__h_recv_counts_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_recv_displs_bytes.data(),
        __recv_displs_bytes.data(),
        __h_recv_displs_bytes.size() * sizeof(*__h_recv_displs_bytes.data()),
        __stream.get());

      __stream.sync();

      const auto __total_recv = (__h_recv_displs_bytes.back() + __h_recv_counts_bytes.back()) / sizeof(_Tp);

      __local_recvd.emplace_back(__stream, __resource, __total_recv, ::cuda::no_init, __env);
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm,
                   __stream,
                   __input,
                   __recvd,
                   __h_send_counts_bytes,
                   __h_send_displs_bytes,
                   __h_recv_counts_bytes,
                   __h_recv_displs_bytes] :
           ::cuda::std::ranges::views::zip(
             __comms,
             __streams,
             __local_inputs,
             __local_recvd,
             __local_h_send_counts_bytes,
             __local_h_send_displs_bytes,
             __local_h_recv_counts_bytes,
             __local_h_recv_displs_bytes))
      {
        __nccl::__ncclAlltoAllv(
          ::cuda::std::to_address(&*::cuda::std::ranges::begin(__input)),
          __h_send_counts_bytes.data(),
          __h_send_displs_bytes.data(),
          __recvd.__get().data(),
          __h_recv_counts_bytes.data(),
          __h_recv_displs_bytes.data(),
          __nccl::__ncclChar,
          __comm.comm(),
          __stream);
      }
    }

    // --- Phase C: merge the p received sorted runs into the final local output
    // ---
    for (auto&& [__comm, __env, __stream, __recvd, __h_recv_counts_bytes, __h_recv_displs_bytes, __inputs] :
         ::cuda::std::ranges::views::zip(
           __comms,
           __envs,
           __streams,
           __local_recvd,
           __local_h_recv_counts_bytes,
           __local_h_recv_displs_bytes,
           __local_inputs))
    {
      // TODO(jfaibussowit):
      //
      // Don't use __tmp and instead write directly to __inputs
      auto __tmp = __buffer<_Tp>{__recvd.__make_empty_like(0)};

      __merge_k_way(__comm, __env, __recvd, __h_recv_counts_bytes, __h_recv_displs_bytes, __cmp, &__tmp);

      __resize_for_overwrite(__inputs, __tmp.size());

      const auto* const __src = __tmp.__get().data();
      auto* const __dst       = ::cuda::std::to_address(&*::cuda::std::ranges::begin(__inputs));

      ::cuda::__driver::__memcpyAsync(__dst, __src, __tmp.size() * sizeof(*__src), __stream.get());
    }
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __rebalance_to_original_counts(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__resource_type>& __resources,
    const ::std::vector<::cuda::std::size_t>& __local_original_sizes,
    const ::std::vector<__buffer<::cuda::std::uint64_t>>& __local_desired_offsets,
    ::cuda::std::uint64_t __N,
    _InputRange&& __local_inputs)
  {
    // The splitter exchange already produced a globally sorted, approximately
    // balanced distribution. Rebalance only corrects the rank ranges from that
    // current distribution back to the original per-rank sizes. Desired offsets
    // are the exclusive scan of the original sizes; the CURRENT offsets are
    // measured here -- the actual post-exchange per-rank sizes, all-gathered
    // and exclusive-scanned -- exactly as the reference does. (Duplicate
    // splitter keys can route an unpredictable share of records to a single
    // rank, so the current distribution must be measured, not predicted from
    // the splitter positions.)
    const auto __comm_size = ::cuda::std::ranges::begin(__comms)->size();

    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts_bytes;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs_bytes;

    ::std::vector<__buffer<_Tp>> __local_rebalanced;

    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    __local_h_send_counts_bytes.reserve(__num_local_inputs);
    __local_h_send_displs_bytes.reserve(__num_local_inputs);
    __local_h_recv_counts_bytes.reserve(__num_local_inputs);
    __local_h_recv_displs_bytes.reserve(__num_local_inputs);
    __local_rebalanced.reserve(__num_local_inputs);

    // ---- Measure the realized post-exchange distribution: all-gather each
    // rank's current __input size, then exclusive-scan into current offsets
    // (length __comm_size, current_offsets[0] == 0).
    ::std::vector<__buffer<::cuda::std::uint64_t>> __local_current_sizes;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __local_current_offsets;
    __local_current_sizes.reserve(__num_local_inputs);
    __local_current_offsets.reserve(__num_local_inputs);

    for (auto&& [__comm, __env, __stream, __resource, __input] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __streams, __resources, __local_inputs))
    {
      const auto __n_current = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));
      auto& __sizes = __local_current_sizes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      ::cuda::__driver::__memcpyAsync(
        __sizes.__get().data() + __comm.rank(), &__n_current, sizeof(__n_current), __stream.get());
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __sizes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_current_sizes))
      {
        auto* const __ptr = __sizes.__get().data();

        __nccl::__ncclAllGather(
          __ptr + __comm.rank(), __ptr, sizeof(*__ptr), __nccl::__ncclChar, __comm.comm(), __stream);
      }
    }

    for (auto&& [__comm, __env, __stream, __resource, __sizes] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __streams, __resources, __local_current_sizes))
    {
      auto& __offsets = __local_current_offsets.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __comm_size,
        ::cub::DeviceScan::ExclusiveSum,
        (__sizes.begin(), __offsets.begin(), __comm_size_fixed, __env));
    }

    for (auto&& [__comm, __env, __stream, __resource, __current_offsets, __desired_offsets, __original_size] :
         ::cuda::std::ranges::views::zip(
           __comms,
           __envs,
           __streams,
           __resources,
           __local_current_offsets,
           __local_desired_offsets,
           __local_original_sizes))
    {
      auto __send_counts_bytes =
        ::cuda::make_buffer<::cuda::std::size_t>(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      auto __send_displs_bytes =
        ::cuda::make_buffer<::cuda::std::size_t>(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      auto __recv_counts_bytes =
        ::cuda::make_buffer<::cuda::std::size_t>(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      auto __recv_displs_bytes =
        ::cuda::make_buffer<::cuda::std::size_t>(__stream, __resource, __comm_size, ::cuda::no_init, __env);

      auto __out = ::cuda::make_zip_iterator(
        __send_counts_bytes.begin(),
        __send_displs_bytes.begin(),
        __recv_counts_bytes.begin(),
        __recv_displs_bytes.begin());

      auto __op =
        [__rank = __comm.rank(),
         __comm_size,
         __N,
         __current_offsets = __current_offsets.__get().data(),
         __desired_offsets = __desired_offsets.__get().data()] _CCCL_HOST_DEVICE(::cuda::std::size_t __peer) {
          // current_offsets[i] is the start of rank i's current
          // (post-exchange) bucket; desired offsets are the original final
          // buckets. Intersect the two global element intervals to derive
          // both send and receive metadata directly.
          const auto __my_src_begin = __current_offsets[__rank];
          const auto __my_src_end   = __rank + 1 == __comm_size ? __N : __current_offsets[__rank + 1];

          const auto __peer_dst_begin = __desired_offsets[__peer];
          const auto __peer_dst_end   = __peer + 1 == __comm_size ? __N : __desired_offsets[__peer + 1];

          const auto __send_begin = ::cuda::std::max(__my_src_begin, __peer_dst_begin);
          const auto __send_end   = ::cuda::std::min(__my_src_end, __peer_dst_end);

          const auto __my_dst_begin = __desired_offsets[__rank];
          const auto __my_dst_end   = __rank + 1 == __comm_size ? __N : __desired_offsets[__rank + 1];

          const auto __peer_src_begin = __current_offsets[__peer];
          const auto __peer_src_end   = __peer + 1 == __comm_size ? __N : __current_offsets[__peer + 1];

          const auto __recv_begin = ::cuda::std::max(__peer_src_begin, __my_dst_begin);
          const auto __recv_end   = ::cuda::std::min(__peer_src_end, __my_dst_end);

          const auto __send_count =
            __send_begin < __send_end
              ? static_cast<::cuda::std::size_t>((__send_end - __send_begin) * sizeof(_Tp))
              : ::cuda::std::size_t{0};
          const auto __recv_count =
            __recv_begin < __recv_end
              ? static_cast<::cuda::std::size_t>((__recv_end - __recv_begin) * sizeof(_Tp))
              : ::cuda::std::size_t{0};

          return ::cuda::std::tuple{
            __send_count,
            __send_count == 0 ? ::cuda::std::size_t{0}
                              : static_cast<::cuda::std::size_t>((__send_begin - __my_src_begin) * sizeof(_Tp)),
            __recv_count,
            __recv_count == 0 ? ::cuda::std::size_t{0}
                              : static_cast<::cuda::std::size_t>((__recv_begin - __my_dst_begin) * sizeof(_Tp))};
        };

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.device(),
        __comm_size,
        ::cub::DeviceTransform::Transform,
        (::cuda::counting_iterator<::cuda::std::size_t>{},
         ::cuda::std::move(__out),
         __comm_size_fixed,
         ::cuda::std::move(__op),
         __env));
      auto& __h_send_counts_bytes = __local_h_send_counts_bytes.emplace_back(__send_counts_bytes.size());
      auto& __h_send_displs_bytes = __local_h_send_displs_bytes.emplace_back(__send_displs_bytes.size());
      auto& __h_recv_counts_bytes = __local_h_recv_counts_bytes.emplace_back(__recv_counts_bytes.size());
      auto& __h_recv_displs_bytes = __local_h_recv_displs_bytes.emplace_back(__recv_displs_bytes.size());

      ::cuda::__driver::__memcpyAsync(
        __h_send_counts_bytes.data(),
        __send_counts_bytes.data(),
        __h_send_counts_bytes.size() * sizeof(*__h_send_counts_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_send_displs_bytes.data(),
        __send_displs_bytes.data(),
        __h_send_displs_bytes.size() * sizeof(*__h_send_displs_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_recv_counts_bytes.data(),
        __recv_counts_bytes.data(),
        __h_recv_counts_bytes.size() * sizeof(*__h_recv_counts_bytes.data()),
        __stream.get());
      ::cuda::__driver::__memcpyAsync(
        __h_recv_displs_bytes.data(),
        __recv_displs_bytes.data(),
        __h_recv_displs_bytes.size() * sizeof(*__h_recv_displs_bytes.data()),
        __stream.get());

      __local_rebalanced.emplace_back(__stream, __resource, __original_size, ::cuda::no_init, __env);
    }

    // Sync for HtoD transfers
    for (auto __stream : __streams)
    {
      __stream.sync();
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      // The rebalance exchange is the only communication in this phase. It
      // moves already globally sorted contiguous rank intervals into the exact
      // original per-rank sizes.
      for (auto&& [__comm,
                   __stream,
                   __input,
                   __out,
                   __h_send_counts_bytes,
                   __h_send_displs_bytes,
                   __h_recv_counts_bytes,
                   __h_recv_displs_bytes] :
           ::cuda::std::ranges::views::zip(
             __comms,
             __streams,
             __local_inputs,
             __local_rebalanced,
             __local_h_send_counts_bytes,
             __local_h_send_displs_bytes,
             __local_h_recv_counts_bytes,
             __local_h_recv_displs_bytes))
      {
        __nccl::__ncclAlltoAllv(
          ::cuda::std::to_address(&*::cuda::std::ranges::begin(__input)),
          __h_send_counts_bytes.data(),
          __h_send_displs_bytes.data(),
          __out.__get().data(),
          __h_recv_counts_bytes.data(),
          __h_recv_displs_bytes.data(),
          __nccl::__ncclChar,
          __comm.comm(),
          __stream);
      }
    }

    for (auto&& [__stream, __input, __out] :
         ::cuda::std::ranges::views::zip(__streams, __local_inputs, __local_rebalanced))
    {
      // This resize is safe only so long as the user promises to free their allocation on the
      // stream that they passed us. For thrust/cuda containers, this is vacuously true
      __resize_for_overwrite(__input, __out.size());

      const auto* __src  = __out.__get().data();
      auto* const __dest = ::cuda::std::to_address(&*::cuda::std::ranges::begin(__input));

      ::cuda::__driver::__memcpyAsync(__dest, __src, __out.size() * sizeof(*__src), __stream.get());
    }
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void
  __execute(_CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __local_inputs, _BinaryOp __cmp)
  {
    // We take number of comms to be the local input
    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    if (!__num_local_inputs)
    {
      // We have no inputs, so... nothing to do
      return;
    }

    const auto __comm_size = static_cast<::cuda::std::size_t>(::cuda::std::ranges::begin(__comms)->size());

    ::std::vector<::cuda::stream_ref> __streams;
    ::std::vector<__resource_type> __resources;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __all_local_sizes;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __all_local_offsets;
    ::std::vector<::cuda::std::size_t> __local_original_sizes;

    // TODO (jfaibussowit): maybe can combine some of these
    __streams.reserve(__num_local_inputs);
    __resources.reserve(__num_local_inputs);
    __all_local_sizes.reserve(__num_local_inputs);
    __all_local_offsets.reserve(__num_local_inputs);
    __local_original_sizes.reserve(__num_local_inputs);

    for (auto&& [__comm, __env, __input] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_inputs))
    {
      const auto __n_local = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));
      const auto __stream  = __streams.emplace_back(__detail::__stream_from_env(__env));
      auto& __resource     = __resources.emplace_back(__detail::__resource_from_env(__env, __comm.device()));
      auto& __sizes = __all_local_sizes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env).__get();

      __local_original_sizes.push_back(__n_local);
      ::cuda::__driver::__memcpyAsync(__sizes.data() + __comm.rank(), &__n_local, sizeof(__n_local), __stream.get());
    }

    {
      const auto _ = __nccl::__auto_nccl_group{};

      for (auto&& [__comm, __stream, __sizes] : ::cuda::std::ranges::views::zip(__comms, __streams, __all_local_sizes))
      {
        auto* const __ptr = __sizes.__get().data();

        __nccl::__ncclAllGather(
          __ptr + __comm.rank(), __ptr, sizeof(*__ptr), __nccl::__ncclChar, __comm.comm(), __stream);
      }
    }

    ::cuda::std::uint64_t __N = 0;

    {
      bool __N_computed = false;

      for (auto&& [__comm, __stream, __resource, __env, __input, __sizes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __resources, __envs, __local_inputs, __all_local_sizes))
      {
        auto& __offsets = __all_local_offsets.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

        __CUDAX_MULTI_GPU_DISPATCH(
          __comm.device(),
          __comm_size,
          ::cub::DeviceScan::ExclusiveSum,
          (__sizes.begin(), __offsets.begin(), __comm_size_fixed, __env));

        if (!__N_computed)
        {
          ::cuda::std::uint64_t __last_offset = 0;
          ::cuda::std::uint64_t __last_size   = 0;

          // The desired-offset scan already encodes the global extent: N =
          // offset[p - 1] + size[p - 1].
          ::cuda::__driver::__memcpyAsync(
            &__last_offset, __offsets.__get().data() + (__comm_size - 1), sizeof(__last_offset), __stream.get());
          ::cuda::__driver::__memcpyAsync(
            &__last_size, __sizes.__get().data() + (__comm_size - 1), sizeof(__last_size), __stream.get());

          __stream.sync();
          __N          = __last_offset + __last_size;
          __N_computed = true;
        }

        {
          const auto __num_items = ::cuda::std::ranges::size(__input);

          __CUDAX_MULTI_GPU_DISPATCH(
            __comm.device(),
            __num_items,
            ::cub::DeviceMergeSort::SortKeys,
            (::cuda::std::ranges::begin(__input), __num_items_fixed, __cmp, __env));
        }
      }
    }

    if (__comm_size == 1 || __N == 0)
    {
      return;
    }

    ::std::vector<__buffer<_Bracket>> __local_Ls;
    ::std::vector<__buffer<_Bracket>> __local_Us;
    ::std::vector<__buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>> __local_I_js;
    ::std::vector<__buffer<_Tp>> __local_probes;
    ::std::vector<__buffer<_Tp>> __local_samples;
    ::std::vector<__buffer<::cuda::std::size_t>> __local_samples_sizes_bytes;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __local_hist;

    __local_Ls.reserve(__num_local_inputs);
    __local_Us.reserve(__num_local_inputs);
    __local_I_js.reserve(__num_local_inputs);
    __local_probes.reserve(__num_local_inputs);
    __local_samples.reserve(__num_local_inputs);
    __local_samples_sizes_bytes.reserve(__num_local_inputs);
    __local_hist.reserve(__num_local_inputs);

    for (auto&& [__comm, __stream, __resource, __env] :
         ::cuda::std::ranges::views::zip(__comms, __streams, __resources, __envs))
    {
      const auto __n_split = __comm_size - 1;
      // TODO(jfaibussowit): these can all be done in parallel on separate
      // utility streams.
      __local_Ls.emplace_back(__stream, __resource, __n_split, _Bracket{0, ::cuda::std::nullopt}, __env);
      __local_Us.emplace_back(__stream, __resource, __n_split, _Bracket{__N, ::cuda::std::nullopt}, __env);
      __local_I_js.emplace_back(
        __stream,
        __resource,
        __n_split,
        ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>{},
        __env);
      __local_probes.emplace_back(__stream, __resource, __env);
      __local_samples.emplace_back(__stream, __resource, __env);
      // differing sizes for NCCL inplace
      __local_samples_sizes_bytes.emplace_back(
        __stream,
        __resource,
        /*__size=*/__comm.rank() == __ROOT_RANK ? __comm_size : 1,
        ::cuda::no_init,
        __env);
      __local_hist.emplace_back(__stream, __resource, __env);
    }

    const auto __K = ::cuda::std::max(
      static_cast<::cuda::std::int32_t>(::cuda::std::ceil(::cuda::std::log10(::cuda::std::log10(__comm_size) / __EPS))),
      1);
    const auto __s_j_interior = 2. * ::cuda::std::log(__comm_size) / __EPS;
    // Lemma 4.6: the OVERALL sample size for round j is
    //
    // Z_j <= 5*p*s_j/s_{j-1} w.h.p.,
    //
    // and with the schedule
    //
    // s_j = (2 ln p / eps)^{j/k}
    //
    // the ratio
    //
    // s_j/s_{j-1} = (2 ln p / eps)^{1/k}
    //
    // is constant across rounds. So the GLOBAL sample size is bounded (w.h.p.) by
    //
    // Z_j = 5*p*(2 ln p/eps)^{1/k}.
    //
    // This is only a w.h.p. bound on a Bernoulli draw, not a hard cap, so it is
    // used as a reserve() hint -- an unlucky overshoot just costs a realloc.
    const auto __Z_j_per_proc = static_cast<::cuda::std::size_t>(::cuda::std::ceil(
      5. * ::cuda::std::pow(2. * ::cuda::std::log(__comm_size) / __EPS, 1.0 / static_cast<double>(__K))));

    for (int __j = 1; __j <= __K; ++__j)
    {
      const auto __s_j  = ::cuda::std::pow(__s_j_interior, static_cast<double>(__j) / static_cast<double>(__K));
      const auto __prob = ::cuda::std::min(__s_j * static_cast<double>(__comm_size) / static_cast<double>(__N), 1.);

      for (auto&& [__stream, __input, __I_j, __samples, __sample_sizes_bytes] : ::cuda::std::ranges::views::zip(
             __streams, __local_inputs, __local_I_js, __local_samples, __local_samples_sizes_bytes))
      {
        __resize_for_overwrite(__samples, __Z_j_per_proc);
        __sample_probes(__stream, __input, __I_j, __prob, __cmp, &__samples, &__sample_sizes_bytes);
      }

      __gather_merge_broadcast(
        __comms, __envs, __streams, __local_samples, __cmp, &__local_samples_sizes_bytes, &__local_probes);

      __local_histogram(__comms, __envs, __local_inputs, __local_probes, __cmp, &__local_hist);

      {
        const auto _ = __nccl::__auto_nccl_group{};

        for (auto&& [__comm, __stream, __hist] : ::cuda::std::ranges::views::zip(__comms, __streams, __local_hist))
        {
          auto* const __ptr = __hist.__get().data();

          __nccl::__ncclAllReduce(
            __ptr,
            __ptr,
            __hist.size(),
            __nccl::__nccl_type_of_v<decltype(*__ptr)>,
            __nccl::__ncclSum,
            __comm.comm(),
            __stream);
        }
      }

      // ---- STEP 3: tighten brackets and rebuild intervals ----
      __update_intervals(__comms, __envs, __local_probes, __local_hist, __N, &__local_Ls, &__local_Us, &__local_I_js);
    }
    // Free some temporary buffers
    __local_samples.clear();
    __local_samples_sizes_bytes.clear();
    __local_I_js.clear();
    __local_hist.clear();

    auto __splitters = __finalize_splitters(__comms, __envs, __local_Ls, __local_Us, __local_probes, __N);

    __data_exchange(__comms, __envs, __streams, __resources, ::cuda::std::move(__splitters), __cmp, __local_inputs);

    __rebalance_to_original_counts(
      __comms, __envs, __streams, __resources, __local_original_sizes, __all_local_offsets, __N, __local_inputs);
  }
};
} // namespace __detail

_CCCL_TEMPLATE(class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
               ::cuda::std::ranges::input_range<_EnvRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange>)
void sort(_CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __range_of_input_ranges, _BinaryOp __cmp = {})
{
  __validate_input_range<_InputRange>();

  using __result_type =
    ::cuda::std::ranges::range_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>>;
  using __env_type = ::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_EnvRange>>;

  __detail::_Sorter<__result_type, __env_type, _BinaryOp>{}.__execute(
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::move(__cmp));
}

_CCCL_TEMPLATE(class _CommRange, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange>)
void sort(_CommRange&& __comms, _InputRange&& __range_of_input_ranges, _BinaryOp __cmp = {})
{
  sort(::cuda::std::forward<_CommRange>(__comms),
       ::cuda::std::ranges::views::repeat(::cuda::std::execution::env<>{}),
       ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
       ::cuda::std::move(__cmp));
}

_CCCL_TEMPLATE(class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(::cuda::std::ranges::random_access_range<_InputRange> _CCCL_AND //
               ::cuda::std::ranges::sized_range<_InputRange>)
void sort(const communicator& __comm, _InputRange&& __input_range, _BinaryOp __cmp = {})
{
  sort(::cuda::std::span<const communicator, 1>{&__comm, /*__count=*/1},
       ::cuda::std::span<_InputRange, 1>{::cuda::std::addressof(__input_range),
                                         /*__count=*/1},
       ::cuda::std::move(__cmp));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_SORT_H
