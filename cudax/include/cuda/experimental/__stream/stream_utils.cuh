//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_STREAM_UTILS_CUH
#define _CUDAX__STREAM_STREAM_UTILS_CUH

#include <cuda/std/detail/__config>
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/experimental/__stream/stream.cuh>

#include <vector>

namespace cuda::experimental
{
template <::cuda::std::size_t... _Idx>
[[nodiscard]] _CCCL_HOST_API auto __replicate_impl(logical_device __dev, ::cuda::std::index_sequence<_Idx...>)
  -> ::cuda::std::array<stream, sizeof...(_Idx)>
{
  return ::cuda::std::array<stream, sizeof...(_Idx)>{((void) _Idx, stream(__dev))...};
}

template <::cuda::std::size_t _Count>
[[nodiscard]] _CCCL_HOST_API auto __replicate_streams(stream_ref __source) -> ::cuda::std::array<stream, _Count>
{
  auto __dev = __source.logical_device();
  return __replicate_impl(__dev, ::cuda::std::make_index_sequence<_Count>{});
}

[[nodiscard]] _CCCL_HOST_API inline auto __replicate_streams(stream_ref __source, ::cuda::std::size_t __count)
  -> ::std::vector<stream>
{
  auto __dev = __source.logical_device();
  ::std::vector<stream> __streams;
  __streams.reserve(__count);
  for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
  {
    __streams.emplace_back(__dev);
  }
  return __streams;
}

template <class _Container>
_CCCL_HOST_API void __replicate_streams_into(stream_ref __source, _Container& __container, ::cuda::std::size_t __count)
{
  auto __dev = __source.logical_device();
  for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
  {
    __container.emplace_back(__dev);
  }
}

template <::cuda::std::size_t _Count>
[[nodiscard]] _CCCL_HOST_API auto stream_ref::replicate() const -> ::cuda::std::array<stream, _Count>
{
  return __replicate_streams<_Count>(*this);
}

[[nodiscard]] _CCCL_HOST_API inline auto stream_ref::replicate(::cuda::std::size_t __count) const
  -> ::std::vector<stream>
{
  return __replicate_streams(*this, __count);
}

template <class _Container>
_CCCL_HOST_API void stream_ref::replicate_into(_Container& __container, ::cuda::std::size_t __count) const
{
  __replicate_streams_into(*this, __container, __count);
}

template <class _Range>
_CCCL_CONCEPT __stream_wait_range =
  ::cuda::std::is_constructible_v<stream_ref, ::cuda::std::ranges::range_value_t<const _Range&>>;

//! @brief Make each target stream wait on `__event`, skipping the source stream itself.
template <class _ToRange, class _Event>
_CCCL_HOST_API void
__wait_all_targets_on_event(const _ToRange& __to_streams, stream_ref __from, const _Event& __local_event)
{
  for (const auto& __to_stream : __to_streams)
  {
    auto __to = stream_ref{__to_stream};
    if (__to != __from)
    {
      __to.wait(__local_event);
    }
  }
}

//! @brief Internal implementation for all-to-all stream waiting between two groups.
template <class _ToRange, class _FromRange>
_CCCL_HOST_API void __all_wait_all_impl(const _ToRange& __to_streams, const _FromRange& __from_streams)
{
  auto __to_begin = ::cuda::std::ranges::begin(__to_streams);
  if (__to_begin == ::cuda::std::ranges::end(__to_streams))
  {
    return;
  }

  auto __first                 = stream_ref{*__to_begin};
  auto __first_device          = __first.device();
  bool __use_per_stream_events = false;
  cuda::event __local_event(__first_device);
  for (const auto& __from_stream : __from_streams)
  {
    auto __from = stream_ref{__from_stream};

    if (!__use_per_stream_events)
    {
      auto __status = ::cuda::__driver::__eventRecordNoThrow(__local_event.get(), __from.get());
      if (__status == cudaSuccess)
      {
        __wait_all_targets_on_event(__to_streams, __from, __local_event);
        continue;
      }

      if (__from.device() != __first_device)
      {
        __use_per_stream_events = true;
      }
      else
      {
        _CCCL_THROW(::cuda::cuda_error, __status, "Failed to record an event");
      }
    }

    if (__use_per_stream_events)
    {
      auto __from_event = cuda::event{__from};
      __wait_all_targets_on_event(__to_streams, __from, __from_event);
    }
  }
}

// TODO: consider accumulating all sources into one target stream first and then
// propagating that dependency to other targets to reduce API calls.
//! @brief Make all streams in `__to_streams` wait on all streams in `__from_streams`.
_CCCL_TEMPLATE(class _ToRange, class _FromRange)
_CCCL_REQUIRES(
  ::cuda::std::ranges::forward_range<const _ToRange&> _CCCL_AND ::cuda::std::ranges::forward_range<const _FromRange&>
    _CCCL_AND __stream_wait_range<_ToRange> _CCCL_AND __stream_wait_range<_FromRange>)
_CCCL_HOST_API void all_wait_all(const _ToRange& __to_streams, const _FromRange& __from_streams)
{
  __all_wait_all_impl(__to_streams, __from_streams);
}

//! @brief Make a single target stream wait on all streams in `__from_streams`.
_CCCL_TEMPLATE(class _FromRange)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<const _FromRange&> _CCCL_AND __stream_wait_range<_FromRange>)
_CCCL_HOST_API void all_wait_all(stream_ref __to_stream, const _FromRange& __from_streams)
{
  __all_wait_all_impl(::cuda::std::span<stream_ref>(&__to_stream, 1), __from_streams);
}

//! @brief Make all streams in `__to_streams` wait on a single source stream.
_CCCL_TEMPLATE(class _ToRange)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<const _ToRange&> _CCCL_AND __stream_wait_range<_ToRange>)
_CCCL_HOST_API void all_wait_all(const _ToRange& __to_streams, stream_ref __from_stream)
{
  __all_wait_all_impl(__to_streams, ::cuda::std::span<stream_ref>(&__from_stream, 1));
}
} // namespace cuda::experimental

#endif // _CUDAX__STREAM_STREAM_UTILS_CUH
