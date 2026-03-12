//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_STREAM_CUH
#define _CUDAX__STREAM_STREAM_CUH

#include <cuda/std/detail/__config>

#include <cuda_runtime_api.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__stream/invalid_stream.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh> // IWYU pragma: export
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief An owning wrapper for cudaStream_t.
struct stream : stream_ref
{
  // 0 is documented as default priority
  static constexpr int default_priority = 0;

  //! @brief Constructs a stream on a specified device and with specified priority
  //!
  //! Priority is defaulted to stream::default_priority
  //!
  //! @throws cuda_error if stream creation fails
  explicit stream(device_ref __dev, int __priority = default_priority)
      : stream_ref(::cuda::__invalid_stream())
  {
    [[maybe_unused]] __ensure_current_device __dev_setter(__dev);
    _CCCL_TRY_CUDA_API(
      ::cudaStreamCreateWithPriority, "Failed to create a stream", &__stream, cudaStreamNonBlocking, __priority);
  }

  //! @brief Constructs a stream on a specified logical device and with specified priority
  //!
  //! Priority is defaulted to stream::default_priority
  //!
  //! @throws cuda_error if stream creation fails
  explicit stream(::cuda::experimental::logical_device __dev, int __priority = default_priority)
      : stream_ref(::cuda::__invalid_stream())
  {
    [[maybe_unused]] __ensure_current_device __dev_setter(__dev);
    _CCCL_TRY_CUDA_API(
      ::cudaStreamCreateWithPriority, "Failed to create a stream", &__stream, cudaStreamNonBlocking, __priority);
  }

  //! @brief Construct a new `stream` object into the moved-from state.
  //!
  //! @post `stream()` returns an invalid stream handle
  // Can't be constexpr because __invalid_stream isn't
  explicit stream(no_init_t) noexcept
      : stream_ref(::cuda::__invalid_stream())
  {}

  //! @brief Move-construct a new `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in moved-from state.
  stream(stream&& __other) noexcept
      : stream(::cuda::std::exchange(__other.__stream, ::cuda::__invalid_stream()))
  {}

  stream(const stream&) = delete;

  //! Destroy the `stream` object
  //!
  //! @note If the stream fails to be destroyed, the error is silently ignored.
  ~stream()
  {
    if (__stream != ::cuda::__invalid_stream())
    {
      // Needs to call driver API in case current device is not set, runtime version would set dev 0 current
      // Alternative would be to store the device and push/pop here
      [[maybe_unused]] auto status = ::cuda::__driver::__streamDestroyNoThrow(__stream);
    }
  }

  //! @brief Move-assign a `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  stream& operator=(stream&& __other) noexcept
  {
    stream __tmp(::cuda::std::move(__other));
    ::cuda::std::swap(__stream, __tmp.__stream);
    return *this;
  }

  stream& operator=(const stream&) = delete;

  //! @brief Construct an `stream` object from a native `cudaStream_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return stream The constructed `stream` object
  //!
  //! @note The constructed `stream` object takes ownership of the native handle.
  [[nodiscard]] static stream from_native_handle(::cudaStream_t __handle)
  {
    return stream(__handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static stream from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static stream from_native_handle(::cuda::std::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaStream_t` handle and give up ownership.
  //!
  //! @return cudaStream_t The native handle being held by the `stream` object.
  //!
  //! @post The stream object is in a moved-from state.
  [[nodiscard]] ::cudaStream_t release()
  {
    return ::cuda::std::exchange(__stream, ::cuda::__invalid_stream());
  }

  //! @brief Returns a \c execution::scheduler that enqueues work on this stream.
  auto get_scheduler() const noexcept -> stream_ref
  {
    return *this;
  }

private:
  // Use `stream::from_native_handle(s)` to construct an owning `stream`
  // object from a `cudaStream_t` handle.
  explicit stream(::cudaStream_t __handle)
      : stream_ref(__handle)
  {}
};

template <size_t... _Idx>
[[nodiscard]] inline auto __replicate_impl(logical_device __dev, ::cuda::std::index_sequence<_Idx...>)
  -> ::cuda::std::array<stream, sizeof...(_Idx)>
{
  return ::cuda::std::array<stream, sizeof...(_Idx)>{((void) _Idx, stream(__dev))...};
}

template <size_t... _Idx>
[[nodiscard]] inline auto
__replicate_prepend_impl(stream&& __source, logical_device __dev, ::cuda::std::index_sequence<_Idx...>)
  -> ::cuda::std::array<stream, sizeof...(_Idx) + 1>
{
  return ::cuda::std::array<stream, sizeof...(_Idx) + 1>{::cuda::std::move(__source), ((void) _Idx, stream(__dev))...};
}

template <size_t _Count>
//! @brief Create a fixed-size group of streams on the same logical device as `__source`.
//!
//! @note This function only creates streams; it does not add synchronization dependencies.
//! @note The streams are created with the default priority, which can be changed by the user. // TODO expose stream
//! attributes?
[[nodiscard]] auto replicate(stream_ref __source) -> ::cuda::std::array<stream, _Count>
{
  auto __dev = __source.logical_device();
  return __replicate_impl(__dev, ::cuda::std::make_index_sequence<_Count>{});
}

//! @brief Create a runtime-sized group of streams on the same logical device as `__source`.
//!
//! @note This function only creates streams; it does not add synchronization dependencies.
[[nodiscard]] inline auto replicate(stream_ref __source, size_t __count) -> ::std::vector<stream>
{
  auto __dev = __source.logical_device();

  ::std::vector<stream> __streams;
  __streams.reserve(__count);

  for (size_t __idx = 0; __idx < __count; ++__idx)
  {
    __streams.emplace_back(__dev);
  }

  return __streams;
}

//! @brief Create a runtime-sized group of streams on the same logical device as `__source` and prepend `__source` at
//! index 0.
//!
//! @note This function only creates streams; it does not add synchronization dependencies.
//! The returned vector has size `__count + 1` and owns all streams.
[[nodiscard]] inline auto replicate_prepend(stream&& __source, size_t __count) -> ::std::vector<stream>
{
  auto __dev = __source.logical_device();

  ::std::vector<stream> __streams;
  __streams.reserve(__count + 1);
  __streams.emplace_back(::cuda::std::move(__source));

  for (size_t __idx = 0; __idx < __count; ++__idx)
  {
    __streams.emplace_back(__dev);
  }

  return __streams;
}

template <size_t _Count>
//! @brief Create a fixed-size group of streams on the same logical device as `__source` and prepend `__source` at
//! index 0.
//!
//! @note This function only creates streams; it does not add synchronization dependencies.
//! The returned array has size `_Count + 1` and owns all streams.
[[nodiscard]] inline auto replicate_prepend(stream&& __source) -> ::cuda::std::array<stream, _Count + 1>
{
  auto __dev = __source.logical_device();
  return __replicate_prepend_impl(::cuda::std::move(__source), __dev, ::cuda::std::make_index_sequence<_Count>{});
}

template <class _Range>
_CCCL_CONCEPT __stream_join_range =
  ::cuda::std::is_same_v<::cuda::std::ranges::range_value_t<const _Range&>, stream_ref>
  || ::cuda::std::is_same_v<::cuda::std::ranges::range_value_t<const _Range&>, stream>;

// TODO: consider accumulating all the dependencies in from range into a single stream in to_streams and then propagate
// that as a dependency to the other streams in to_streams This has a drawback of introducing an extra dependency on
// that selected stream and other streams in to_streams, but results in less API calls overall.
//! @brief Internal implementation for joining two stream groups.
//!
//! For each stream in `__from_streams`, this function makes each non-identical stream
//! in `__to_streams` wait on it.
template <class _ToRange, class _FromRange>
inline void __join_impl(const _ToRange& __to_streams, const _FromRange& __from_streams)
{
  auto __to_begin = ::cuda::std::ranges::begin(__to_streams);
  if (__to_begin == ::cuda::std::ranges::end(__to_streams))
  {
    return;
  }

  auto __first                 = stream_ref(*__to_begin);
  auto __first_device          = __first.device();
  bool __use_per_stream_events = false;
  cuda::event __event(__first_device);
  for (const auto& __from_stream : __from_streams)
  {
    auto __from = stream_ref(__from_stream);

    if (!__use_per_stream_events)
    {
      auto __status = ::cuda::__driver::__eventRecordNoThrow(__event.get(), __from.get());
      if (__status == cudaSuccess)
      {
        for (const auto& __to_stream : __to_streams)
        {
          auto __to = stream_ref(__to_stream);
          if (__to != __from)
          {
            __to.wait(__event);
          }
        }
        continue;
      }

      // If shared event record failed, detect cross-device mismatch and
      // switch to per-source-stream events; otherwise propagate the failure.
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
      // Some stream/device combinations cannot record into a shared event.
      // Fall back to an event allocated for each source stream's device.
      auto __from_event = cuda::event(__from);

      for (const auto& __to_stream : __to_streams)
      {
        auto __to = stream_ref(__to_stream);
        if (__to != __from)
        {
          __to.wait(__from_event);
        }
      }
    }
  }
}

//! @brief Synchronize one group of streams with another.
//!
//! Every stream in `__to_streams` waits on work from every stream in `__from_streams`,
//! excluding identical stream pairs.
_CCCL_TEMPLATE(class _ToRange, class _FromRange)
_CCCL_REQUIRES(
  ::cuda::std::ranges::forward_range<const _ToRange&> _CCCL_AND ::cuda::std::ranges::forward_range<const _FromRange&>
    _CCCL_AND __stream_join_range<_ToRange> _CCCL_AND __stream_join_range<_FromRange>)
inline void join(const _ToRange& __to_streams, const _FromRange& __from_streams)
{
  __join_impl(__to_streams, __from_streams);
}

//! @brief Synchronize a single target stream with a source stream group.
_CCCL_TEMPLATE(class _FromRange)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<const _FromRange&> _CCCL_AND __stream_join_range<_FromRange>)
inline void join(stream_ref __to_stream, const _FromRange& __from_streams)
{
  __join_impl(::cuda::std::span<stream_ref>(&__to_stream, 1), __from_streams);
}

//! @brief Synchronize a target stream group with a single source stream.
_CCCL_TEMPLATE(class _ToRange)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<const _ToRange&> _CCCL_AND __stream_join_range<_ToRange>)
inline void join(const _ToRange& __to_streams, stream_ref __from_stream)
{
  __join_impl(__to_streams, ::cuda::std::span<stream_ref>(&__from_stream, 1));
}

// TODO should there be join for single stream on both sides that lowers to stream1.wait(stream2)?
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__STREAM_STREAM_CUH
