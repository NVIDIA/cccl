//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CONTAINER_GRAPH_BUFFER_CUH
#define _CUDAX__CONTAINER_GRAPH_BUFFER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/invalid_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cstddef>
#  include <cuda/std/initializer_list>
#  include <cuda/std/span>

#  include <cuda/experimental/__graph/copy_bytes.cuh>
#  include <cuda/experimental/__graph/fill_bytes.cuh>
#  include <cuda/experimental/__graph/graph_memory_resource.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @rst
//! .. _cudax-container-graph-buffer:
//!
//! Graph buffer
//! ------------
//!
//! ``graph_buffer`` provides typed device memory allocated as a CUDA graph node.
//! It mirrors the API of ``cuda::buffer`` but takes a ``path_builder&`` instead of
//! a ``stream_ref``. Allocation inserts a ``cuGraphAddMemAllocNode`` into the graph.
//!
//! Memory can be freed in three ways:
//!   - ``destroy(path_builder&)`` — inserts a free node into the graph
//!   - ``destroy(stream_ref)`` — frees asynchronously on a stream (for memory that outlives the graph)
//!   - Destructor — frees on the stored stream if one was set via ``set_stream()``
//!
//! If the destructor runs with no stream set and the buffer is non-empty, it asserts
//! in debug mode. In release mode the memory leaks.
//!
//! @endrst
//! @tparam _Tp The element type stored in the buffer. Must be trivially copyable.
template <class _Tp>
class graph_buffer
{
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "graph_buffer requires T to be trivially copyable.");

public:
  using value_type      = _Tp;
  using pointer         = _Tp*;
  using const_pointer   = const _Tp*;
  using size_type       = ::cuda::std::size_t;
  using properties_list = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

private:
  graph_memory_resource __mr_;
  size_type __count_       = 0;
  _Tp* __buf_              = nullptr;
  ::cudaStream_t __stream_ = ::cuda::__invalid_stream();

  [[nodiscard]] _CCCL_HOST_API pointer __get_data() const noexcept
  {
    return __buf_;
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch.
  [[nodiscard]] _CCCL_HOST_API friend auto transform_launch_argument(::cuda::stream_ref, graph_buffer& __self) noexcept
    -> ::cuda::std::span<_Tp>
  {
    return {__self.__get_data(), __self.__count_};
  }

  //! @brief Causes the buffer to be treated as a const span when passed to cudax::launch.
  [[nodiscard]] _CCCL_HOST_API friend auto
  transform_launch_argument(::cuda::stream_ref, const graph_buffer& __self) noexcept -> ::cuda::std::span<const _Tp>
  {
    return {__self.__get_data(), __self.__count_};
  }

public:
  graph_buffer() = delete;

  //! @brief Allocates uninitialized storage for \p __count elements.
  _CCCL_HOST_API graph_buffer(path_builder& __pb, graph_memory_resource __mr, size_type __count, ::cuda::no_init_t)
      : __mr_(::cuda::std::move(__mr))
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : static_cast<_Tp*>(__mr_.allocate(__pb, __count_ * sizeof(_Tp), alignof(_Tp))))
  {}

  //! @brief Allocates storage and fills with \p __value.
  _CCCL_HOST_API graph_buffer(path_builder& __pb, graph_memory_resource __mr, size_type __count, const _Tp& __value)
      : __mr_(::cuda::std::move(__mr))
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : static_cast<_Tp*>(__mr_.allocate(__pb, __count_ * sizeof(_Tp), alignof(_Tp))))
  {
    if (__count_ > 0)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        ::cuda::std::uint8_t __byte_val =
          static_cast<::cuda::std::uint8_t>(reinterpret_cast<const unsigned char&>(__value));
        ::cuda::experimental::fill_bytes(__pb, ::cuda::std::span<_Tp>(__get_data(), __count_), __byte_val);
      }
      else
      {
        // TODO: support non-zero multi-byte values via a kernel node
        ::cuda::experimental::fill_bytes(
          __pb, ::cuda::std::span<_Tp>(__get_data(), __count_), static_cast<::cuda::std::uint8_t>(0));
      }
    }
  }

  //! @brief Allocates storage and copies from a contiguous span.
  _CCCL_HOST_API graph_buffer(path_builder& __pb, graph_memory_resource __mr, ::cuda::std::span<const _Tp> __src)
      : __mr_(::cuda::std::move(__mr))
      , __count_(__src.size())
      , __buf_(__count_ == 0 ? nullptr : static_cast<_Tp*>(__mr_.allocate(__pb, __count_ * sizeof(_Tp), alignof(_Tp))))
  {
    if (__count_ > 0)
    {
      ::cuda::experimental::copy_bytes(__pb, __src, ::cuda::std::span<_Tp>{__get_data(), __count_});
    }
  }

  //! @brief Allocates storage and copies from an initializer list.
  _CCCL_HOST_API graph_buffer(path_builder& __pb, graph_memory_resource __mr, ::cuda::std::initializer_list<_Tp> __ilist)
      : graph_buffer(__pb, ::cuda::std::move(__mr), ::cuda::std::span<const _Tp>{__ilist.begin(), __ilist.size()})
  {}

  graph_buffer(const graph_buffer&)            = delete;
  graph_buffer& operator=(const graph_buffer&) = delete;

  //! @brief Move-constructs from another graph_buffer.
  _CCCL_HOST_API graph_buffer(graph_buffer&& __other) noexcept
      : __mr_(::cuda::std::move(__other.__mr_))
      , __count_(::cuda::std::exchange(__other.__count_, 0))
      , __buf_(::cuda::std::exchange(__other.__buf_, nullptr))
      , __stream_(::cuda::std::exchange(__other.__stream_, ::cuda::__invalid_stream()))
  {}

  //! @brief Move-assigns from another graph_buffer.
  _CCCL_HOST_API graph_buffer& operator=(graph_buffer&& __other) noexcept
  {
    if (this != &__other)
    {
      _CCCL_ASSERT(__buf_ == nullptr || __stream_ != ::cuda::__invalid_stream(),
                   "graph_buffer move-assigned over non-empty buffer with no stream set");
      if (__buf_ != nullptr && __stream_ != ::cuda::__invalid_stream())
      {
        destroy(::cuda::stream_ref{__stream_});
      }
      __mr_     = ::cuda::std::move(__other.__mr_);
      __count_  = ::cuda::std::exchange(__other.__count_, 0);
      __buf_    = ::cuda::std::exchange(__other.__buf_, nullptr);
      __stream_ = ::cuda::std::exchange(__other.__stream_, ::cuda::__invalid_stream());
    }
    return *this;
  }

  //! @brief Destructor. Frees device memory on the stored stream if one was set.
  _CCCL_HOST_API ~graph_buffer()
  {
    if (__buf_ != nullptr)
    {
      _CCCL_ASSERT(__stream_ != ::cuda::__invalid_stream(),
                   "graph_buffer destroyed with live memory but no stream set. "
                   "Call set_stream(), destroy(stream_ref), or destroy(path_builder&) before destruction.");
      if (__stream_ != ::cuda::__invalid_stream())
      {
        destroy(::cuda::stream_ref{__stream_});
      }
    }
  }

  //! @brief Set the stream to use for automatic cleanup in the destructor.
  _CCCL_HOST_API void set_stream(::cuda::stream_ref __stream) noexcept
  {
    __stream_ = __stream.get();
  }

  //! @brief Returns the stream set for automatic cleanup.
  [[nodiscard]] _CCCL_HOST_API ::cuda::stream_ref stream() const noexcept
  {
    return ::cuda::stream_ref{__stream_};
  }

  //! @brief Insert a free node into the graph to deallocate the buffer.
  _CCCL_HOST_API graph_node_ref destroy(path_builder& __pb)
  {
    if (__buf_ == nullptr)
    {
      return graph_node_ref{};
    }

    __mr_.deallocate(__pb, __buf_, __count_ * sizeof(_Tp), alignof(_Tp));
    auto __free_node = __pb.get_dependencies()[0];
    __buf_           = nullptr;
    __count_         = 0;
    return graph_node_ref{__free_node, __pb.get_native_graph_handle()};
  }

  //! @brief Free the buffer's device memory asynchronously on a stream.
  _CCCL_HOST_API void destroy(::cuda::stream_ref __stream)
  {
    if (__buf_ != nullptr)
    {
      __mr_.deallocate(__stream, __buf_, __count_ * sizeof(_Tp), alignof(_Tp));
      __buf_   = nullptr;
      __count_ = 0;
    }
  }

  [[nodiscard]] _CCCL_HOST_API pointer data() noexcept
  {
    return __get_data();
  }

  [[nodiscard]] _CCCL_HOST_API const_pointer data() const noexcept
  {
    return __get_data();
  }

  [[nodiscard]] _CCCL_HOST_API pointer begin() noexcept
  {
    return __get_data();
  }

  [[nodiscard]] _CCCL_HOST_API const_pointer begin() const noexcept
  {
    return __get_data();
  }

  [[nodiscard]] _CCCL_HOST_API pointer end() noexcept
  {
    return __get_data() + __count_;
  }

  [[nodiscard]] _CCCL_HOST_API const_pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr size_type size() const noexcept
  {
    return __count_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr size_type size_bytes() const noexcept
  {
    return __count_ * sizeof(_Tp);
  }

  [[nodiscard]] _CCCL_HOST_API constexpr bool empty() const noexcept
  {
    return __count_ == 0;
  }

  [[nodiscard]] _CCCL_HOST_API const graph_memory_resource& memory_resource() const noexcept
  {
    return __mr_;
  }
};

//! @brief Create a graph_buffer with uninitialized storage.
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API graph_buffer<_Tp>
make_buffer(path_builder& __pb, graph_memory_resource __mr, ::cuda::std::size_t __count, ::cuda::no_init_t)
{
  return graph_buffer<_Tp>{__pb, ::cuda::std::move(__mr), __count, ::cuda::no_init};
}

//! @brief Create a graph_buffer filled with a value.
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API graph_buffer<_Tp>
make_buffer(path_builder& __pb, graph_memory_resource __mr, ::cuda::std::size_t __count, const _Tp& __value)
{
  return graph_buffer<_Tp>{__pb, ::cuda::std::move(__mr), __count, __value};
}

//! @brief Create a graph_buffer from a span of data.
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API graph_buffer<_Tp>
make_buffer(path_builder& __pb, graph_memory_resource __mr, ::cuda::std::span<const _Tp> __src)
{
  return graph_buffer<_Tp>{__pb, ::cuda::std::move(__mr), __src};
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__CONTAINER_GRAPH_BUFFER_CUH
