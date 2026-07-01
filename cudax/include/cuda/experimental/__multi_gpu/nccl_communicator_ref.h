//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_REF_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__nccl/nccl_api.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
//! @brief The `nccl_transportable` concept verifies that a particular type is transportable by
//! NCCL.
//!
//! @tparam _Tp The type checked for NCCL transportability.
//!
//! A type is NCCL-transportable when it can be communicated either through a native NCCL
//! datatype mapping or as an uninterpreted byte sequence. Trivially copyable types are sent by
//! byte count using NCCL character transport. `void` is also accepted for byte-oriented
//! operations where the count is already expressed in bytes.
template <class _Tp>
_CCCL_CONCEPT nccl_transportable =
  ::cuda::experimental::__nccl::__has_nccl_type_of<_Tp> || ::cuda::is_trivially_copyable_v<_Tp>
  || ::cuda::std::is_void_v<_Tp>;

//! @brief The `nccl_reducible` concept verifies that a particular pair of type and operation
//! is directly mappable to a NCCL reduction operation.
//!
//! @tparam _Tp The data type checked for NCCL reduction support.
//! @tparam _Op The operator type checked for NCCL reduction support.
//!
//! A type and reduction operator are NCCL-reducible when the type has a native NCCL datatype
//! mapping and the operator has a native NCCL reduction-operator mapping. Note that these are
//! stricter requirements than `nccl_transportable`. Reductions require an exact type match.
template <class _Tp, class _Op>
_CCCL_CONCEPT nccl_reducible =
  ::cuda::experimental::__nccl::__has_nccl_type_of<_Tp> && ::cuda::experimental::__nccl::__has_nccl_redop_of<_Op>;

//! @brief A non-owning wrapper around a NCCL communicator (`ncclComm_t`).
//!
//! `nccl_communicator_ref` adapts a previously-created NCCL communicator to the
//! `cuda::experimental` communicator model, exposing NCCL's point-to-point and collective
//! operations as member functions. It does not own the underlying communicator: the caller
//! is responsible for creating it (e.g. via `ncclCommInitRank`) and destroying it once it is
//! no longer in use, and for keeping it alive for the lifetime of this object.
//!
//! Each communication method takes a reference to a `group_guard_type` as its first
//! argument. The guard represents an active NCCL group (see
//! https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html). A guard
//! **must** be held whenever more than one communication call is issued together, in
//! particular when issuing calls from a loop. An individual NCCL call may block waiting for
//! the matching call on its peers, so issuing interdependent operations one at a time without
//! a group can deadlock. The guard is obtained from `group_guard()` and submits the
//! grouped operations when it is destroyed.
class _CCCL_VISIBILITY_DEFAULT nccl_communicator_ref
{
  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
  __as_bytes_count(::cuda::std::size_t __count) noexcept
  {
    if constexpr (::cuda::std::is_void_v<_Tp>)
    {
      // We assume if the incoming pointer is cv void* then the count already is in bytes
      return __count;
    }
    else
    {
      return __count * sizeof(_Tp);
    }
    _CCCL_UNREACHABLE();
  }

public:
  // Hide the fact that we proxy the types
#ifdef _CCCL_DOXYGEN_INVOKED
  //! @brief The native NCCL communicator handle type.
  using native_handle_type = ::ncclComm_t;
  //! @brief RAII guard representing an active NCCL group; see `group_guard()`.
  using group_guard_type = /* implementation-defined */;
#else
  using native_handle_type = ::cuda::experimental::__nccl::__ncclComm_t;
  using group_guard_type   = ::cuda::experimental::__nccl::__ensure_nccl_group;
#endif

  //! @brief Disallow direct construction from `NCCL_COMM_NULL`.
  _CCCL_HIDE_FROM_ABI nccl_communicator_ref(::cuda::std::nullptr_t) = delete;

  //! @brief Construct a communicator from an existing NCCL communicator handle.
  //!
  //! The communicator is not owned; the caller retains responsibility for its lifetime and
  //! destruction.
  //!
  //! @param __comm The NCCL communicator handle to wrap. Must outlive this object.
  //!
  //! @throws std::invalid_argument If `__comm` is `NCCL_COMM_NULL`.
  _CCCL_HOST_API nccl_communicator_ref(native_handle_type __comm)
      : nccl_communicator_ref{
          __comm, ::cuda::experimental::logical_device{::cuda::experimental::__nccl::__ncclCommCuDevice(__comm)}}
  {}

  //! @brief Construct a communicator from an existing NCCL communicator handle.
  //!
  //! The communicator is not owned; the caller retains responsibility for its lifetime and
  //! destruction.
  //!
  //! @param __comm The NCCL communicator handle to wrap. Must outlive this object.
  //! @param __device The logical device the communicator is expected to be associated with.
  //!
  //! @throws std::invalid_argument If `__comm` is `NCCL_COMM_NULL`.
  //! @throws std::runtime_error If the device reported by NCCL does not match `__device`.
  _CCCL_HOST_API nccl_communicator_ref(native_handle_type __comm, logical_device __device)
      : __comm_{[&] {
        if (__comm == nullptr)
        {
          _CCCL_THROW(::std::invalid_argument, "Invalid NCCL communicator: NCCL_COMM_NULL");
        }
        return __comm;
      }()}
      , __device_{::cuda::std::move(__device)}
      , __rank_{::cuda::experimental::__nccl::__ncclCommUserRank(native_handle())}
      , __size_{::cuda::experimental::__nccl::__ncclCommCount(native_handle())}
  {
    if (const auto __nccl_device = ::cuda::experimental::__nccl::__ncclCommCuDevice(native_handle());
        __nccl_device != logical_device().underlying_device())
    {
      _CCCL_THROW(::std::runtime_error,
                  "Inconsistent devices, NCCL communicator device and provided logical device do not match");
    }
  }

  //! @brief Retrieve the underlying native NCCL communicator handle.
  //!
  //! @return The wrapped NCCL handle.
  [[nodiscard]] _CCCL_HOST_API constexpr native_handle_type native_handle() const noexcept
  {
    return __comm_;
  }

  //! @brief Retrieve the rank of this process within the communicator.
  //!
  //! @return The caller's rank, in the range `[0, size())`.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::int32_t rank() const noexcept
  {
    return __rank_;
  }

  //! @brief Retrieve the number of ranks in the communicator.
  //!
  //! @return The total number of ranks participating in the communicator.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::int32_t size() const noexcept
  {
    return __size_;
  }

  //! @brief Open a new NCCL group.
  //!
  //! Returns an RAII guard that starts a NCCL group on construction and submits it on
  //! destruction. The returned guard must be passed to the communication methods and kept
  //! alive while issuing them; this is required whenever multiple, possibly interdependent,
  //! operations are issued together (for example inside a loop) to avoid deadlock.
  //!
  //! @return A `group_guard_type` bound to a freshly-opened NCCL group.
  [[nodiscard]] _CCCL_HOST_API group_guard_type group_guard() const
  {
    return {};
  }

  //! @brief Retrieve the logical device this communicator is associated with.
  //!
  //! @return A reference to the logical device passed at construction.
  [[nodiscard]] _CCCL_HOST_API constexpr const ::cuda::experimental::logical_device& logical_device() const noexcept
  {
    return __device_;
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Send a buffer to a single peer rank (point-to-point).
  //!
  //! @tparam _Tp The element type of the buffer. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __buf The buffer to send, holding `__count` elements of type `_Tp`.
  //! @param[in] __count The number of elements to send.
  //! @param[in] __peer The rank to send the data to.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void
  send(group_guard_type&,
       const _Tp* __buf,
       ::cuda::std::size_t __count,
       ::cuda::std::int32_t __peer,
       ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclSend(
      __buf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      __peer,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Receive a buffer from a single peer rank (point-to-point).
  //!
  //! @tparam _Tp The element type of the buffer. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[out] __buf The buffer to receive into, sized for `__count` elements of type `_Tp`.
  //! @param[in] __count The number of elements to receive.
  //! @param[in] __peer The rank to receive the data from.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void
  recv(group_guard_type&,
       _Tp* __buf,
       ::cuda::std::size_t __count,
       ::cuda::std::int32_t __peer,
       ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclRecv(
      __buf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      __peer,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Reduce buffers from all ranks onto a single root rank.
  //!
  //! Element-wise combines `__sendbuf` across every rank using `_Op` and stores the result
  //! in `__recvbuf` on `__root`. Unlike the byte-transporting operations, the element type
  //! and reduction operator are mapped onto native NCCL datatype and reduction-operator
  //! values, so `_Tp` must have a corresponding NCCL datatype and `_Op` a corresponding NCCL
  //! reduction operator.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy the data-type part of
  //! `nccl_reducible` with `_Op`.
  //! @tparam _Op The reduction operator type. `_Op` must satisfy the reduction-operator part
  //! of `nccl_reducible` with `_Tp`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The buffer contributed by this rank, holding `__count` elements.
  //! @param[out] __recvbuf The buffer receiving the result on `__root`; ignored on other
  //! ranks.
  //! @param[in] __count The number of elements to reduce.
  //! @param[in] (unnamed) The reduction operator.
  //! @param[in] __root The rank that receives the reduced result.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(nccl_reducible<_Tp, _Op>)
  _CCCL_HOST_API void reduce(
    group_guard_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    const _Op&,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclReduce(
      __sendbuf,
      __recvbuf,
      __count,
      ::cuda::experimental::__nccl::__nccl_type_of_v<_Tp>,
      ::cuda::experimental::__nccl::__nccl_redop_of_v<_Op>,
      __root,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Reduce buffers from all ranks and distribute the result to every rank.
  //!
  //! Element-wise combines `__sendbuf` across every rank using `_Op` and stores the result
  //! in `__recvbuf` on all ranks. As with `reduce()`, the element type and reduction
  //! operator are mapped onto native NCCL datatype and reduction-operator values.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy the data-type part of
  //! `nccl_reducible` with `_Op`.
  //! @tparam _Op The reduction operator type. `_Op` must satisfy the reduction-operator part
  //! of `nccl_reducible` with `_Tp`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The buffer contributed by this rank, holding `__count` elements.
  //! @param[out] __recvbuf The buffer receiving the result on every rank.
  //! @param[in] __count The number of elements to reduce.
  //! @param[in] (unnamed) The reduction operator.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(nccl_reducible<_Tp, _Op>)
  _CCCL_HOST_API void all_reduce(
    group_guard_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    const _Op&,
    ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclAllReduce(
      __sendbuf,
      __recvbuf,
      __count,
      ::cuda::experimental::__nccl::__nccl_type_of_v<_Tp>,
      ::cuda::experimental::__nccl::__nccl_redop_of_v<_Op>,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Gather equal-sized buffers from all ranks onto a single root rank.
  //!
  //! Collects `__count` elements from every rank's `__sendbuf` into `__recvbuf` on
  //! `__root`, laid out contiguously in rank order. `__recvbuf` is only written on
  //! `__root` and must be sized for `__count * size()` elements there.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The buffer contributed by this rank, holding `__count` elements.
  //! @param[out] __recvbuf The gathered output on `__root`; ignored on other ranks.
  //! @param[in] __count The number of elements contributed by each rank.
  //! @param[in] __root The rank that receives the gathered data.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void gather(
    group_guard_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclGather(
      __sendbuf,
      __recvbuf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      __root,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

private:
  // Try as hard as possible to hint to compilers that they can outline this function instead
  // of generating a copy of it for every type instation. We can probably go further here by
  // marking this as _CCCL_NOINLINE, but then we miss out on some nice optimizations for
  // size-count constant folding.
  _CCCL_HOST_API void __gather_v_impl(
    group_guard_type& __guard,
    const char* __sendbuf_bytes,
    ::cuda::std::size_t __send_count,
    char* __recvbuf_bytes,
    const ::cuda::std::size_t* __h_recv_counts,
    const ::cuda::std::size_t* __h_displs,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream,
    ::cuda::std::size_t __type_size) const
  {
    const auto __send_count_bytes = __type_size * __send_count;

    if (rank() == __root)
    {
      const auto __size = size();

      for (::cuda::std::int32_t __peer = 0; __peer < __size; ++__peer)
      {
        const auto __recv_count_bytes = __type_size * __h_recv_counts[__peer];
        const auto __displs_bytes     = __type_size * __h_displs[__peer];

        if (__recv_count_bytes == 0)
        {
          continue;
        }

        auto* const __recv_ptr_bytes = __recvbuf_bytes + __displs_bytes;

        if (__peer == __root)
        {
          if (__send_count_bytes != __recv_count_bytes)
          {
            _CCCL_THROW(::cuda::experimental::__nccl::nccl_error,
                        ::cuda::experimental::__nccl::__ncclInvalidArgument,
                        "Mismatched self-copy count in Gatherv");
          }

          // Unclear whether CUDA driver also makes this optimization
          if (__sendbuf_bytes != __recv_ptr_bytes)
          {
            ::cuda::__driver::__memcpyAsync(__recv_ptr_bytes, __sendbuf_bytes, __send_count_bytes, __stream.get());
          }
        }
        else
        {
          recv(__guard, __recv_ptr_bytes, __recv_count_bytes, __peer, __stream);
        }
      }
    }
    else if (__send_count)
    {
      send(__guard, __sendbuf_bytes, __send_count_bytes, __root, __stream);
    }
  }

public:
  //! @brief Gather variable-sized buffers from all ranks onto a single root rank.
  //!
  //! Like `gather()`, but each rank may contribute a different number of elements and the
  //! root places each contribution at a caller-specified offset. This is built from
  //! point-to-point `send()` / `recv()` calls; the root's own contribution is copied
  //! directly on `__stream` rather than sent. `__h_recv_counts` and `__h_displs` are read
  //! on the root only and must point to host memory with `size()` entries each.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] __guard An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The buffer contributed by this rank, holding `__send_count`
  //! elements.
  //! @param[in] __send_count The number of elements this rank contributes.
  //! @param[out] __recvbuf The gathered output on `__root`; ignored on other ranks.
  //! @param[in] __h_recv_counts Host array of per-rank element counts (read on `__root`).
  //! @param[in] __h_displs Host array of per-rank element offsets into `__recvbuf` (read on
  //! `__root`).
  //! @param[in] __root The rank that receives the gathered data.
  //! @param[in] __stream The stream to enqueue the operations on.
  //!
  //! @throws nccl_error if a count mismatch is detected for the root's self-copy, or if an
  //!         underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void gather_v(
    group_guard_type& __guard,
    const _Tp* __sendbuf,
    ::cuda::std::size_t __send_count,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __h_recv_counts,
    const ::cuda::std::size_t* __h_displs,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    __gather_v_impl(
      __guard,
      reinterpret_cast<const char*>(__sendbuf),
      __send_count,
      reinterpret_cast<char*>(__recvbuf),
      __h_recv_counts,
      __h_displs,
      __root,
      __stream,
      __as_bytes_count<_Tp>(/*__count=*/1));
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Gather equal-sized buffers from all ranks and distribute the result to every rank.
  //!
  //! Collects `__count` elements from every rank's `__sendbuf` into `__recvbuf` on all
  //! ranks, laid out contiguously in rank order. `__recvbuf` must be sized for
  //! `__count * size()` elements on every rank.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The buffer contributed by this rank, holding `__count` elements.
  //! @param[out] __recvbuf The gathered output, written on every rank.
  //! @param[in] __count The number of elements contributed by each rank.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void all_gather(
    group_guard_type&, const _Tp* __sendbuf, _Tp* __recvbuf, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
    const
  {
    ::cuda::experimental::__nccl::__ncclAllGather(
      __sendbuf,
      __recvbuf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Broadcast a buffer from a single root rank to every rank.
  //!
  //! Copies `__count` elements from `__sendbuf` on `__root` into `__recvbuf` on every
  //! rank. `__sendbuf` is only read on `__root`.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The source buffer on `__root`; ignored on other ranks.
  //! @param[out] __recvbuf The buffer receiving the broadcast data on every rank.
  //! @param[in] __count The number of elements to broadcast.
  //! @param[in] __root The rank that supplies the data.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void broadcast(
    group_guard_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    ::cuda::experimental::__nccl::__ncclBroadcast(
      __sendbuf,
      __recvbuf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      __root,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

  //! @brief Exchange equal-sized blocks between every pair of ranks.
  //!
  //! Rank `i` sends the block of `__count` elements at offset `j * __count` in `__sendbuf`
  //! to rank `j`, which stores it at offset `i * __count` in its `__recvbuf`. Both buffers
  //! must be sized for `__count * size()` elements.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] (unnamed) An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The send buffer, holding one `__count`-element block per rank.
  //! @param[out] __recvbuf The receive buffer, holding one `__count`-element block per rank.
  //! @param[in] __count The number of elements exchanged with each rank.
  //! @param[in] __stream The stream to enqueue the operation on.
  //!
  //! @throws nccl_error if the underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void all_to_all(
    group_guard_type&, const _Tp* __sendbuf, _Tp* __recvbuf, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
    const
  {
    ::cuda::experimental::__nccl::__ncclAlltoAll(
      __sendbuf,
      __recvbuf,
      __as_bytes_count<_Tp>(__count),
      ::cuda::experimental::__nccl::__ncclChar,
      native_handle(),
      __stream);
  }

  // ------------------------------------------------------------------------------------------

private:
  // Try as hard as possible to hint to compilers that they can outline this function instead
  // of generating a copy of it for every type instation. We can probably go further here by
  // marking this as _CCCL_NOINLINE, but then we miss out on some nice optimizations for
  // size-count constant folding.
  _CCCL_HOST_API void __all_to_all_v_impl(
    group_guard_type& __guard,
    const char* __sendbuf_bytes,
    const ::cuda::std::size_t* __h_send_counts,
    const ::cuda::std::size_t* __h_send_displs,
    char* __recvbuf_bytes,
    const ::cuda::std::size_t* __h_recv_counts,
    const ::cuda::std::size_t* __h_recv_displs,
    ::cuda::stream_ref __stream,
    ::cuda::std::size_t __type_size) const
  {
    const auto __rank = rank();
    const auto __size = size();

    for (::cuda::std::int32_t __peer = 0; __peer < __size; ++__peer)
    {
      const auto __send_count_bytes = __type_size * __h_send_counts[__peer];
      const auto __recv_count_bytes = __type_size * __h_recv_counts[__peer];

      const auto __send_displs_bytes = __type_size * __h_send_displs[__peer];
      const auto __recv_displs_bytes = __type_size * __h_recv_displs[__peer];

      if (__peer == __rank)
      {
        if (__send_count_bytes != __recv_count_bytes)
        {
          _CCCL_THROW(::cuda::experimental::__nccl::nccl_error,
                      ::cuda::experimental::__nccl::__ncclInvalidArgument,
                      "Mismatched self-copy count in AlltoAllv");
        }

        if (__send_count_bytes == 0)
        {
          continue;
        }

        const auto* const __send_ptr_bytes = __sendbuf_bytes + __send_displs_bytes;
        auto* const __recv_ptr_bytes       = __recvbuf_bytes + __recv_displs_bytes;

        if (__send_ptr_bytes != __recv_ptr_bytes)
        {
          ::cuda::__driver::__memcpyAsync(__recv_ptr_bytes, __send_ptr_bytes, __send_count_bytes, __stream.get());
        }
        continue;
      }

      if (__recv_count_bytes)
      {
        recv(__guard, __recvbuf_bytes + __recv_displs_bytes, __recv_count_bytes, __peer, __stream);
      }

      if (__send_count_bytes)
      {
        send(__guard, __sendbuf_bytes + __send_displs_bytes, __send_count_bytes, __peer, __stream);
      }
    }
  }

public:
  //! @brief Exchange variable-sized blocks between every pair of ranks.
  //!
  //! Like `all_to_all()`, but the block exchanged with each rank may have a different size
  //! and a caller-specified offset within the send and receive buffers. This is built from
  //! point-to-point `send()` / `recv()` calls; the block destined for the calling rank
  //! itself is copied directly on `__stream` rather than sent. All four count/displacement
  //! arrays are read on host and must have `size()` entries each, indexed by peer rank.
  //!
  //! @tparam _Tp The element type of the buffers. `_Tp` must satisfy `nccl_transportable`.
  //!
  //! @param[in] __guard An active group guard obtained from `group_guard()`.
  //! @param[in] __sendbuf The send buffer.
  //! @param[in] __h_send_counts Host array of per-peer element counts to send.
  //! @param[in] __h_send_displs Host array of per-peer element offsets into `__sendbuf`.
  //! @param[out] __recvbuf The receive buffer.
  //! @param[in] __h_recv_counts Host array of per-peer element counts to receive.
  //! @param[in] __h_recv_displs Host array of per-peer element offsets into `__recvbuf`.
  //! @param[in] __stream The stream to enqueue the operations on.
  //!
  //! @throws nccl_error if a count mismatch is detected for the rank's self-copy, or if an
  //!         underlying NCCL call fails.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(nccl_transportable<_Tp>)
  _CCCL_HOST_API void all_to_all_v(
    group_guard_type& __guard,
    const _Tp* __sendbuf,
    const ::cuda::std::size_t* __h_send_counts,
    const ::cuda::std::size_t* __h_send_displs,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __h_recv_counts,
    const ::cuda::std::size_t* __h_recv_displs,
    ::cuda::stream_ref __stream) const
  {
    __all_to_all_v_impl(
      __guard,
      reinterpret_cast<const char*>(__sendbuf),
      __h_send_counts,
      __h_send_displs,
      reinterpret_cast<char*>(__recvbuf),
      __h_recv_counts,
      __h_recv_displs,
      __stream,
      __as_bytes_count<_Tp>(/*__count=*/1));
  }

private:
  native_handle_type __comm_{};
  ::cuda::experimental::logical_device __device_;
  // Cache these so we can make the accessors noexcept
  ::cuda::std::int32_t __rank_{};
  ::cuda::std::int32_t __size_{};
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_REF_H
