//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_CONCEPTS_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_CONCEPTS_H

#include <cuda/std/detail/__config> // IWYU pragma: export

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace cuda::experimental
{
// Needed because the C++17 concept emulation can't handle the implicit first template
// parameter of real concepts.
template <class _Tp>
_CCCL_CONCEPT __convertible_to_int32 = ::cuda::std::convertible_to<_Tp, ::cuda::std::int32_t>;

// Requires communicator<_Comm> to be checked first, but that invokes a circular dependency
template <class _Comm, class _Ptr = void*>
_CCCL_CONCEPT __has_send = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __buf,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __peer,
  ::cuda::stream_ref __stream)(_Same_as(void) __comm.send(
  ::cuda::std::declval<typename _Comm::group_guard_type&>(), __buf, __count, __peer, __stream));

// Requires communicator<_Comm> to be checked first, but that invokes a circular dependency
template <class _Comm, class _Ptr = void*>
_CCCL_CONCEPT __has_recv = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __buf,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __peer,
  ::cuda::stream_ref __stream)(_Same_as(void) __comm.recv(
  ::cuda::std::declval<typename _Comm::group_guard_type&>(), __buf, __count, __peer, __stream));

template <class _Comm>
_CCCL_CONCEPT __communicator = _CCCL_REQUIRES_EXPR((_Comm), _Comm& __comm)(
  typename(typename _Comm::native_handle_type),
  _Same_as(typename _Comm::native_handle_type) __comm.native_handle(),
  noexcept(__comm.native_handle()),
  _Satisfies(__convertible_to_int32) __comm.rank(),
  _Satisfies(__convertible_to_int32) __comm.size(),
  typename(typename _Comm::group_guard_type),
  _Same_as(typename _Comm::group_guard_type) __comm.group_guard(),
  requires(__has_send<_Comm>),
  requires(__has_recv<_Comm>) //
);

// ==========================================================================================

// Use a typed pointer as default here, since the op may need to instantiated with a
// dereferenceable pointer type for reductions
template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_reduce = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::std::plus<> __op,
  ::cuda::std::int32_t __root,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.reduce(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __op, __root, __stream));

// ==========================================================================================

// Use a typed pointer as default here, since the op may need to instantiated with a
// dereferenceable pointer type for reductions
template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_all_reduce = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::std::plus<> __op,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.all_reduce(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __op, __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_gather = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __root,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.gather(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __root, __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_gather_v = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  ::cuda::std::size_t __send_count,
  _Ptr __recvbuff,
  const ::cuda::std::size_t* __recv_counts,
  const ::cuda::std::size_t* __displs,
  ::cuda::std::int32_t __root,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.gather_v(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(),
    __sendbuff,
    __send_count,
    __recvbuff,
    __recv_counts,
    __displs,
    __root,
    __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_all_gather = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.all_gather(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_broadcast = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __root,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.broadcast(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __root, __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_all_to_all = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.all_to_all(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(), __sendbuff, __recvbuff, __count, __stream));

// ==========================================================================================

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_all_to_all_v = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  const ::cuda::std::size_t* __send_counts,
  const ::cuda::std::size_t* __send_displs,
  _Ptr __recvbuff,
  const ::cuda::std::size_t* __recv_counts,
  const ::cuda::std::size_t* __recv_displs,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void) __comm.all_to_all_v(
    ::cuda::std::declval<typename _Comm::group_guard_type&>(),
    __sendbuff,
    __send_counts,
    __send_displs,
    __recvbuff,
    __recv_counts,
    __recv_displs,
    __stream));
} // namespace cuda::experimental
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_CONCEPTS_H
