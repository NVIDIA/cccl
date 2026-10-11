//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMUNICATOR_H
#define _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMUNICATOR_H

#include <cuda/std/detail/__config>

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
#include <cuda/std/__concepts/same_as.h> // IWYU pragma: keep
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN

template <class _Comm>
using __group_guard_t _CCCL_NODEBUG = typename ::cuda::std::remove_cvref_t<_Comm>::group_guard_type;

template <class _Comm, class _Ptr = void*>
_CCCL_CONCEPT __has_send = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __buf,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __peer,
  ::cuda::stream_ref __stream)(
  _Same_as(void) __comm.send(::cuda::std::declval<__group_guard_t<_Comm>&>(), __buf, __count, __peer, __stream));

template <class _Comm, class _Ptr = void*>
_CCCL_CONCEPT __has_recv = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __buf,
  ::cuda::std::size_t __count,
  ::cuda::std::int32_t __peer,
  ::cuda::stream_ref __stream)(
  _Same_as(void) __comm.recv(::cuda::std::declval<__group_guard_t<_Comm>&>(), __buf, __count, __peer, __stream));

// C++17 concept emulation cannot handle the implicit first template parameter of concepts.
template <class _Tp>
_CCCL_CONCEPT __convertible_to_int32 = ::cuda::std::convertible_to<_Tp, ::cuda::std::int32_t>;

template <class _Comm>
_CCCL_CONCEPT __communicator_impl = _CCCL_REQUIRES_EXPR((_Comm), _Comm& __comm)(
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

template <class _Comm>
_CCCL_CONCEPT __communicator = __communicator_impl<::cuda::std::remove_cvref_t<_Comm>>;

_CCCL_END_NAMESPACE_CUDA_MGMN

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMUNICATOR_H
