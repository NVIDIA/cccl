//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__nccl/nccl_api.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class nccl_communicator
{
  struct __private_tag
  {};

public:
  // Hide the fact that we proxy the types
#ifdef _CCCL_DOXYGEN_INVOKED
  using native_handle_type = ncclComm_t;
  using group_token_type   = /* implementation-defined */;
#else
  using native_handle_type = __nccl::__ncclComm_t;
  using group_token_type   = __nccl::__ensure_nccl_group;
#endif

  _CCCL_HOST_API nccl_communicator(native_handle_type __comm, logical_device __device)
      : nccl_communicator{__private_tag{}, ::cuda::std::move(__comm), ::cuda::std::move(__device)}
  {}

  [[nodiscard]] _CCCL_HOST_API constexpr native_handle_type native_handle() const noexcept
  {
    return __comm_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::int32_t rank() const
  {
    return __nccl::__ncclCommUserRank(native_handle());
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::int32_t size() const
  {
    return __nccl::__ncclCommCount(native_handle());
  }

  [[nodiscard]] _CCCL_HOST_API group_token_type group_token() const
  {
    return {};
  }

  [[nodiscard]] _CCCL_HOST_API constexpr const logical_device& device() const noexcept
  {
    return __device_;
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_HOST_API void
  send(const group_token_type&,
       const void* __buf,
       ::cuda::std::size_t __count,
       ::cuda::std::int32_t __peer,
       ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclSend(__buf, __count, __nccl::__ncclChar, __peer, native_handle(), __stream);
  }

  _CCCL_HOST_API void send_sync(
    const group_token_type& __tok, const void* __buf, ::cuda::std::size_t __count, ::cuda::std::int32_t __peer) const
  {
    send(__tok, __buf, __count, __peer, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_HOST_API void
  recv(const group_token_type&,
       void* __buf,
       ::cuda::std::size_t __count,
       ::cuda::std::int32_t __peer,
       ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclRecv(__buf, __count, __nccl::__ncclChar, __peer, native_handle(), __stream);
  }

  _CCCL_HOST_API void
  recv_sync(const group_token_type& __tok, void* __buf, ::cuda::std::size_t __count, ::cuda::std::int32_t __peer) const
  {
    recv(__tok, __buf, __count, __peer, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp> _CCCL_AND __nccl::__has_nccl_redop<_Op>)
  _CCCL_HOST_API void reduce(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    _Op,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclReduce(
      __sendbuf,
      __recvbuf,
      __count,
      __nccl::__nccl_type_of_v<_Tp>,
      __nccl::__nccl_redop_of_v<_Op>,
      __root,
      native_handle(),
      __stream);
  }

  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp> _CCCL_AND __nccl::__has_nccl_redop<_Op>)
  _CCCL_HOST_API void reduce_sync(
    const group_token_type& __tok,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    _Op __op,
    ::cuda::std::int32_t __root) const
  {
    reduce(__tok, __sendbuf, __recvbuf, __count, __op, __root, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp> _CCCL_AND __nccl::__has_nccl_redop<_Op>)
  _CCCL_HOST_API void all_reduce(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    _Op,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclAllReduce(
      __sendbuf,
      __recvbuf,
      __count,
      __nccl::__nccl_type_of_v<_Tp>,
      __nccl::__nccl_redop_of_v<_Op>,
      native_handle(),
      __stream);
  }

  _CCCL_TEMPLATE(class _Tp, class _Op)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp> _CCCL_AND __nccl::__has_nccl_redop<_Op>)
  _CCCL_HOST_API void all_reduce_sync(
    const group_token_type& __tok, const _Tp* __sendbuf, _Tp* __recvbuf, ::cuda::std::size_t __count, _Op __op) const
  {
    all_reduce(__tok, __sendbuf, __recvbuf, __count, __op, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void gather(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclGather(
      __sendbuf, __recvbuf, __count, __nccl::__nccl_type_of_v<_Tp>, __root, native_handle(), __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void gather_sync(
    const group_token_type& __tok,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root) const
  {
    gather(__tok, __sendbuf, __recvbuf, __count, __root, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void gather_v(
    const group_token_type&,
    const _Tp* __sendbuf,
    ::cuda::std::size_t __send_count,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __recv_counts,
    const ::cuda::std::size_t* __displs,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclGatherv(
      __sendbuf,
      __send_count,
      __recvbuf,
      __nccl::__nccl_type_of_v<_Tp>,
      __recv_counts,
      __displs,
      __root,
      native_handle(),
      __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void gather_v_sync(
    const group_token_type& __tok,
    const _Tp* __sendbuf,
    ::cuda::std::size_t __send_count,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __recv_counts,
    const ::cuda::std::size_t* __displs,
    ::cuda::std::int32_t __root) const
  {
    gather_v(__tok, __sendbuf, __send_count, __recvbuf, __recv_counts, __displs, __root, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_gather(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclAllGather(__sendbuf, __recvbuf, __count, __nccl::__nccl_type_of_v<_Tp>, native_handle(), __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_gather_sync(
    const group_token_type& __tok, const _Tp* __sendbuf, _Tp* __recvbuf, ::cuda::std::size_t __count) const
  {
    all_gather(__tok, __sendbuf, __recvbuf, __count, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void broadcast(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclBroadcast(
      __sendbuf, __recvbuf, __count, __nccl::__nccl_type_of_v<_Tp>, __root, native_handle(), __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void broadcast_sync(
    const group_token_type& __tok,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::std::int32_t __root) const
  {
    broadcast(__tok, __sendbuf, __recvbuf, __count, __root, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_to_all(
    const group_token_type&,
    const _Tp* __sendbuf,
    _Tp* __recvbuf,
    ::cuda::std::size_t __count,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclAlltoAll(__sendbuf, __recvbuf, __count, __nccl::__nccl_type_of_v<_Tp>, native_handle(), __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_to_all_sync(
    const group_token_type& __tok, const _Tp* __sendbuf, _Tp* __recvbuf, ::cuda::std::size_t __count) const
  {
    all_to_all(__tok, __sendbuf, __recvbuf, __count, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_to_all_v(
    const group_token_type&,
    const _Tp* __sendbuf,
    const ::cuda::std::size_t* __send_counts,
    const ::cuda::std::size_t* __send_displs,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __recv_counts,
    const ::cuda::std::size_t* __recv_displs,
    ::cuda::stream_ref __stream) const
  {
    __nccl::__ncclAlltoAllv(
      __sendbuf,
      __send_counts,
      __send_displs,
      __recvbuf,
      __recv_counts,
      __recv_displs,
      __nccl::__nccl_type_of_v<_Tp>,
      native_handle(),
      __stream);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__nccl::__has_nccl_type_of<_Tp>)
  _CCCL_HOST_API void all_to_all_v_sync(
    const group_token_type& __tok,
    const _Tp* __sendbuf,
    const ::cuda::std::size_t* __send_counts,
    const ::cuda::std::size_t* __send_displs,
    _Tp* __recvbuf,
    const ::cuda::std::size_t* __recv_counts,
    const ::cuda::std::size_t* __recv_displs) const
  {
    all_to_all_v(__tok, __sendbuf, __send_counts, __send_displs, __recvbuf, __recv_counts, __recv_displs, ::CUstream{});
  }

  // ------------------------------------------------------------------------------------------

private:
  // Grand central constructor, all other constructors must come through this one
  _CCCL_HOST_API nccl_communicator(const __private_tag, native_handle_type __comm, logical_device __device)
      : __comm_{[&] {
        if (const auto __nccl_device = __nccl::__ncclCommCuDevice(__comm);
            __nccl_device != __device.underlying_device())
        {
          _CCCL_THROW(::std::runtime_error,
                      "Inconsistent devices, NCCL communicator device and provided logical device do not match");
        }
        return __comm;
      }()}
      , __device_{::cuda::std::move(__device)}
  {}

  native_handle_type __comm_{};
  logical_device __device_;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H
