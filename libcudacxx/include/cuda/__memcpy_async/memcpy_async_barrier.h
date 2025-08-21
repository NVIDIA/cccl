// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_
#define _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier.h>
#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/__barrier/barrier_thread_scope.h>
#include <cuda/__memcpy_async/completion_mechanism.h>
#include <cuda/__memcpy_async/dispatch_memcpy_async.h>
#include <cuda/__memcpy_async/is_local_smem_barrier.h>
#include <cuda/__memcpy_async/memcpy_completion.h>
#include <cuda/__memcpy_async/try_get_barrier_handle.h>
#include <cuda/__memory/aligned_size.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct __single_thread_group
{
  _CCCL_API inline void sync() const {}
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::size_t size() const
  {
    return 1;
  };
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::size_t thread_rank() const
  {
    return 0;
  };
};

template <typename _Group, class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment __memcpy_async_barrier(
  _Group const& __group, _Tp* __destination, _Tp const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "memcpy_async requires a trivially copyable type");

  // 1. Determine which completion mechanisms can be used with the current
  // barrier. A local shared memory barrier, i.e., block-scope barrier in local
  // shared memory, supports the mbarrier_complete_tx mechanism in addition to
  // the async group mechanism.
  ::cuda::std::uint32_t __allowed_completions =
    ::cuda::__is_local_smem_barrier(__barrier)
      ? (::cuda::std::uint32_t(__completion_mechanism::__async_group)
         | ::cuda::std::uint32_t(__completion_mechanism::__mbarrier_complete_tx))
      : ::cuda::std::uint32_t(__completion_mechanism::__async_group);

  // Alignment: Use the maximum of the alignment of _Tp and that of a possible cuda::aligned_size_t.
  constexpr auto __align = ::cuda::std::max(alignof(_Tp), __get_size_align_v<_Size>);
  // Cast to char pointers. We don't need the type for alignment anymore and
  // erasing the types reduces the number of instantiations of down-stream
  // functions.
  char* __dest_char      = reinterpret_cast<char*>(__destination);
  char const* __src_char = reinterpret_cast<char const*>(__source);

  // 2. Issue actual copy instructions.
  ::cuda::std::uint64_t* __bh = nullptr;
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (__bh = ::cuda::__is_local_smem_barrier(__barrier) ? ::cuda::__try_get_barrier_handle(__barrier) : nullptr;))
#endif // __cccl_ptx_isa >= 800
  auto __cm =
    ::cuda::__dispatch_memcpy_async<__align>(__group, __dest_char, __src_char, __size, __allowed_completions, __bh);

  // 3. Synchronize barrier with copy instructions.
  return __memcpy_completion_impl::__defer(__cm, __group, __size, __barrier);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_
