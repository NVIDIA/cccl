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

#ifndef _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_
#define _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_

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
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************
 * cuda::memcpy_async dispatch helper functions
 *
 * - __get_size_align struct to determine the alignment from a size type.
 ***********************************************************************/

// The __get_size_align struct provides a way to query the guaranteed
// "alignment" of a provided size. In this case, an n-byte aligned size means
// that the size is a multiple of n.
//
// Use as follows:
// static_assert(__get_size_align<size_t>::align == 1)
// static_assert(__get_size_align<aligned_size_t<n>>::align == n)

// Default impl: always returns 1.
template <typename, typename = void>
struct __get_size_align
{
  static constexpr int align = 1;
};

// aligned_size_t<n> overload: return n.
template <typename T>
struct __get_size_align<T, _CUDA_VSTD::void_t<decltype(T::align)>>
{
  static constexpr int align = T::align;
};

////////////////////////////////////////////////////////////////////////////////

struct __single_thread_group
{
  _LIBCUDACXX_HIDE_FROM_ABI void sync() const {}
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::size_t size() const
  {
    return 1;
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::size_t thread_rank() const
  {
    return 0;
  };
};

template <typename _Group, class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment __memcpy_async_barrier(
  _Group const& __group, _Tp* __destination, _Tp const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  static_assert(_CCCL_TRAIT(_CUDA_VSTD::is_trivially_copyable, _Tp), "memcpy_async requires a trivially copyable type");

  // 1. Determine which completion mechanisms can be used with the current
  // barrier. A local shared memory barrier, i.e., block-scope barrier in local
  // shared memory, supports the mbarrier_complete_tx mechanism in addition to
  // the async group mechanism.
  _CUDA_VSTD::uint32_t __allowed_completions =
    __is_local_smem_barrier(__barrier)
      ? (_CUDA_VSTD::uint32_t(__completion_mechanism::__async_group)
         | _CUDA_VSTD::uint32_t(__completion_mechanism::__mbarrier_complete_tx))
      : _CUDA_VSTD::uint32_t(__completion_mechanism::__async_group);

  // Alignment: Use the maximum of the alignment of _Tp and that of a possible cuda::aligned_size_t.
  constexpr _CUDA_VSTD::size_t __size_align = __get_size_align<_Size>::align;
  constexpr _CUDA_VSTD::size_t __align      = (alignof(_Tp) < __size_align) ? __size_align : alignof(_Tp);
  // Cast to char pointers. We don't need the type for alignment anymore and
  // erasing the types reduces the number of instantiations of down-stream
  // functions.
  char* __dest_char      = reinterpret_cast<char*>(__destination);
  char const* __src_char = reinterpret_cast<char const*>(__source);

  // 2. Issue actual copy instructions.
  auto __bh = __try_get_barrier_handle(__barrier);
  auto __cm = __dispatch_memcpy_async<__align>(__group, __dest_char, __src_char, __size, __allowed_completions, __bh);

  // 3. Synchronize barrier with copy instructions.
  return __memcpy_completion_impl::__defer(__cm, __group, __size, __barrier);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_BARRIER_H_
