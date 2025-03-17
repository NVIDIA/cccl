//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___BARRIER_POLL_TESTER_H
#define __LIBCUDACXX___BARRIER_POLL_TESTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Barrier>
class __barrier_poll_tester_phase
{
  _Barrier const* __this;
  typename _Barrier::arrival_token __phase;

public:
  _LIBCUDACXX_HIDE_FROM_ABI
  __barrier_poll_tester_phase(_Barrier const* __this_, typename _Barrier::arrival_token&& __phase_)
      : __this(__this_)
      , __phase(_CUDA_VSTD::move(__phase_))
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool operator()() const
  {
    return __this->__try_wait(__phase);
  }
};

template <class _Barrier>
class __barrier_poll_tester_parity
{
  _Barrier const* __this;
  bool __parity;

public:
  _LIBCUDACXX_HIDE_FROM_ABI __barrier_poll_tester_parity(_Barrier const* __this_, bool __parity_)
      : __this(__this_)
      , __parity(__parity_)
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool operator()() const
  {
    return __this->__try_wait_parity(__parity);
  }
};

template <class _Barrier>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
__call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase)
{
  return __b.__try_wait(_CUDA_VSTD::move(__phase));
}

template <class _Barrier>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __call_try_wait_parity(const _Barrier& __b, bool __parity)
{
  return __b.__try_wait_parity(__parity);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___BARRIER_POLL_TESTER_H
