//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_API_ATOMIC_CRTP_H
#define __LIBCUDACXX___ATOMIC_API_ATOMIC_CRTP_H

#include <cuda/std/detail/__config>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __atomic_crtp_accessor defines a way to statically fetch the atomic storage object
// which owns the stored atomic.
template <typename _Impl, typename _Sto>
struct __atomic_crtp_accessor {
    _CCCL_HOST_DEVICE
    inline auto __this_atom() -> _Sto* {
        return static_cast<_Impl*>(this)->__get_atom();
    }
    _CCCL_HOST_DEVICE
    inline auto __this_atom() const -> const _Sto* {
        return static_cast<const _Impl*>(this)->__get_atom();
    }
        _CCCL_HOST_DEVICE
    inline auto __this_atom() volatile -> volatile _Sto* {
        return static_cast<volatile _Impl*>(this)->__get_atom();
    }
        _CCCL_HOST_DEVICE
    inline auto __this_atom() const volatile -> const volatile _Sto* {
        return static_cast<const volatile _Impl*>(this)->__get_atom();
    }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif __LIBCUDACXX___ATOMIC_API_ATOMIC_CRTP_H
