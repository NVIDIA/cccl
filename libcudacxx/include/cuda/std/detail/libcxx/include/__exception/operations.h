// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___EXCEPTION_OPERATIONS_H
#define _LIBCUDACXX___EXCEPTION_OPERATIONS_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__availability"
#include "../cstddef"
#include "../cstdlib"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

#if _LIBCUDACXX_STD_VER <= 14 \
    || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS) \
    || defined(_LIBCUDACXX_BUILDING_LIBRARY)
typedef void (*unexpected_handler)();
_LIBCUDACXX_FUNC_VIS unexpected_handler set_unexpected(unexpected_handler) _NOEXCEPT;
_LIBCUDACXX_FUNC_VIS unexpected_handler get_unexpected() _NOEXCEPT;
_LIBCUDACXX_NORETURN _LIBCUDACXX_FUNC_VIS void unexpected();
#endif

typedef void (*terminate_handler)();
_LIBCUDACXX_FUNC_VIS terminate_handler set_terminate(terminate_handler) _NOEXCEPT;
_LIBCUDACXX_FUNC_VIS terminate_handler get_terminate() _NOEXCEPT;

_LIBCUDACXX_FUNC_VIS bool uncaught_exception() _NOEXCEPT;
_LIBCUDACXX_FUNC_VIS _LIBCUDACXX_AVAILABILITY_UNCAUGHT_EXCEPTIONS int uncaught_exceptions() _NOEXCEPT;

class _LIBCUDACXX_TYPE_VIS exception_ptr;

_LIBCUDACXX_FUNC_VIS exception_ptr current_exception() _NOEXCEPT;
_LIBCUDACXX_NORETURN _LIBCUDACXX_FUNC_VIS void rethrow_exception(exception_ptr);

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___EXCEPTION_OPERATIONS_H
