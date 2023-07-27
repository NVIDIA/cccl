//===----------------------- functional.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "functional"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifdef _LIBCUDACXX_ABI_BAD_FUNCTION_CALL_KEY_FUNCTION
bad_function_call::~bad_function_call() _NOEXCEPT
{
}

const char*
bad_function_call::what() const _NOEXCEPT
{
    return "std::bad_function_call";
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD
