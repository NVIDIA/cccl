//===---------------- condition_variable_destructor.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Define ~condition_variable.
//
// On some platforms ~condition_variable has been made trivial and the
// definition is only provided for ABI compatibility.

#include "__config"
#include "__threading_support"

#if !defined(_LIBCUDACXX_HAS_NO_THREADS)
# if _LIBCUDACXX_ABI_VERSION == 1 || !defined(_LIBCUDACXX_HAS_TRIVIAL_CONDVAR_DESTRUCTION)
#   define NEEDS_CONDVAR_DESTRUCTOR
# endif
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifdef NEEDS_CONDVAR_DESTRUCTOR

class _LIBCUDACXX_TYPE_VIS condition_variable
{
    __libcpp_condvar_t __cv_ = _LIBCUDACXX_CONDVAR_INITIALIZER;
public:
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr condition_variable() noexcept = default;

    ~condition_variable();

    condition_variable(const condition_variable&) = delete;
    condition_variable& operator=(const condition_variable&)  = delete;
};

condition_variable::~condition_variable()
{
    __libcpp_condvar_destroy(&__cv_);
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD
