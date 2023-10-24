//===--------------------- mutex_destructor.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Define ~mutex.
//
// On some platforms ~mutex has been made trivial and the definition is only
// provided for ABI compatibility.
//
// In order to avoid ODR violations within libc++ itself, we need to ensure
// that *nothing* sees the non-trivial mutex declaration. For this reason
// we re-declare the entire class in this file instead of using
// _LIBCUDACXX_BUILDING_LIBRARY to change the definition in the headers.

#include "__config"
#include "__threading_support"

#if !defined(_LIBCUDACXX_HAS_NO_THREADS)
#if _LIBCUDACXX_ABI_VERSION == 1 || !defined(_LIBCUDACXX_HAS_TRIVIAL_MUTEX_DESTRUCTION)
#define NEEDS_MUTEX_DESTRUCTOR
#endif
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifdef NEEDS_MUTEX_DESTRUCTOR
class _LIBCUDACXX_TYPE_VIS mutex
{
    __libcpp_mutex_t __m_ = _LIBCUDACXX_MUTEX_INITIALIZER;

public:
    _LIBCUDACXX_ALWAYS_INLINE _LIBCUDACXX_INLINE_VISIBILITY
    constexpr mutex() = default;
    mutex(const mutex&) = delete;
    mutex& operator=(const mutex&) = delete;
    ~mutex() noexcept;
};


mutex::~mutex() noexcept
{
    __libcpp_mutex_destroy(&__m_);
}

#endif // !_LIBCUDACXX_HAS_NO_THREADS
_LIBCUDACXX_END_NAMESPACE_STD

