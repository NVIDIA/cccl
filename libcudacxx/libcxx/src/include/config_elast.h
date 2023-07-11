//===----------------------- config_elast.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_CONFIG_ELAST
#define _LIBCUDACXX_CONFIG_ELAST

#include <__config>

#if defined(_LIBCUDACXX_MSVCRT_LIKE)
#include <stdlib.h>
#else
#include <errno.h>
#endif

#if defined(ELAST)
#define _LIBCUDACXX_ELAST ELAST
#elif defined(_NEWLIB_VERSION)
#define _LIBCUDACXX_ELAST __ELASTERROR
#elif defined(__Fuchsia__)
// No _LIBCUDACXX_ELAST needed on Fuchsia
#elif defined(__wasi__)
// No _LIBCUDACXX_ELAST needed on WASI
#elif defined(__linux__) || defined(_LIBCUDACXX_HAS_MUSL_LIBC)
#define _LIBCUDACXX_ELAST 4095
#elif defined(__APPLE__)
// No _LIBCUDACXX_ELAST needed on Apple
#elif defined(__sun__)
#define _LIBCUDACXX_ELAST ESTALE
#elif defined(_LIBCUDACXX_MSVCRT_LIKE)
#define _LIBCUDACXX_ELAST (_sys_nerr - 1)
#else
// Warn here so that the person doing the libcxx port has an easier time:
#warning ELAST for this platform not yet implemented
#endif

#endif // _LIBCUDACXX_CONFIG_ELAST
