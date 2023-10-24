// -*- C++ -*-
//===--------------- support/xlocale/__posix_l_fallback.h -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These are reimplementations of some extended locale functions ( *_l ) that
// are normally part of POSIX.  This shared implementation provides parts of the
// extended locale support for libc's that normally don't have any (like
// Android's bionic and Newlib).
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H
#define _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H

#ifdef __cplusplus
extern "C" {
#endif

_LIBCUDACXX_HIDE_FROM_ABI int isalnum_l(int c, locale_t) {
  return ::isalnum(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isalpha_l(int c, locale_t) {
  return ::isalpha(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isblank_l(int c, locale_t) {
  return ::isblank(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iscntrl_l(int c, locale_t) {
  return ::iscntrl(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isdigit_l(int c, locale_t) {
  return ::isdigit(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isgraph_l(int c, locale_t) {
  return ::isgraph(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int islower_l(int c, locale_t) {
  return ::islower(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isprint_l(int c, locale_t) {
  return ::isprint(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int ispunct_l(int c, locale_t) {
  return ::ispunct(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isspace_l(int c, locale_t) {
  return ::isspace(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isupper_l(int c, locale_t) {
  return ::isupper(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int isxdigit_l(int c, locale_t) {
  return ::isxdigit(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswalnum_l(wint_t c, locale_t) {
  return ::iswalnum(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswalpha_l(wint_t c, locale_t) {
  return ::iswalpha(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswblank_l(wint_t c, locale_t) {
  return ::iswblank(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswcntrl_l(wint_t c, locale_t) {
  return ::iswcntrl(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswdigit_l(wint_t c, locale_t) {
  return ::iswdigit(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswgraph_l(wint_t c, locale_t) {
  return ::iswgraph(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswlower_l(wint_t c, locale_t) {
  return ::iswlower(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswprint_l(wint_t c, locale_t) {
  return ::iswprint(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswpunct_l(wint_t c, locale_t) {
  return ::iswpunct(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswspace_l(wint_t c, locale_t) {
  return ::iswspace(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswupper_l(wint_t c, locale_t) {
  return ::iswupper(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int iswxdigit_l(wint_t c, locale_t) {
  return ::iswxdigit(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int toupper_l(int c, locale_t) {
  return ::toupper(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int tolower_l(int c, locale_t) {
  return ::tolower(c);
}

_LIBCUDACXX_HIDE_FROM_ABI wint_t towupper_l(wint_t c, locale_t) {
  return ::towupper(c);
}

_LIBCUDACXX_HIDE_FROM_ABI wint_t towlower_l(wint_t c, locale_t) {
  return ::towlower(c);
}

_LIBCUDACXX_HIDE_FROM_ABI int strcoll_l(const char *s1, const char *s2,
                                               locale_t) {
  return ::strcoll(s1, s2);
}

_LIBCUDACXX_HIDE_FROM_ABI size_t strxfrm_l(char *dest, const char *src,
                                                  size_t n, locale_t) {
  return ::strxfrm(dest, src, n);
}

_LIBCUDACXX_HIDE_FROM_ABI size_t strftime_l(char *s, size_t max,
                                                   const char *format,
                                                   const struct tm *tm, locale_t) {
  return ::strftime(s, max, format, tm);
}

_LIBCUDACXX_HIDE_FROM_ABI int wcscoll_l(const wchar_t *ws1,
                                               const wchar_t *ws2, locale_t) {
  return ::wcscoll(ws1, ws2);
}

_LIBCUDACXX_HIDE_FROM_ABI size_t wcsxfrm_l(wchar_t *dest, const wchar_t *src,
                                                  size_t n, locale_t) {
  return ::wcsxfrm(dest, src, n);
}

#ifdef __cplusplus
}
#endif

#endif // _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H
