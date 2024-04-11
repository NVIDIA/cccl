//===----------------------------------------------------------------------===##
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===##

extern "C" int printf(char const* format, ...);

int main()
{
  char const* compiler_type = "unknown";
  unsigned major_version    = 0;
  unsigned minor_version    = 0;
  unsigned patch_level      = 0;
  unsigned default_dialect  = 3;

#if defined(__NVCC__)
  compiler_type = "nvcc";
  major_version = __CUDACC_VER_MAJOR__;
  minor_version = __CUDACC_VER_MINOR__;
  patch_level   = __CUDACC_VER_BUILD__;
#elif defined(__NVCOMPILER)
  compiler_type = "nvhpc";
  major_version = __NVCOMPILER;
  minor_version = ___NVCOMPILER_MINOR__;
  patch_level   = ___NVCOMPILER_PATCHLEVEL__;
#elif defined(__INTEL_COMPILER)
  compiler_type = "icc";
  major_version = __INTEL_COMPILER / 100;
  minor_version = (__INTEL_COMPILER % 100) / 10;
  patch_level   = __INTEL_COMPILER % 10;
#elif defined(__clang__)
// Treat Apple's LLVM fork differently.
#  if defined(__apple_build_version__)
  compiler_type = "apple-clang";
#  else
  compiler_type = "clang";
#  endif
  major_version = __clang_major__;
  minor_version = __clang_minor__;
  patch_level   = __clang_patchlevel__;
#elif defined(_MSC_VER)
  compiler_type = "msvc";
  major_version = _MSC_FULL_VER / 10000000;
  minor_version = _MSC_FULL_VER / 100000 % 100;
  patch_level   = _MSC_FULL_VER % 100000;
#elif defined(__GNUC__)
  compiler_type = "gcc";
  major_version = __GNUC__;
  minor_version = __GNUC_MINOR__;
  patch_level   = __GNUC_PATCHLEVEL__;
#endif

#if defined(_MSC_VER)
#  if !defined(_MSVC_LANG)
  default_dialect = 3;
#  elif _MSVC_LANG <= 201103L
  default_dialect = 11;
#  elif _MSVC_LANG <= 201402L
  default_dialect = 14;
#  elif _MSVC_LANG <= 201703L
  default_dialect = 17;
#  else
  default_dialect = 20;
#  endif
#else
#  if __cplusplus <= 199711L
  default_dialect = 3;
#  elif __cplusplus <= 201103L
  default_dialect = 11;
#  elif __cplusplus <= 201402L
  default_dialect = 14;
#  elif __cplusplus <= 201703L
  default_dialect = 17;
#  else
  default_dialect = 20;
#  endif
#endif

  printf(
    "(\"%s\", (%d, %d, %d), \"c++%02u\")\n", compiler_type, major_version, minor_version, patch_level, default_dialect);
}
