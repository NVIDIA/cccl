//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Host-side target identity used by libnvcc's cc1 and lld invocations. The
// device side is always NVPTX; the host side follows the machine libnvcc was
// built for. Everything here is a compile-time constant so a freestanding
// test can check it without loading LLVM.
//
// HOSTJIT_HOST_TRIPLE   clang -triple / -aux-triple for the host
// HOSTJIT_HOST_CPU      clang -target-cpu / -aux-target-cpu for the host
// HOSTJIT_LLD_EMULATION ld.lld -m value for the host object format
// HOSTJIT_MULTIARCH     Debian/Ubuntu multiarch directory name under
//                       /usr/include (libc keeps bits/ there; cc1 does not
//                       add it on its own, the clang driver normally does)

#if defined(_WIN32)
#  define HOSTJIT_HOST_TRIPLE   "x86_64-pc-windows-msvc"
#  define HOSTJIT_HOST_CPU      "x86-64"
#  define HOSTJIT_LLD_EMULATION "i386pep"
#  define HOSTJIT_MULTIARCH     ""
#elif defined(__aarch64__)
#  define HOSTJIT_HOST_TRIPLE   "aarch64-unknown-linux-gnu"
#  define HOSTJIT_HOST_CPU      "generic"
#  define HOSTJIT_LLD_EMULATION "aarch64linux"
#  define HOSTJIT_MULTIARCH     "aarch64-linux-gnu"
#elif defined(__x86_64__)
#  define HOSTJIT_HOST_TRIPLE   "x86_64-pc-linux-gnu"
#  define HOSTJIT_HOST_CPU      "x86-64"
#  define HOSTJIT_LLD_EMULATION "elf_x86_64"
#  define HOSTJIT_MULTIARCH     "x86_64-linux-gnu"
#else
#  error "libnvcc: unsupported host architecture; add it to host_target.h"
#endif
