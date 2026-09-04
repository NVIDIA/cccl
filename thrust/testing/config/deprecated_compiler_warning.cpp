// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verifies that the deprecated host compiler warning in
// thrust/detail/config/config.h actually reaches the user.
//
// The warning is emitted through `#pragma GCC warning`, which GCC and Clang
// silently drop from system headers, so config.h has to issue it before marking
// itself as one. See NVIDIA/cccl#1620.
//
// The compiler detection is faked below so that the warning is tested on
// every configuration, and not only on the deprecated compilers themselves,
// which CI does not build with.

// CCCL builds its own tests with _CCCL_NO_SYSTEM_HEADER, which prevents its
// headers from being treated as system headers. Here, we have to drop that again;
// the warning is only at risk of being dropped when the header is a system header,
// which is true when the users compile it.
#undef _CCCL_NO_SYSTEM_HEADER

#include <cuda/__cccl_config>

#undef _CCCL_COMPILER_GCC
#define _CCCL_COMPILER_GCC() (6, 0)

#include <thrust/detail/config.h>

int main()
{
  return 0;
}
