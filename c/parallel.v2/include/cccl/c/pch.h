//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <stddef.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

// Write the resolved PCH cache directory into `out` as a NUL-terminated string.
//
// The location depends on a resolution chain (CCCL_PCH_CACHE_DIR, then
// XDG_CACHE_HOME, then the user's home cache, then a uid-scoped temp
// directory), so callers that want to inspect or clear the cache must ask
// rather than reconstruct it.
//
// Returns the buffer size required including the terminator, in the manner of
// snprintf: if the return value is greater than `out_size` the path was
// truncated and the call should be repeated with a larger buffer. Passing
// out_size == 0 (with out == NULL) queries the required size.
//
// Returns 0 if no usable cache directory could be resolved, in which case PCH
// is inactive and there is nothing to inspect or clear.
CCCL_C_API size_t cccl_hostjit_pch_cache_dir(char* out, size_t out_size);

CCCL_C_EXTERN_C_END

// NOLINTEND(modernize-use-using)
