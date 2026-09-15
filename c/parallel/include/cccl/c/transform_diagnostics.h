//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

// Returns a diagnostic from the most recent unary/binary transform compile or
// transform load call on this thread, or "" if no diagnostic is available.
// Build calls perform compile followed by load, and report the failing stage.
// The library owns the returned string. It remains valid until the next such
// call on the same thread. Compute and cleanup calls do not change it.
CCCL_C_API const char* cccl_transform_last_error(void);

CCCL_C_EXTERN_C_END
