//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nvbench/nvbench.cuh>

#if _CCCL_HAS_INT128()
NVBENCH_DECLARE_TYPE_STRINGS(__int128_t, "I128", "__int128_t");
#endif // _CCCL_HAS_INT128()
