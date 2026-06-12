// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single source of truth for the compile-time gates the tile transform headers
// share. Two macros:
//
//   _CCCL_CUB_HAS_TILE_TRANSFORM()
//     True when nvcc is compiling in tile mode (--enable-tile, i.e.
//     _CCCL_TILE_COMPILATION()) AND the toolkit is CTK 13.4+. tile C++ exists
//     since 13.3, but we require 13.4: the 13.3 tile compiler has too many
//     codegen issues, so 13.4 is the supported floor. (C++20 is enforced by
//     cuda_tile.h itself with an explicit #error.) The sm_80+ requirement is
//     handled at runtime in the dispatch + NV_IF_TARGET in the kernels, not
//     here, since this gate is host+device. When false, the tile headers
//     (kernel / tuning / dispatch / traits) are skipped entirely.
//
//   _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
//     True when the dispatch hook in cub::DeviceTransform should fire. Same as
//     _CCCL_CUB_HAS_TILE_TRANSFORM() plus the user opt-in macro
//     CCCL_ENABLE_TILE_TRANSFORM_DISPATCH.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _CCCL_CUB_HAS_TILE_TRANSFORM() (_CCCL_TILE_COMPILATION() && _CCCL_CTK_AT_LEAST(13, 4))

#if _CCCL_CUB_HAS_TILE_TRANSFORM() && defined(CCCL_ENABLE_TILE_TRANSFORM_DISPATCH)
#  define _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED() 1
#else
#  define _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED() 0
#endif
