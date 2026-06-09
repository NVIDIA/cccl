// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single source of truth for the compile-time gates the tile transform headers
// share. Two macros:
//
//   _CCCL_CUB_HAS_TILE_TRANSFORM()
//     True when CUB's tile transform machinery is available: CTK 13.3 or newer,
//     C++20 (tile DSL requires it), and the tile compilation trajectory
//     (--enable-tile). When false, the tile headers (kernel / tuning / dispatch
//     / traits) are skipped entirely.
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

#define _CCCL_CUB_HAS_TILE_TRANSFORM() \
  (_CCCL_CTK_AT_LEAST(13, 3) && _CCCL_STD_VER >= 2020 && _CCCL_TILE_COMPILATION())

#define _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED() \
  (_CCCL_CUB_HAS_TILE_TRANSFORM() && defined(CCCL_ENABLE_TILE_TRANSFORM_DISPATCH))
