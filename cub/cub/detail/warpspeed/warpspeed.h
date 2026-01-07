// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/allocators/smem_allocator.h>
#include <cub/detail/warpspeed/constant_assert.h>
#include <cub/detail/warpspeed/make_warp_uniform.cuh>
#include <cub/detail/warpspeed/special_registers.cuh>
#include <cub/detail/warpspeed/sync_handler.h>
#include <cub/detail/warpspeed/values.h>

// TODO: rename .cuh headers to .h
#include <cub/detail/warpspeed/resource/smem_phase.cuh>
#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/resource/smem_resource.cuh>
#include <cub/detail/warpspeed/resource/smem_resource_raw.cuh>
#include <cub/detail/warpspeed/resource/smem_stage.cuh>
#include <cub/detail/warpspeed/squad/squad.h>
#include <cub/detail/warpspeed/squad/squad_desc.h>
