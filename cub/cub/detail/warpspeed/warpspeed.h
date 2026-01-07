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

#include <cub/detail/warpspeed/allocators/SmemAllocator.h>
#include <cub/detail/warpspeed/constantAssert.h>
#include <cub/detail/warpspeed/makeWarpUniform.cuh>
#include <cub/detail/warpspeed/SpecialRegisters.cuh>
#include <cub/detail/warpspeed/SyncHandler.h>
#include <cub/detail/warpspeed/values.h>

// TODO: rename .cuh headers to .h
#include <cub/detail/warpspeed/resource/SmemPhase.cuh>
#include <cub/detail/warpspeed/resource/SmemRef.cuh>
#include <cub/detail/warpspeed/resource/SmemResource.cuh>
#include <cub/detail/warpspeed/resource/SmemResourceRaw.cuh>
#include <cub/detail/warpspeed/resource/SmemStage.cuh>
#include <cub/detail/warpspeed/squad/Squad.h>
#include <cub/detail/warpspeed/squad/SquadDesc.h>
