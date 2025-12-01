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

#include <cub/device/dispatch/kernels/warpspeed/allocators/SmemAllocator.h>
#include <cub/device/dispatch/kernels/warpspeed/constantAssert.h>
#include <cub/device/dispatch/kernels/warpspeed/makeWarpUniform.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SyncHandler.h>
#include <cub/device/dispatch/kernels/warpspeed/values.h>

// TODO: rename .cuh headers to .h
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemPhase.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemRef.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemResource.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemResourceRaw.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemStage.cuh>
#include <cub/device/dispatch/kernels/warpspeed/squad/Squad.h>
#include <cub/device/dispatch/kernels/warpspeed/squad/SquadDesc.h>
