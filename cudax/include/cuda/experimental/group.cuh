//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_GROUP
#define _CUDA_EXPERIMENTAL_GROUP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/group.cuh>
#include <cuda/experimental/__group/implicit_hierarchy.cuh>
#include <cuda/experimental/__group/mapping/group_as.cuh>
#include <cuda/experimental/__group/mapping/group_by.cuh>
#include <cuda/experimental/__group/queries.cuh>
#include <cuda/experimental/__group/synchronizer/barrier_synchronizer.cuh>
#include <cuda/experimental/__group/synchronizer/lane_synchronizer.cuh>
#include <cuda/experimental/__group/this_group.cuh>
#include <cuda/experimental/__group/traits.cuh>

#endif // _CUDA_EXPERIMENTAL_GROUP
