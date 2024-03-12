// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_BARRIER_CLUSTER_H_
#define _CUDA_PTX_BARRIER_CLUSTER_H_

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include "../ptx_dot_variants.h"
#include "../ptx_helper_functions.h"
#include "../../../cstdint"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// 9.7.12.3. Parallel Synchronization and Communication Instructions: barrier.cluster
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
/*
// barrier.cluster.arrive; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.wait; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_wait()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.wait;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .release }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(
  sem_release_t)
{
  // __sem == sem_release (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive.release;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// Marked volatile
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(
  sem_relaxed_t)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive.relaxed;"
      :
      :
      :
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.wait.sem; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_wait(
  sem_acquire_t)
{
  // __sem == sem_acquire (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.wait.acquire;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_BARRIER_CLUSTER_H_
