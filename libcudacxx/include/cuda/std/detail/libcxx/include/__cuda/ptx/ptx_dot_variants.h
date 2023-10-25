// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//


#ifndef _CUDA_PTX_DOT_VARIANTS_H_
#define _CUDA_PTX_DOT_VARIANTS_H_

#include "../../type_traits" // std::integral_constant

/*
 * Public integral constant types and values for ".variant"s:
 *
 * - .sem
 * - .space
 * - .scope
 *
 * For each .variant, the code below defines:
 * - An enum `dot_variant` with each possible value
 * - A type template `variant_t<dot_variant>`
 * - Types `variant_A_t`, ..., `variant_Z_t`
 * - Constexpr values `variant_A` of type `variant_A_t`
 *
 * These types enable specifying fine-grained overloads of a PTX binding. If a
 * binding can handle multiple variants, then it is defined as:
 *
 * template <dot_variant var>
 * [...] void ptx_binding(variant_t<var> __v) { ... }
 *
 * If it only handles a single variant, then it is defined as:
 *
 * [...] void ptx_binding(variant_A __v) { ... }
 *
 * If two variants have different behaviors or return types (see .space
 * overloads of mbarrier.arrive.expect_tx for an example), then these can be
 * provided as separate overloads of the same function:
 *
 * [...] void ptx_binding(variant_A __v) { ... }
 * [...] int ptx_binding(variant_B __v) { ... }
 *
 */

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operation-types
enum class dot_sem
{
  acq_rel,
  acquire,
  relaxed,
  release,
  sc,
  weak
  // mmio?
  // volatile?
};

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces
enum class dot_space
{
  reg,
  sreg,
  const_mem, // Using const_mem as `const` is reserved in C++.
  global,
  local,
  param,
  shared, // The PTX spelling is shared::cta
  shared_cluster, // The PTX spelling is shared::cluster, but we might want to go for cluster here.
  tex // deprecated
  // generic?
};

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope
enum class dot_scope
{
  cta,
  cluster,
  gpu,
  sys
};

template <dot_sem sem>
using sem_t         = _CUDA_VSTD::integral_constant<dot_sem, sem>;
using sem_acq_rel_t = sem_t<dot_sem::acq_rel>;
using sem_acquire_t = sem_t<dot_sem::acquire>;
using sem_relaxed_t = sem_t<dot_sem::relaxed>;
using sem_release_t = sem_t<dot_sem::release>;
using sem_sc_t      = sem_t<dot_sem::sc>;
using sem_weak_t    = sem_t<dot_sem::weak>;

static constexpr sem_acq_rel_t sem_acq_rel{};
static constexpr sem_acquire_t sem_acquire{};
static constexpr sem_relaxed_t sem_relaxed{};
static constexpr sem_release_t sem_release{};
static constexpr sem_sc_t sem_sc{};
static constexpr sem_weak_t sem_weak{};

template <dot_space spc>
using space_t                = _CUDA_VSTD::integral_constant<dot_space, spc>;
using space_const_mem_t      = space_t<dot_space::const_mem>;
using space_global_t         = space_t<dot_space::global>;
using space_local_t          = space_t<dot_space::local>;
using space_param_t          = space_t<dot_space::param>;
using space_reg_t            = space_t<dot_space::reg>;
using space_shared_t         = space_t<dot_space::shared>;
using space_shared_cluster_t = space_t<dot_space::shared_cluster>;
using space_sreg_t           = space_t<dot_space::sreg>;
using space_tex_t            = space_t<dot_space::tex>;

static constexpr space_const_mem_t space_const_mem{};
static constexpr space_global_t space_global{};
static constexpr space_local_t space_local{};
static constexpr space_param_t space_param{};
static constexpr space_reg_t space_reg{};
static constexpr space_shared_t space_shared{};
static constexpr space_shared_cluster_t space_shared_cluster{};
static constexpr space_sreg_t space_sreg{};
static constexpr space_tex_t space_tex{};

template <dot_scope scope>
using scope_t         = _CUDA_VSTD::integral_constant<dot_scope, scope>;
using scope_cluster_t = scope_t<dot_scope::cluster>;
using scope_cta_t     = scope_t<dot_scope::cta>;
using scope_gpu_t     = scope_t<dot_scope::gpu>;
using scope_sys_t     = scope_t<dot_scope::sys>;

static constexpr scope_cluster_t scope_cluster{};
static constexpr scope_cta_t scope_cta{};
static constexpr scope_gpu_t scope_gpu{};
static constexpr scope_sys_t scope_sys{};

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_DOT_VARIANTS_H_
