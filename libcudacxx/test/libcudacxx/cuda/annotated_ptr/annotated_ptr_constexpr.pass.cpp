//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/annotated_ptr>

#include "test_macros.h"

__host__ __device__ constexpr bool test_public_methods()
{
  using namespace cuda;
  using annotated_ptr                       = cuda::annotated_ptr<const int, access_property::persisting>;
  using annotated_smem_ptr [[maybe_unused]] = cuda::annotated_ptr<const int, access_property::shared>;
  using annotated_ptr_b                     = cuda::annotated_ptr<int, access_property::persisting>;
  annotated_ptr a{}; // default constructor
  annotated_ptr b{a}; // copy constructor
  annotated_ptr c{cuda::std::move(a)}; // move constructor
  b = a; // copy assignment
  b = cuda::std::move(a); // move assignment
  NV_IF_TARGET(NV_IS_DEVICE, (annotated_smem_ptr d{nullptr};)) // pointer constructor
  annotated_ptr e{annotated_ptr_b{}}; // constructor from another annotated_ptr
  auto ptr  = e.operator->();
  auto diff = a - b;
  auto pred = static_cast<bool>(a);
  auto prop = a.__property();
  unused(c);
  unused(ptr);
  unused(diff);
  unused(pred);
  unused(prop);
  return true;
}

__host__ __device__ constexpr bool test_interleave_values()
{
  using namespace cuda;
  constexpr auto normal = __l2_interleave(__l2_evict_t::_L2_Evict_Unchanged, __l2_evict_t::_L2_Evict_Unchanged, 1.0f);
  constexpr auto streaming  = __l2_interleave(__l2_evict_t::_L2_Evict_First, __l2_evict_t::_L2_Evict_Unchanged, 1.0f);
  constexpr auto persisting = __l2_interleave(__l2_evict_t::_L2_Evict_Last, __l2_evict_t::_L2_Evict_Unchanged, 1.0f);
  constexpr auto normal_demote =
    __l2_interleave(__l2_evict_t::_L2_Evict_Normal_Demote, __l2_evict_t::_L2_Evict_Unchanged, 1.0f);
  static_assert(normal == __l2_interleave_normal);
  static_assert(streaming == __l2_interleave_streaming);
  static_assert(persisting == __l2_interleave_persisting);
  static_assert(normal_demote == __l2_interleave_normal_demote);
  return true;
}

int main(int, char**)
{
  static_assert(test_interleave_values());
  static_assert(test_public_methods());
  return 0;
}
