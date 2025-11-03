//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/places/places.cuh>

namespace cuda::experimental::stf
{
class logical_data_untyped;

/**
 * @brief Interface of the reduction operators (how to initialize data, how to
 *        apply the reduction)
 */
class reduction_operator_base
{
public:
  // For now, we ignore commutativity !
  reduction_operator_base(bool /* is_commutative */) {}

  virtual ~reduction_operator_base() {}

  reduction_operator_base& operator=(const reduction_operator_base&) = delete;
  reduction_operator_base(const reduction_operator_base&)            = delete;

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen fails here

  // Reduction operator (inout, in)
  virtual void op_untyped(
    logical_data_untyped& d,
    const data_place& inout_memory_node,
    instance_id_t inout_instance_id,
    const data_place& in_memory_node,
    instance_id_t in_instance_id,
    const exec_place& e,
    event_list& prereq_in) = 0;

  // Initialization operator
  virtual void init_op_untyped(
    logical_data_untyped& d,
    const data_place& out_memory_node,
    instance_id_t out_instance_id,
    const exec_place& e,
    event_list& prereq_in) = 0;

#endif // _CCCL_DOXYGEN_INVOKED

private:
  // not used for now ...
  // bool is_commutative;
};
} // end namespace cuda::experimental::stf
