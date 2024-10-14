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

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/places/places.cuh>

namespace cuda::experimental::stf
{
class logical_data_untyped;

/**
 * TODO : Nice description of the class comes here.
 */
class reduction_operator_base
{
public:
  // For now, we ignore commutativity !
  reduction_operator_base(bool /* is_commutative */) {}

  virtual ~reduction_operator_base() {}

  reduction_operator_base& operator=(const reduction_operator_base&) = delete;
  reduction_operator_base(const reduction_operator_base&)            = delete;

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

private:
  // not used for now ...
  // bool is_commutative;
};

} // end namespace cuda::experimental::stf
