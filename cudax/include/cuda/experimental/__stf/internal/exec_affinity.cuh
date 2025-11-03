//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Facilities to define the place affinity.
 *
 * Setting an affinity will help defining what is the default execution place.
 * They are used in combination with loop_dispatch to dispatch computation over
 * a set of execution places.
 *
 * \see loop_dispatch
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/core.cuh>

#include <cassert>
#include <memory>
#include <stack>
#include <vector>

namespace cuda::experimental::stf
{
class exec_place;

/**
 * @brief Defines the current execution places associated with a context.
 *
 * When an affinity is set, the first entry of the current execution places
 * defines the default execution place used in CUDASTF constructs when no
 * execution place is supplied.
 */
class exec_affinity
{
public:
  exec_affinity()  = default;
  ~exec_affinity() = default;

  /**
   * @brief Set the current affinity to a vector of execution places
   */
  void push(::std::vector<::std::shared_ptr<exec_place>> p)
  {
    s.push(mv(p));
  }

  /**
   * @brief Set the current affinity to a single execution place
   */
  void push(::std::shared_ptr<exec_place> p)
  {
    s.push(::std::vector<::std::shared_ptr<exec_place>>{::std::move(p)});
  }

  /**
   * @brief Restore the affinity to its value before calling push
   */
  void pop()
  {
    s.pop();
  }

  /**
   * @brief Indicates if an affinity was set or not
   */
  bool has_affinity() const
  {
    return !s.empty();
  }

  /**
   * @brief Get a reference to the vector of place pointers at the top of the stack (ie. thread's current affinity)
   */
  const auto& top() const
  {
    return s.top();
  }

private:
  // A stack per thread
  // (We use vectors of shared_ptr because exec_place implementation cannot
  // be available so we rely on type erasure)
  static thread_local ::std::stack<::std::vector<::std::shared_ptr<exec_place>>> s;
};

// Define the static thread_local member outside the class
inline thread_local ::std::stack<::std::vector<::std::shared_ptr<exec_place>>> exec_affinity::s;
} // end namespace cuda::experimental::stf
