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

#include <cuda/experimental/__stf/utility/core.cuh>

#include <cassert>
#include <variant>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief Repeats a blocks of code for a fixed number of steps
 *
 * This construct aims at allowing CUDASTF to automate some optimizations such
 * as introducing new stages, or doing some scheduling work.
 */
template <typename context_t>
class repeat_scope
{
public:
  static constexpr size_t tasks_per_stage = 200;

  repeat_scope(context_t& ctx, size_t count)
      : condition(count)
      , ctx(ctx)
  {}
  repeat_scope(context_t& ctx, ::std::function<bool()> condition)
      : condition(mv(condition))
      , ctx(ctx)
  {}

  template <typename Fun>
  void operator->*(Fun&& f)
  {
    size_t task_cnt = 0;
    for (size_t iter = 0; next(); ++iter)
    {
      size_t before_cnt = ctx.task_count();

      static_assert(::std::is_invocable_v<Fun, context_t, size_t>, "Incorrect lambda function signature.");

      f(ctx, iter);

      size_t after_cnt = ctx.task_count();
      assert(after_cnt >= before_cnt);

      // If there is more than a specific number of tasks, fire a new stage !
      task_cnt += after_cnt - before_cnt;
      if (task_cnt > tasks_per_stage)
      {
        ctx.change_stage();
        task_cnt = 0;
      }
    }
  }

private:
  bool next()
  {
    return condition.index() == 0
           ? ::std::get<size_t>(condition)-- > 0
           : ::std::get<::std::function<bool()>>(condition)();
  }

  // Number of iterations, or a function which evaluates if we continue
  ::std::variant<size_t, ::std::function<bool()>> condition;
  // The supporting context for this construct
  context_t& ctx;
};
} // end namespace cuda::experimental::stf::reserved
