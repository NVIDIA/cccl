//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Implementation of the host_launch construct
 *
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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>
#include <cuda/experimental/__stf/internal/thread_hierarchy.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>

namespace cuda::experimental::stf
{

class graph_ctx;
class stream_ctx;

namespace reserved
{

/**
 * @brief Result of `host_launch` (below)
 *
 * @tparam Deps Types of dependencies
 *
 * @see `host_launch`
 */
template <typename Ctx, bool called_from_launch, typename... Deps>
class host_launch_scope
{
public:
  host_launch_scope(Ctx& ctx, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
  {}

  host_launch_scope(const host_launch_scope&)            = delete;
  host_launch_scope& operator=(const host_launch_scope&) = delete;
  // move-constructible
  host_launch_scope(host_launch_scope&&) = default;

  /**
   * @brief Sets the symbol for this object.
   *
   * This method moves the provided string into the internal symbol member and returns a reference to the current
   * object, allowing for method chaining.
   *
   * @param s The string to set as the symbol.
   * @return A reference to the current object.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Takes a lambda function and executes it on the host in a graph callback node.
   *
   * @tparam Fun type of lambda function
   * @param f Lambda function to execute
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    auto t = ctx.task(exec_place::host());
    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    cudaEvent_t start_event, end_event;
    const bool record_time = t.schedule_task() || statistics.is_calibrating_to_file();

    t.start();

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    SCOPE(exit)
    {
      t.end_uncleared();
      if constexpr (::std::is_same_v<Ctx, stream_ctx>)
      {
        if (record_time)
        {
          cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
          cuda_safe_call(cudaEventSynchronize(end_event));

          float milliseconds = 0;
          cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

          if (dot.is_tracing())
          {
            dot.template add_vertex_timing<typename Ctx::task_type>(t, milliseconds, -1);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }
      t.clear();
    };

    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    auto payload = [&]() {
      if constexpr (called_from_launch)
      {
        return tuple_prepend(thread_hierarchy<>(), deps.instance(t));
      }
      else
      {
        return deps.instance(t);
      }
    }();
    auto* wrapper = new ::std::pair<Fun, decltype(payload)>{::std::forward<Fun>(f), mv(payload)};

    auto callback = [](void* untyped_wrapper) {
      auto w = static_cast<decltype(wrapper)>(untyped_wrapper);
      SCOPE(exit)
      {
        delete w;
      };

      constexpr bool fun_invocable_task_deps = reserved::is_applicable_v<Fun, decltype(payload)>;
      constexpr bool fun_invocable_task_non_void_deps =
        reserved::is_applicable_v<Fun, remove_void_interface_t<decltype(payload)>>;

      static_assert(fun_invocable_task_deps || fun_invocable_task_non_void_deps,
                    "Incorrect lambda function signature in host_launch.");

      if constexpr (fun_invocable_task_deps)
      {
        ::std::apply(::std::forward<Fun>(w->first), mv(w->second));
      }
      else if constexpr (fun_invocable_task_non_void_deps)
      {
        ::std::apply(::std::forward<Fun>(w->first), reserved::remove_void_interface(mv(w->second)));
      }
    };

    if constexpr (::std::is_same_v<Ctx, graph_ctx>)
    {
      cudaHostNodeParams params = {.fn = callback, .userData = wrapper};

      // Put this host node into the child graph that implements the graph_task<>
      auto lock = t.lock_ctx_graph();
      cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
    }
    else
    {
      cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), callback, wrapper));
    }
  }

private:
  ::std::string symbol;
  Ctx& ctx;
  task_dep_vector<Deps...> deps;
};

} // end namespace reserved

} // end namespace cuda::experimental::stf
