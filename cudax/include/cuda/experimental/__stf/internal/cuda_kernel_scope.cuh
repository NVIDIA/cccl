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
 *
 * @brief Implementation of the cuda_kernel and cuda_kernel_chain constructs
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

namespace cuda::experimental::stf
{

class graph_ctx;
class stream_ctx;

/**
 * @brief Description of a CUDA kernel
 *
 * This is used to describe kernels passed to the `ctx.cuda_kernel` and
 * `ctx.cuda_kernel_chain` API calls.
 */
struct cuda_kernel_desc
{
  template <typename Fun, typename... Args>
  cuda_kernel_desc(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, Args... args)
      : func((const void*) func)
      , gridDim(gridDim_)
      , blockDim(blockDim_)
      , sharedMem(sharedMem_)
  {
    using TupleType = ::std::tuple<::std::decay_t<Args>...>;

    // We first copy all arguments into a tuple because the kernel
    // implementation needs pointers to the argument, so we cannot use
    // directly those passed in the pack of arguments
    auto arg_tuple = ::std::make_shared<TupleType>(std::forward<Args>(args)...);

    // Ensure we are packing arguments of the proper types to call func
    static_assert(::std::is_invocable_v<Fun, Args...>);

    // Get the address of every tuple entry
    ::std::apply(
      [this](auto&... elems) {
        // Push back the addresses of each tuple element into the args vector
        ((args_ptr.push_back(static_cast<void*>(&elems))), ...);
      },
      *arg_tuple);

    // Save the tuple in a typed erased value
    arg_tuple_type_erased = mv(arg_tuple);
  }

  /* __global__ function */
  const void* func;
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;

  // Vector of pointers to the arg_tuple which saves arguments in a typed-erased way
  ::std::vector<void*> args_ptr;

private:
  ::std::shared_ptr<void> arg_tuple_type_erased;
};

namespace reserved
{

/**
 * @brief Implementation of the CUDA kernel construct
 *
 * If the chained flag is set, we expect to have a chain of kernels, otherwise a single kernel
 */
template <typename Ctx, bool chained, typename... Deps>
class cuda_kernel_scope
{
public:
  cuda_kernel_scope(Ctx& ctx, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
  {}

  // Provide an explicit execution place
  cuda_kernel_scope(Ctx& ctx, exec_place e_place, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
      , e_place(mv(e_place))
  {}

  cuda_kernel_scope(const cuda_kernel_scope&)            = delete;
  cuda_kernel_scope& operator=(const cuda_kernel_scope&) = delete;
  // move-constructible
  cuda_kernel_scope(cuda_kernel_scope&&) = default;

  /// Add a set of dependencies
  template <typename... Pack>
  void add_deps(task_dep_untyped first, Pack&&... pack)
  {
    dynamic_deps.push_back(mv(first));
    if constexpr (sizeof...(Pack) > 0)
    {
      add_deps(::std::forward<Pack>(pack)...);
    }
  }

  template <typename T>
  decltype(auto) get(size_t submitted_index) const
  {
    _CCCL_ASSERT(untyped_t.has_value(), "uninitialized task");
    return untyped_t->template get<T>(submitted_index);
  }

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
    // If a place is specified, use it
    auto t = e_place ? ctx.task(e_place.value()) : ctx.task();

    // So that we can use get to retrieve dynamic dependencies
    untyped_t = t;

    t.add_deps(deps);

    // Append all dynamic deps
    for (auto& d : dynamic_deps)
    {
      t.add_deps(mv(d));
    }
    dynamic_deps.clear();

    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    cudaEvent_t start_event, end_event;
    const bool record_time = t.schedule_task() || statistics.is_calibrating_to_file();

    t.start();

    int device = -1;

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
            dot.template add_vertex_timing<typename Ctx::task_type>(t, milliseconds, device);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }

      t.clear();

      // Now that we have executed 'f', we do not need to access it anymore
      untyped_t.reset();
    };

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaGetDevice(&device)); // We will use this to force it during the next run
        // Events must be created here to avoid issues with multi-gpu
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    // When chained is enable, we expect a vector of kernel description which should be executed one after the other
    if constexpr (chained)
    {
      ::std::vector<cuda_kernel_desc> res = ::std::apply(f, deps.instance(t));
      assert(!res.empty());

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        auto lock = t.lock_ctx_graph();
        auto& g   = t.get_ctx_graph();

        // We have two situations : either there is a single kernel and we put the kernel in the context's
        // graph, or we rely on a child graph
        if (res.size() == 1)
        {
          insert_one_kernel(res[0], t.get_node(), g);
        }
        else
        {
          ::std::vector<cudaGraphNode_t>& chain = t.get_node_chain();
          chain.resize(res.size());

          // Create a chain of kernels
          for (size_t i = 0; i < res.size(); i++)
          {
            insert_one_kernel(res[i], chain[i], g);
            if (i > 0)
            {
              cuda_safe_call(cudaGraphAddDependencies(g, &chain[i - 1], &chain[i], 1));
            }
          }
        }
      }
      else
      {
        // Rely on stream semantic to have a dependency between the kernels
        for (auto& k : res)
        {
          cuda_safe_call(
            cudaLaunchKernel(k.func, k.gridDim, k.blockDim, k.args_ptr.data(), k.sharedMem, t.get_stream()));
        }
      }
    }
    else
    {
      // We have an unchained cuda_kernel, which means there is a single
      // CUDA kernel described, and the function should return a single
      // descriptor, not a vector
      static_assert(!chained);

      cuda_kernel_desc res = ::std::apply(f, deps.instance(t));

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        auto lock = t.lock_ctx_graph();
        insert_one_kernel(res, t.get_node(), t.get_ctx_graph());
      }
      else
      {
        cuda_safe_call(
          cudaLaunchKernel(res.func, res.gridDim, res.blockDim, res.args_ptr.data(), res.sharedMem, t.get_stream()));
      }
    }
  }

private:
  /* Add a kernel to a CUDA graph given its description */
  auto insert_one_kernel(cuda_kernel_desc& k, cudaGraphNode_t& n, cudaGraph_t& g) const
  {
    cudaKernelNodeParams kconfig;
    kconfig.blockDim       = k.blockDim;
    kconfig.extra          = nullptr;
    kconfig.func           = const_cast<void*>(k.func);
    kconfig.gridDim        = k.gridDim;
    kconfig.kernelParams   = k.args_ptr.data();
    kconfig.sharedMemBytes = k.sharedMem;
    cuda_safe_call(cudaGraphAddKernelNode(&n, g, nullptr, 0, &kconfig));
  }

  ::std::string symbol;
  Ctx& ctx;
  // Statically defined deps
  task_dep_vector<Deps...> deps;

  // Dependencies added with add_deps
  ::std::vector<task_dep_untyped> dynamic_deps;
  // Used to retrieve deps with t.get<>(...)
  ::std::optional<task> untyped_t;

  ::std::optional<exec_place> e_place;
};

} // end namespace reserved
} // end namespace cuda::experimental::stf
