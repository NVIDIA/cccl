//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Algorithm construct, to embed a CUDA graph within a task
 */

#pragma once

#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/internal/context.cuh>

#include <map>

namespace cuda::experimental::stf
{
/**
 * @brief Algorithms are a mechanism to implement reusable task sequences implemented by the means of CUDA graphs nested
 * within a task.
 *
 * The underlying CUDA graphs are cached so that they are temptatively reused
 * when the algorithm is run again. Nested algorithms are internally implemented as child graphs.
 */
class algorithm
{
private:
  template <typename context_t, typename... Deps>
  class runner_impl
  {
  public:
    runner_impl(context_t& _ctx, algorithm& _alg, task_dep<Deps>... _deps)
        : alg(_alg)
        , ctx(_ctx)
        , deps(mv(_deps)...)
    {}

    template <typename Fun>
    void operator->*(Fun&& fun)
    {
      std::apply(
        [&](task_dep<Deps>&... unpacked_deps) {
          if (getenv("CUDASTF_ALGORITHM_INLINE"))
          {
            alg.run_inline(std::forward<Fun>(fun), ctx, unpacked_deps...);
          }
          else
          {
            alg.run_as_task(std::forward<Fun>(fun), ctx, unpacked_deps...);
          }
        },
        deps);
    }

    algorithm& alg;
    context_t& ctx;
    ::std::tuple<task_dep<Deps>...> deps;
  };

public:
  algorithm(::std::string _symbol = "algorithm")
      : symbol(mv(_symbol))
  {}

  /* Inject the execution of the algorithm within a CUDA graph */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run_in_graph(Fun&& fun, parent_ctx_t& parent_ctx, cudaGraph_t graph, const Args&... args)
  {
    const size_t hashValue = hash_all(args...);
    ::std::shared_ptr<cudaGraph_t> inner_graph;

    if (auto search = graph_cache.find(hashValue); search != graph_cache.end())
    {
      inner_graph = search->second;
    }
    else
    {
      graph_ctx gctx(parent_ctx.async_resources());

      // Useful for tools
      gctx.set_parent_ctx(parent_ctx);
      gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

      auto current_data_place = gctx.default_exec_place().affine_data_place();

      // Call fun, transforming the tuple of instances into a tuple of logical data
      // Our infrastructure currently does not like to work with
      // constant types for the data interface so we pretend this is
      // a modifiable data if necessary
      fun(gctx, gctx.logical_data(to_rw_type_of(args), current_data_place)...);

      inner_graph = gctx.finalize_as_graph();

      // TODO validate that the graph is reusable before storing it !
      // fprintf(stderr, "CACHE graph...\n");
      graph_cache[hashValue] = inner_graph;
    }

    cudaGraphNode_t c;
    cuda_safe_call(cudaGraphAddChildGraphNode(&c, graph, nullptr, 0, *inner_graph));
  }

  /* This simply executes the algorithm within the existing context. This
   * makes it possible to observe the impact of an algorithm by disabling it in
   * practice (without bloating the code with both the algorithm and the original
   * code) */
  template <typename context_t, typename Fun, typename... Deps>
  void run_inline(Fun&& fun, context_t& ctx, const task_dep<Deps>&... deps)
  {
    fun(ctx, logical_data<Deps>(deps.get_data())...);
  }

  template <typename Fun, typename... Deps>
  void run_as_task(Fun&& fun, stream_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(mv(deps)...).set_symbol(symbol)->*[this, &fun, &ctx](cudaStream_t stream, const Deps&... args) {
      this->run(::std::forward<Fun>(fun), ctx, stream, args...);
    };
  }

  template <typename Fun, typename... Deps>
  void run_as_task(Fun&& fun, graph_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(mv(deps)...).set_symbol(symbol)->*[this, &fun, &ctx](cudaGraph_t g, const Deps&... args) {
      this->run_in_graph(::std::forward<Fun>(fun), ctx, g, args...);
    };
  }

  /**
   * @brief Executes `fun` within a task that takes a pack of dependencies
   *
   * As an alternative, the run_as_task_dynamic may take a variable number of dependencies
   */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun&& fun, context& ctx, task_dep<Deps>... deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task(::std::forward<Fun>(fun), actual_ctx, mv(deps)...);
      },
      ctx.payload);
  }

  /* Helper to run the algorithm in a stream_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun&& fun, stream_ctx& ctx, task_dep_vector_untyped deps)
  {
    auto t = ctx.task();
    t.add_deps(mv(deps));
    t.set_symbol(symbol);

    t->*[this, &fun, &ctx, &t](cudaStream_t stream) {
      this->run_dynamic(::std::forward<Fun>(fun), ctx, stream, t);
    };
  }

  /* Helper to run the algorithm in a graph_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun&& /* fun */, graph_ctx& /* ctx */, const ::std::vector<task_dep_untyped>& /* deps */)
  {
    /// TODO
    abort();
  }

  /**
   * @brief Executes `fun` within a task that takes a vector of untyped dependencies.
   *
   * This is an alternative for run_as_task which may take a variable number of dependencies
   */
  template <typename Fun>
  void run_as_task_dynamic(Fun&& fun, context& ctx, task_dep_vector_untyped deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task_dynamic(::std::forward<Fun>(fun), actual_ctx, mv(deps));
      },
      ctx.payload);
  }

  /**
   * @brief Helper to use algorithm using the ->* idiom instead of passing the implementation as an argument of
   * run_as_task
   *
   * example:
   *    algorithm alg;
   *    alg.runner(ctx, lX.read(), lY.rw())->*[](context inner_ctx, logical_data<slice<double>> X,
   * logical_data<slice<double>> Y) { inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto
   * x, auto y) { y(i) = 2.0*x(i);
   *        };
   *    };
   *
   *
   * Which is equivalent to:
   * auto fn = [](context inner_ctx, logical_data<slice<double>> X,  logical_data<slice<double>> Y) {
   *     inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto x, auto y) {
   *         y(i) = 2.0*x(i);
   *     }
   * };
   *
   * algorithm alg;
   * alg.run_as_task(fn, ctx, lX.read(), lY.rw());
   */
  template <typename context_t, typename... Deps>
  runner_impl<context_t, Deps...> runner(context_t& ctx, task_dep<Deps>... deps)
  {
    return runner_impl(ctx, *this, mv(deps)...);
  }

  auto setup_allocator(graph_ctx& gctx, cudaStream_t stream)
  {
    // Use a pooled allocator: this avoids calling the underlying "uncached"
    // allocator too often by making larger allocations which can be used for
    // multiple small allocations
    gctx.set_allocator(block_allocator<pooled_allocator>(gctx));

    // The uncached allocator allocates the (large) blocks of memory required
    // by the allocator. Within CUDA graphs, using memory nodes is expensive,
    // and caching a graph with memory nodes may appear as "leaking" memory.
    // We thus use the stream_adapter allocator which relies stream-based
    // asynchronous allocator API (cudaMallocAsync, cudaFreeAsync)
    // The resources reserved by this allocator can be released asynchronously
    // after the submission of the CUDA graph.
    auto wrapper = stream_adapter(gctx, stream);

    gctx.update_uncached_allocator(wrapper.allocator());

    return wrapper;
  }

  /* Execute the algorithm as a CUDA graph and launch this graph in a CUDA
   * stream */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run(Fun&& fun, parent_ctx_t& parent_ctx, cudaStream_t stream, const Args&... args)
  {
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    // This will setup allocators to avoid created CUDA graph memory nodes, and
    // defer the allocations and deallocations to the cudaMallocAsync API
    // instead. These resources need to be released later with .clear()
    auto adapter = setup_allocator(gctx, stream);

    auto current_data_place = gctx.default_exec_place().affine_data_place();

    // Call fun with all arguments transformed to logical data
    // Our infrastructure currently does not like to work with constant
    // types for the data interface so we pretend this is a modifiable
    // data if necessary
    fun(gctx, gctx.logical_data(to_rw_type_of(args), current_data_place)...);

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg;

    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      if (reserved::try_updating_executable_graph(*pe, *gctx_graph))
      {
        eg = pe;
        break;
      }
    }

    if (!eg)
    {
      eg = ::std::shared_ptr<cudaGraphExec_t>(new cudaGraphExec_t, [](cudaGraphExec_t* p) {
        cudaGraphExecDestroy(*p);
      });

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(eg.get(), *gctx_graph, 0));

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));

    // Free resources allocated through the adapter
    adapter.clear();
  }

  /* Contrary to `run`, we here have a dynamic set of dependencies for the
   * task, so fun does not take a pack of data instances as a parameter */
  template <typename Fun, typename parent_ctx_t, typename task_t>
  void run_dynamic(Fun&& fun, parent_ctx_t& parent_ctx, cudaStream_t stream, task_t& t)
  {
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    // This will setup allocators to avoid created CUDA graph memory nodes, and
    // defer the allocations and deallocations to the cudaMallocAsync API
    // instead. These resources need to be released later with .clear()
    auto adapter = setup_allocator(gctx, stream);

    auto current_place = gctx.default_exec_place();

    ::std::forward<Fun>(fun)(gctx, t);

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg;

    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      if (reserved::try_updating_executable_graph(*pe, *gctx_graph))
      {
        eg = pe;
        break;
      }
    }

    if (!eg)
    {
      auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
        cudaGraphExecDestroy(*pGraphExec);
      };
      eg = ::std::shared_ptr<cudaGraphExec_t>(new cudaGraphExec_t, cudaGraphExecDeleter);

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(eg.get(), *gctx_graph, 0));

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));

    // Free resources allocated through the adapter
    adapter.clear();
  }

private:
  // Generate a DOT output of a CUDA graph using CUDA
  void dump_algorithm(const ::std::shared_ptr<cudaGraph_t>& gctx_graph)
  {
    if (getenv("CUDASTF_DUMP_ALGORITHMS"))
    {
      static int print_to_dot_cnt = 0; // Warning: not thread-safe
      ::std::string filename      = "algo_" + symbol + "_" + ::std::to_string(print_to_dot_cnt++) + ".dot";
      cudaGraphDebugDotPrint(*gctx_graph, filename.c_str(), cudaGraphDebugDotFlags(0));
    }
  }

  ::std::map<cudaStream_t, ::std::vector<::std::shared_ptr<cudaGraphExec_t>>> cached_exec_graphs;

  // Cache executable graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraphExec_t>> exec_graph_cache;

  // Cache CUDA graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraph_t>> graph_cache;

  ::std::string symbol;
};
} // end namespace cuda::experimental::stf
