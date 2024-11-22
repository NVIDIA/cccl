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
 * @brief Implement tasks in the CUDA graph backend (graph_ctx)
 *
 * @see graph_ctx
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

#include <cuda/experimental/__stf/graph/internal/event_types.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh> // graph_task<> has-a backend_ctx_untyped
#include <cuda/experimental/__stf/internal/frozen_logical_data.cuh>
#include <cuda/experimental/__stf/internal/logical_data.cuh>

namespace cuda::experimental::stf
{

template <typename... Deps>
class graph_task;

/** @brief This describes an untyped task within a CUDA graph.
 *
 * A graph task is implemented as a child graph in the graph associated to the
 * context. The body of the task is in the child graph, and CUDASTF introduces
 * the dependencies from and to that child graph, in addition to extra nodes
 * intended to implement data transfers, or allocations for example.
 *
 * For graph tasks generated automatically by CUDASTF which are only made of a
 * single CUDA graph node, we may use the graph node directly rather than
 * embedding it in a child graph.
 */
template <>
class graph_task<> : public task
{
public:
  // A cudaGraph_t is needed
  graph_task() = delete;

  graph_task(backend_ctx_untyped ctx, cudaGraph_t g, size_t epoch, exec_place e_place = exec_place::current_device())
      : task(mv(e_place))
      , ctx_graph(EXPECT(g))
      , epoch(epoch)
      , ctx(mv(ctx))
  {
    this->ctx.increment_task_count();
  }

  graph_task(graph_task&&)            = default;
  graph_task& operator=(graph_task&&) = default;

  graph_task(graph_task&)                  = default;
  graph_task(const graph_task&)            = default;
  graph_task& operator=(const graph_task&) = default;

  graph_task& start()
  {
    event_list prereqs = acquire(ctx);

    // The CUDA graph API does not like duplicate dependencies
    prereqs.optimize();

    // Reserve for better performance
    ready_dependencies.reserve(prereqs.size());

    for (auto& e : prereqs)
    {
      auto ge = reserved::graph_event(e, reserved::use_dynamic_cast);
      if (ge->epoch == epoch)
      {
        ready_dependencies.push_back(ge->node);
      }
    }

    return *this;
  }

  /* End the task, but do not clear its data structures yet */
  graph_task<>& end_uncleared()
  {
    cudaGraphNode_t n;

    auto done_prereqs = event_list();

    // We either created independant task nodes, or a child graph. We need
    // to inject input dependencies, and make the task completion depend on
    // task nodes or the child graph.
    if (task_nodes.size() > 0)
    {
      for (auto& node : task_nodes)
      {
#ifndef NDEBUG
        // Ensure the node does not have dependencies yet
        size_t num_deps;
        cuda_safe_call(cudaGraphNodeGetDependencies(node, nullptr, &num_deps));
        assert(num_deps == 0);

        // Ensure there are no output dependencies either (or we could not
        // add input dependencies later)
        size_t num_deps_out;
        cuda_safe_call(cudaGraphNodeGetDependentNodes(node, nullptr, &num_deps_out));
        assert(num_deps_out == 0);
#endif

        // Repeat node as many times as there are input dependencies
        ::std::vector<cudaGraphNode_t> out_array(ready_dependencies.size(), node);
        cuda_safe_call(
          cudaGraphAddDependencies(ctx_graph, ready_dependencies.data(), out_array.data(), ready_dependencies.size()));
        auto gnp = reserved::graph_event(node, epoch);
        gnp->set_symbol(ctx, "done " + get_symbol());
        /* This node is now the output dependency of the task */
        done_prereqs.add(mv(gnp));
      }
    }
    else
    {
      // Note that if nothing was done in the task, this will create a child
      // graph too, which will be useful as a node to synchronize with anyway.
      const cudaGraph_t childGraph = get_graph();

      const cudaGraphNode_t* deps = ready_dependencies.data();

      assert(ctx_graph);
      /* This will duplicate the childGraph so we can destroy it after */
      cuda_safe_call(cudaGraphAddChildGraphNode(&n, ctx_graph, deps, ready_dependencies.size(), childGraph));

      // Destroy the child graph unless we should not
      if (must_destroy_child_graph)
      {
        cuda_safe_call(cudaGraphDestroy(childGraph));
      }

      auto gnp = reserved::graph_event(n, epoch);
      gnp->set_symbol(ctx, "done " + get_symbol());
      /* This node is now the output dependency of the task */
      done_prereqs.add(mv(gnp));
    }

    release(ctx, done_prereqs);

    return *this;
  }

  graph_task<>& end()
  {
    end_uncleared();
    clear();
    return *this;
  }

  void populate_deps_scheduling_info() const
  {
    // Error checking copied from acquire() in acquire_release()

    int index        = 0;
    const auto& deps = get_task_deps();
    for (const auto& dep : deps)
    {
      if (!dep.get_data().is_initialized())
      {
        fprintf(stderr, "Error: dependency number %d is an uninitialized logical data.\n", index);
        abort();
      }
      dep.set_symbol(dep.get_data().get_symbol());
      dep.set_data_footprint(dep.get_data().get_data_interface().data_footprint());
      index++;
    }
  }

  /**
   * @brief Use the scheduler to assign a device to this task
   *
   * @return returns true if the task's time needs to be recorded
   */
  bool schedule_task()
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    const bool is_auto = get_exec_place().affine_data_place() == data_place::device_auto;
    bool calibrate     = false;

    // We need to know the data footprint if scheduling or calibrating tasks
    if (is_auto || statistics.is_calibrating())
    {
      populate_deps_scheduling_info();
    }

    if (is_auto)
    {
      auto [place, needs_calibration] = ctx.schedule_task(*this);
      set_exec_place(place);
      calibrate = needs_calibration;
    }

    return dot.is_timing() || (calibrate && statistics.is_calibrating());
  }

  /**
   * @brief Invokes a lambda that takes either a `cudaStream_t` or a `cudaGraph_t`. Dependencies must be
   * set with `add_deps` manually before this call.
   *
   * @tparam Fun Type of lambda to call, must accept either a `cudaStream_t` or a `cudaGraph_t` as sole argument
   * @param f lambda function to call
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    // cudaEvent_t start_event, end_event;

    bool record_time = schedule_task();

    if (statistics.is_calibrating_to_file())
    {
      record_time = true;
    }

    start();

    if (record_time)
    {
      // Events must be created here to avoid issues with multi-gpu
      // cuda_safe_call(cudaEventCreate(&start_event));
      // cuda_safe_call(cudaEventCreate(&end_event));
      // cuda_safe_call(cudaEventRecord(start_event));
    }

    SCOPE(exit)
    {
      end_uncleared();
      if (record_time)
      {
        // cuda_safe_call(cudaEventRecord(end_event));
        // cuda_safe_call(cudaEventSynchronize(end_event));

        float milliseconds = 0;
        // cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

        if (dot.is_tracing())
        {
          dot.template add_vertex_timing<task>(*this, milliseconds);
        }

        if (statistics.is_calibrating())
        {
          statistics.log_task_time(*this, milliseconds);
        }
      }
      clear();
    };

    // Default for the first argument is a `cudaStream_t`.
    if constexpr (::std::is_invocable_v<Fun, cudaStream_t>)
    {
      //
      // CAPTURE the lambda
      //

      // Get a stream from the pool associated to the execution place
      cudaStream_t capture_stream = get_exec_place().getStream(ctx.async_resources(), true).stream;

      cudaGraph_t childGraph = nullptr;
      cuda_safe_call(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));

      // Launch the user provided function
      f(capture_stream);

      cuda_safe_call(cudaStreamEndCapture(capture_stream, &childGraph));

      // This implements the child graph of the `graph_task<>`, we will later
      // insert the proper dependencies around it
      set_child_graph(childGraph);
    }
    else
    {
      //
      // Give the lambda a child graph
      //

      // Create a child graph in the `graph_task<>`
      cudaGraph_t childGraph = get_graph();

      // Launch the user provided function
      f(childGraph);
    }
  }

  // Return the child graph, and create it if it does not exist yet
  cudaGraph_t& get_graph()
  {
    // We either use a child graph or task nodes, not both
    assert(task_nodes.empty());

    // Lazy creation
    if (child_graph == nullptr)
    {
      cuda_safe_call(cudaGraphCreate(&child_graph, 0));
      must_destroy_child_graph = true;
    }

    return child_graph;
  }

  // Create a node in the graph
  cudaGraphNode_t& get_node()
  {
    // We either use a child graph or task nodes, not both
    assert(!child_graph);

    // Create a new entry and return it
    task_nodes.emplace_back();
    return task_nodes.back();
  }

  // Get the graph associated to the whole context (not the task)
  cudaGraph_t& get_ctx_graph()
  {
    return ctx_graph;
  }

  void set_current_place(pos4 p)
  {
    get_exec_place().as_grid().set_current_place(ctx, p);
  }

  void unset_current_place()
  {
    get_exec_place().as_grid().unset_current_place(ctx);
  }

  const exec_place& get_current_place() const
  {
    return get_exec_place().as_grid().get_current_place();
  }

private:
  // So that graph_ctx can access set_child_graph
  template <typename... Deps>
  friend class graph_task;

  // If the child graph was created using a capture mechanism, for example
  void set_child_graph(cudaGraph_t explicit_g)
  {
    child_graph              = explicit_g;
    must_destroy_child_graph = false;
  }

  /* The child graph associated to that `graph_task<>`, this was either created
   * explicitly, or by the means of a capture mechanism. */
  cudaGraph_t child_graph       = nullptr;
  bool must_destroy_child_graph = false;

  /* If the task corresponds to independant graph nodes, we do not use a
   * child graph, but add nodes directly */
  ::std::vector<cudaGraphNode_t> task_nodes;

  /* This is the support graph associated to the entire context */
  cudaGraph_t ctx_graph = nullptr;
  size_t epoch          = 0;

  ::std::vector<cudaGraphNode_t> ready_dependencies;

  backend_ctx_untyped ctx;
};

/**
 * @brief Provides a graph scope object within which functions invoked are
 * either captured in a graph or passed to a subgraph, depending on the
 * function type.  Invocation is carried by means of operator->*. Dependencies
 * are typed appropriately.
 */
template <typename... Deps>
class graph_task : public graph_task<>
{
public:
  graph_task(backend_ctx_untyped ctx, cudaGraph_t g, size_t epoch, exec_place e_place, task_dep<Deps>... deps)
      : graph_task<>(mv(ctx), g, epoch, mv(e_place))
  {
    static_assert(sizeof(*this) == sizeof(graph_task<>), "Cannot add state - it would be lost by slicing.");
    add_deps(mv(deps)...);
  }

  /**
   * @brief Set the symbol object
   *
   * @param s
   * @return graph_task&
   */
  graph_task& set_symbol(::std::string s) &
  {
    graph_task<>::set_symbol(mv(s));
    return *this;
  }

  graph_task&& set_symbol(::std::string s) &&
  {
    graph_task<>::set_symbol(mv(s));
    return mv(*this);
  }

#if _CCCL_COMPILER(MSVC)
  // TODO (miscco): figure out why MSVC is complaining about unreachable code here
  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code
#endif // _CCCL_COMPILER(MSVC)

  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    // cudaEvent_t start_event, end_event;

    bool record_time = schedule_task();

    if (statistics.is_calibrating_to_file())
    {
      record_time = true;
    }

    start();

    if (record_time)
    {
      // Events must be created here to avoid issues with multi-gpu
      // cuda_safe_call(cudaEventCreate(&start_event));
      // cuda_safe_call(cudaEventCreate(&end_event));
      // cuda_safe_call(cudaEventRecord(start_event));
    }

    SCOPE(exit)
    {
      end_uncleared();
      if (record_time)
      {
        // cuda_safe_call(cudaEventRecord(end_event));
        // cuda_safe_call(cudaEventSynchronize(end_event));

        float milliseconds = 0;
        // cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

        if (dot.is_tracing())
        {
          dot.template add_vertex_timing<task>(*this, milliseconds);
        }

        if (statistics.is_calibrating())
        {
          statistics.log_task_time(*this, milliseconds);
        }
      }
      clear();
    };

    if (dot.is_tracing())
    {
      dot.template add_vertex<task, logical_data_untyped>(*this);
    }

    // Default for the first argument is a `cudaStream_t`.
    if constexpr (::std::is_invocable_v<Fun, cudaStream_t, Deps...>)
    {
      //
      // CAPTURE the lambda
      //

      // Get a stream from the pool associated to the execution place
      cudaStream_t capture_stream = get_exec_place().getStream(ctx.async_resources(), true).stream;

      cudaGraph_t childGraph = nullptr;
      cuda_safe_call(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));

      // Launch the user provided function
      ::std::apply(f, tuple_prepend(mv(capture_stream), typed_deps()));

      cuda_safe_call(cudaStreamEndCapture(capture_stream, &childGraph));

      // Save this child graph as the implementation of the
      // graph_task<>. CUDASTF will then add all necessary
      // dependencies, or data transfers, allocations etc.
      // Since this was captured, we will not destroy that graph (should we ?)
      set_child_graph(childGraph);
    }
    else
    {
      static_assert(::std::is_invocable_v<Fun, cudaGraph_t, Deps...>, "Incorrect lambda function signature.");
      //
      // Give the lambda a child graph
      //

      // This lazily creates a childGraph which will be destroyed when the task ends
      cudaGraph_t childGraph = get_graph();

      // Launch the user provided function
      ::std::apply(f, tuple_prepend(mv(childGraph), typed_deps()));
    }
  }
#if _CCCL_COMPILER(MSVC)
  _CCCL_DIAG_POP
#endif // _CCCL_COMPILER(MSVC)

private:
  auto typed_deps()
  {
    return make_tuple_indexwise<sizeof...(Deps)>([&](auto i) {
      return this->get<::std::tuple_element_t<i, ::std::tuple<Deps...>>>(i);
    });
  }
};

} // namespace cuda::experimental::stf
