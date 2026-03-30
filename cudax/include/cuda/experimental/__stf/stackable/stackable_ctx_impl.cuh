//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Implementation of the stackable_ctx class

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <algorithm>
#include <atomic>
#include <iostream>
#include <shared_mutex>
#include <stack>
#include <thread>

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/__stf/internal/context.cuh"
#include "cuda/experimental/__stf/internal/task.cuh"
#include "cuda/experimental/__stf/stackable/conditional_nodes.cuh"
#include "cuda/experimental/__stf/stackable/stackable_node_hierarchy.cuh"
#include "cuda/experimental/__stf/stackable/stackable_task_dep.cuh"
#include "cuda/experimental/__stf/utility/hash.cuh"
#include "cuda/experimental/__stf/utility/source_location.cuh"

namespace cuda::experimental::stf
{
template <typename T>
class stackable_logical_data;

//! \brief Base class with a virtual pop method to enable type erasure
//!
//! This is used to implement the automatic call to pop() on logical data when
//! a context node is popped, as we need to keep a vector of logical data that
//! were imported without knowing their type.
//!
class stackable_logical_data_impl_state_base
{
public:
  virtual ~stackable_logical_data_impl_state_base()                                      = default;
  virtual void pop_before_finalize(int ctx_offset)                                       = 0;
  virtual void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) = 0;
};

// Combine access mode for a logical data id into a small vector.
// Used to merge duplicate data accesses within a single task's deps
// (e.g. read + write on the same data becomes rw).
inline void combine_access_mode(::std::vector<::std::pair<int, access_mode>>& combined, int id, access_mode m)
{
  auto it = ::std::find_if(combined.begin(), combined.end(), [id](const ::std::pair<int, access_mode>& entry) {
    return entry.first == id;
  });
  if (it != combined.end())
  {
    it->second = it->second | m;
  }
  else
  {
    combined.emplace_back(id, m);
  }
}

inline access_mode lookup_combined_mode(const ::std::vector<::std::pair<int, access_mode>>& combined, int id)
{
  auto it = ::std::find_if(combined.begin(), combined.end(), [id](const ::std::pair<int, access_mode>& entry) {
    return entry.first == id;
  });
  _CCCL_ASSERT(it != combined.end(), "internal error: logical data id not found in combined modes");
  return it->second;
}

//! \brief This class defines a context that behaves as a context which can have nested subcontexts (implemented as
//! local CUDA graphs)
class stackable_ctx
{
public:
  template <typename T>
  using logical_data_t = ::cuda::experimental::stf::stackable_logical_data<T>;

  //! @brief Defers task creation until all dependencies are known.
  //!
  //! In a regular context, task() immediately creates the underlying task and
  //! add_deps() appends to it -- access modes are already resolved.
  //!
  //! In a stackable context, we cannot create the task immediately because
  //! add_deps() may later introduce additional dependencies on the same logical
  //! data with a different access mode (e.g. read initially, then write via
  //! add_deps). The data must be imported (pushed/frozen) with the *combined*
  //! mode (rw in that example), but once data is frozen with a given mode it
  //! cannot be upgraded.
  //!
  //! deferred_task_builder therefore collects all dependencies first, combines
  //! access modes per logical data, validates and auto-pushes with the correct
  //! combined mode, and only then creates the real task via the underlying
  //! context. The task is concretized lazily on the first call to operator->*
  //! (when the task body is provided) or set_symbol.
  template <typename... Deps>
  class deferred_task_builder
  {
  public:
    // Safe to store a reference: the builder is a short-lived temporary used
    // in a single expression (e.g. ctx.task(deps)->*lambda) and is never
    // persisted beyond the statement that created it.
    stackable_ctx& sctx_;
    int offset_;

    exec_place exec_place_;
    ::std::tuple<Deps...> task_deps_tuple_;

    // Type-erased metadata for additional dependencies added via add_deps()
    struct additional_dep_info
    {
      int logical_data_id;
      access_mode mode;
      ::std::function<void(stackable_ctx&, int, access_mode)> validate_access_op;
      ::std::function<task_dep_untyped(int, access_mode)> resolve_op;
    };
    ::std::vector<additional_dep_info> additional_deps_;

    // Store the concrete task (base class), set by concretize_deferred_task
    ::std::optional<::cuda::experimental::stf::task> concrete_task_;

    // Optional symbol to be applied to the underlying task when concretized
    ::std::optional<::std::string> symbol_;

    template <typename ExecPlace>
    deferred_task_builder(stackable_ctx& sctx, int offset, ExecPlace&& exec_place, Deps&&... deps)
        : sctx_(sctx)
        , offset_(offset)
        , exec_place_(::std::move(exec_place))
        , task_deps_tuple_(::std::forward<Deps>(deps)...)
    {
      static_assert((reserved::is_stackable_task_dep_v<::std::decay_t<Deps>> && ...),
                    "All dependency arguments must be stackable task dependencies");
    }

    // Add more dependencies via add_deps
    template <typename... MoreDeps>
    auto& add_deps(MoreDeps&&... deps)
    {
      auto store_dep = [this](const auto& dep) {
        static_assert(reserved::is_stackable_task_dep_v<::std::decay_t<decltype(dep)>>,
                      "add_deps in stackable context only accepts stackable task dependencies");

        additional_dep_info info;
        info.logical_data_id = dep.get_d().get_unique_id();
        info.mode            = dep.get_access_mode();

        auto logical_data       = dep.get_d();
        info.validate_access_op = [logical_data](stackable_ctx& sctx, int offset, access_mode mode) mutable {
          logical_data.validate_access(offset, sctx, mode);
        };

        info.resolve_op = [logical_data](int offset, access_mode mode) mutable -> task_dep_untyped {
          auto& ld = logical_data.get_ld(offset);
          return task_dep_untyped(ld, mode);
        };

        additional_deps_.push_back(::std::move(info));
      };
      (store_dep(deps), ...);
      return *this;
    }

  private:
    // Concretize the deferred task: combine access modes, validate/push, resolve deps, create the task.
    template <typename TaskAction>
    auto concretize_deferred_task(TaskAction&& action)
    {
      return ::std::apply(
        [this, &action](auto&&... initial_args) {
          // 1. Combine access modes across all dependencies (initial + additional)
          ::std::vector<::std::pair<int, access_mode>> combined_modes;

          if constexpr (sizeof...(initial_args) > 0)
          {
            auto combine = [&](const auto& arg) {
              combine_access_mode(combined_modes, arg.get_d().get_unique_id(), arg.get_access_mode());
            };
            (combine(initial_args), ...);
          }

          for (const auto& info : additional_deps_)
          {
            combine_access_mode(combined_modes, info.logical_data_id, info.mode);
          }

          // 2. Validate and auto-push with combined modes
          if constexpr (sizeof...(initial_args) > 0)
          {
            auto validate = [&](const auto& arg) {
              arg.get_d().validate_access(
                offset_, sctx_, lookup_combined_mode(combined_modes, arg.get_d().get_unique_id()));
            };
            (validate(initial_args), ...);
          }

          for (const auto& info : additional_deps_)
          {
            info.validate_access_op(sctx_, offset_, lookup_combined_mode(combined_modes, info.logical_data_id));
          }

          // 3. Resolve initial deps and create the task
          auto task = [&]() {
            auto& ctx = sctx_.get_ctx(offset_);

            auto task_deps = ::std::apply(
              [this](auto&&... deps) {
                return ::std::make_tuple(deps.resolve(offset_)...);
              },
              task_deps_tuple_);

            return ::std::apply(
              [&ctx, exec_place = exec_place_](auto&&... deps) {
                return ctx.task(exec_place, deps...);
              },
              task_deps);
          }();

          // 4. Resolve and add additional deps from add_deps calls
          for (const auto& info : additional_deps_)
          {
            task.add_deps(info.resolve_op(offset_, lookup_combined_mode(combined_modes, info.logical_data_id)));
          }

          if (symbol_.has_value())
          {
            task.set_symbol(*symbol_);
          }

          concrete_task_ = task.get_base_task();
          return action(task);
        },
        task_deps_tuple_);
    }

  public:
    // Execute with lambda - concretize the deferred task and execute
    template <typename F>
    auto operator->*(F&& f)
    {
      return concretize_deferred_task([&f](auto& task) {
        return task->*::std::forward<F>(f);
      });
    }

    // Start the task - concretize the deferred task and start
    auto& start()
    {
      concretize_deferred_task([](auto& task) {
        task.start();
        return 0; // dummy return for consistency
      });
      return *this;
    }

    // Set symbol for the task - store for later application when concretized
    auto& set_symbol(::std::string s) &
    {
      symbol_ = ::std::move(s);
      return *this;
    }

    auto&& set_symbol(::std::string s) &&
    {
      symbol_ = ::std::move(s);
      return ::std::move(*this);
    }

    // Set exec_place for the task
    template <typename ExecPlace>
    auto& set_exec_place(ExecPlace&& ep) &
    {
      exec_place_ = ::std::forward<ExecPlace>(ep);
      return *this;
    }

    template <typename ExecPlace>
    auto&& set_exec_place(ExecPlace&& ep) &&
    {
      exec_place_ = ::std::forward<ExecPlace>(ep);
      return ::std::move(*this);
    }

    // Add get method for compatibility with test code
    template <typename T>
    auto get(size_t index) const
    {
      _CCCL_ASSERT(concrete_task_.has_value(), "get() called before task was concretized via ->* or start()");
      return concrete_task_->template get<T>(index);
    }
  };

  class impl
  {
    friend class stackable_ctx;

  private:
    /*
     * Base class for all nested context types
     */
    class ctx_node_base
    {
    public:
      ctx_node_base(cudaStream_t support_stream = nullptr, ::std::shared_ptr<stream_adapter> alloc_adapters = nullptr)
          : support_stream(mv(support_stream))
          , alloc_adapters(mv(alloc_adapters))
      {}

      virtual ~ctx_node_base() = default;

      // Virtual methods for context-specific behavior
      virtual event_list finalize() = 0;

      virtual cudaGraph_t get_graph() const
      {
        return nullptr;
      }

      ctx_node_base(ctx_node_base&&) noexcept            = default;
      ctx_node_base& operator=(ctx_node_base&&) noexcept = default;

      // To avoid prematurely destroying data created in a nested context, we
      // need to hold a reference to them. Since we do not store types the
      // reference are kept in a type-erased manner.
      //
      // This happens for example in this case where we want to defer the release
      // of the resources of "a" until we call pop() because this is when we would
      // have submitted the CUDA graph where "a" is used. Destroying it earlier
      // would mean we destroy that memory before the graph is even launched.
      //
      // ctx.push()
      // {
      //    auto a = ctx.logical_data(...);
      //    ... use a ...
      // }
      // ctx.pop()
      void retain_data(::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
      {
        // This keeps a reference to the shared_ptr until retained_data is destroyed
        retained_data.push_back(mv(data_impl));
      }

      // Ensure we know this data has been imported in the local context
      void track_pushed_data(::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
      {
        _CCCL_ASSERT(data_impl, "invalid value");
        pushed_data.push_back(mv(data_impl));
      }

      ctx_node_base* parent_ctx_node = nullptr;
      context ctx;
      cudaStream_t support_stream;
      // A wrapper to forward allocations from a node to its parent node (none is used at the root level)
      ::std::shared_ptr<stream_adapter> alloc_adapters;

      // This keeps track of the logical data that were pushed in this ctx node
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> pushed_data;

      // Where was the push() called ?
      ::cuda::std::source_location callsite;

      // The async resource handle used in this context
      ::std::optional<async_resources_handle> async_handle;

      // Collection of events to start the context (based on the freeze
      // operations to get data imported into the context)
      event_list ctx_prereqs;

    protected:
      // If we want to keep the state of some logical data implementations until this node is popped
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> retained_data;

    public:
      // Nested contexts do not clear their allocator because the memory they
      // use needs to be valid until we have launched the graph.
      ::std::vector<::std::shared_ptr<stream_adapter>> retained_adapters;

      // Indicate if we need to clear the adapters or pass them to the parent
      bool clear_adapters = true;
    };

    /*
     * Stream context node - uses regular stream context finalization
     */
    class stream_ctx_node : public ctx_node_base
    {
    public:
      // Default constructor that creates a stream context
      stream_ctx_node()
      {
        ctx = stream_ctx();
      }

      event_list finalize() override
      {
        // This is blocking so there is no dependency in event_list
        ctx.finalize();
        return event_list{};
      }
    };

    /*
     * Graph context node - uses explicit graph management
     */
    class graph_ctx_node : public ctx_node_base
    {
    public:
      // Constructor for graph contexts with optional conditional support via config
      graph_ctx_node(ctx_node_base* parent_node,
                     async_resources_handle handle,
                     const ::cuda::std::source_location& loc,
                     [[maybe_unused]] const push_while_config& config = push_while_config{})
      {
        // Graph context nodes create and set up the internal CUDA graph
        _CCCL_ASSERT(parent_node != nullptr, "Graph context must have a valid parent");

        cudaGraph_t parent_graph = parent_node->ctx.graph();

        /* The parent is either a stream_ctx, or a graph itself. If this is a graph, we either add a new child graph, or
         * a conditional node. */

        if (parent_graph)
        {
          nested_graph   = true;
          clear_adapters = false;
#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
          if (config.conditional_handle != nullptr)
          {
            // We will add a conditional node to an existing graph, we do not create a new graph.
            graph = parent_graph;
          }
          else
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
          {
            cudaGraph_t dummy_graph;
            cuda_safe_call(cudaGraphCreate(&dummy_graph, 0));

            // The dependencies to this child graph will be added later
            cudaGraphNode_t n;
            cuda_safe_call(cudaGraphAddChildGraphNode(&n, parent_graph, nullptr, 0, dummy_graph));

            // cudaGraphAddChildGraphNode clones dummy_graph into the parent
            cuda_safe_call(cudaGraphDestroy(dummy_graph));

            // Get the graph described by the child, not the graph that was
            // cloned into the child graph node so that changes are reflected
            // in it.
            cuda_safe_call(cudaGraphChildGraphNodeGetGraph(n, &graph));
            input_node  = n;
            output_node = n;
          }
        }
        else
        {
          cuda_safe_call(cudaGraphCreate(&graph, 0));
        }

        // This is the graph which will be used by our STF context. If we need
        // to use a conditional node later, will will change that value.
        cudaGraph_t sub_graph = graph;

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
        if (config.conditional_handle != nullptr)
        {
          // Create the conditional handle and store it in the provided pointer
          cuda_safe_call(cudaGraphConditionalHandleCreate(
            config.conditional_handle, graph, config.default_launch_value, config.flags));

          // Create conditional node parameters
          cudaGraphNodeParams cParams = {};
          cParams.type                = cudaGraphNodeTypeConditional;
          cParams.conditional.handle  = *config.conditional_handle;
          cParams.conditional.type    = config.conditional_type;
          cParams.conditional.size    = 1;

          // Add conditional node to parent graph
          cudaGraphNode_t conditionalNode;
#  if _CCCL_CTK_AT_LEAST(13, 0)
          cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, nullptr, nullptr, 0, &cParams));
#  else
          cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, nullptr, 0, &cParams));
#  endif

          // Get the body graph from the conditional node
          sub_graph = cParams.conditional.phGraph_out[0];

          // XXX if we ever reexecute that graph again, the conditional handle
          // will have been set to false when we get out of the loop, so we
          // reset it here, until CUDA provides a way to do that automatically
          // (if ever possible)
          cudaKernelNodeParams kconfig;
          kconfig.gridDim        = 1;
          kconfig.blockDim       = 1;
          kconfig.extra          = nullptr;
          kconfig.func           = (void*) reserved::condition_reset<true>;
          void* kconfig_args[1]  = {const_cast<void*>(static_cast<const void*>(config.conditional_handle))};
          kconfig.kernelParams   = kconfig_args;
          kconfig.sharedMemBytes = 0;

          cudaGraphNode_t reset_node;
          cuda_safe_call(cudaGraphAddKernelNode(&reset_node, graph, &conditionalNode, 1, &kconfig));

          input_node  = conditionalNode;
          output_node = reset_node;
        }
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

        auto& parent_ctx = parent_node->ctx;

        // Get a stream from previous context
        if (nested_graph)
        {
          // Reuse the support stream of the parent ctx node
          support_stream = parent_node->support_stream;
        }
        else
        {
          support_stream = parent_ctx.pick_stream();
        }

        auto gctx = graph_ctx(sub_graph, support_stream, handle);

        // Set up context properties
        gctx.set_parent_ctx(parent_ctx);
        gctx.get_dot()->set_ctx_symbol("graph[" + ::std::string(loc.file_name()) + ":" + ::std::to_string(loc.line())
                                       + "(" + loc.function_name() + ")]");

        // Create stream adapter and update allocator
        // Note that nested contexts use their own allocator as well, but use the same support stream...
        alloc_adapters = ::std::make_shared<stream_adapter>(gctx, support_stream);
        gctx.update_uncached_allocator(alloc_adapters->allocator());

        // Set the context and save async handle
        ctx             = gctx;
        parent_ctx_node = parent_node;
        async_handle    = mv(handle);
      }

      event_list finalize() override
      {
        // Use finalize_as_graph pattern for explicit graph management
        _CCCL_ASSERT(ctx.is_graph_ctx(), "graph_ctx_node must contain a graph context");

        ctx.to_graph_ctx().finalize_as_graph();

        auto& parent_ctx = parent_ctx_node->ctx;

        // This was either a new graph (with a stream_ctx parent) or a nested
        // graph ctx node. If this was a nested context, we do not need to
        // instantiate and launch, but we need to enforce dependencies by
        // adding input deps
        if (nested_graph)
        {
          // Record body graph complexity for stats (nested graphs are not
          // independently cached, so instantiate/update counts stay at 0).
          cudaGraph_t body = ctx.to_graph_ctx().get_graph();

          auto* cache_stat = ctx.graph_get_cache_stat();
          if (cache_stat)
          {
            cuda_safe_call(cudaGraphGetNodes(body, nullptr, &cache_stat->nnodes));
#if _CCCL_CTK_AT_LEAST(13, 0)
            cuda_safe_call(cudaGraphGetEdges(body, nullptr, nullptr, nullptr, &cache_stat->nedges));
#else
            cuda_safe_call(cudaGraphGetEdges(body, nullptr, nullptr, &cache_stat->nedges));
#endif
          }

          static const bool dump_nested =
            (getenv("CUDASTF_DUMP_GRAPHS") != nullptr) || (getenv("CUDASTF_DEBUG_STACKABLE_DOT") != nullptr);
          if (dump_nested)
          {
            static ::std::atomic<int> nested_cnt{0};
            ::std::string filename = "nested_graph" + ::std::to_string(nested_cnt++) + ".dot";
            cuda_safe_call(cudaGraphDebugDotPrint(body, filename.c_str(), cudaGraphDebugDotFlags(0)));
          }

          cudaGraph_t support_graph = parent_ctx.graph();
          size_t graph_stage        = parent_ctx.stage();

          // Transfer resources from nested context to parent context
          // This works because the completion of the parent context depends on the completion of the nested context
          parent_ctx.import_resources_from(ctx);

          // Add dependencies from the get operations to the graph node that
          // corresponds to the child graph or conditional node
          ::std::vector<cudaGraphNode_t> ctx_ready_nodes =
            reserved::join_with_graph_nodes(parent_ctx.get_backend(), ctx_prereqs, graph_stage);
          if (!ctx_ready_nodes.empty())
          {
            // Create a vector of input_node repeated for each dependency
            ::std::vector<cudaGraphNode_t> to_nodes(ctx_ready_nodes.size(), input_node);
#if _CCCL_CTK_AT_LEAST(13, 0)
            cuda_safe_call(cudaGraphAddDependencies(
              support_graph, ctx_ready_nodes.data(), to_nodes.data(), nullptr, ctx_ready_nodes.size()));
#else // _CCCL_CTK_AT_LEAST(13, 0)
            cuda_safe_call(
              cudaGraphAddDependencies(support_graph, ctx_ready_nodes.data(), to_nodes.data(), ctx_ready_nodes.size()));
#endif // _CCCL_CTK_AT_LEAST(13, 0)
          }

          auto output_node_event = reserved::graph_event(output_node, graph_stage, support_graph);

          return event_list(mv(output_node_event));
        }

        // Honour CUDASTF_DUMP_GRAPHS (same env var as graph_ctx::instantiate)
        // and the stackable-specific CUDASTF_DEBUG_STACKABLE_DOT.
        static const bool dump_graphs =
          (getenv("CUDASTF_DUMP_GRAPHS") != nullptr) || (getenv("CUDASTF_DEBUG_STACKABLE_DOT") != nullptr);
        if (dump_graphs)
        {
          static ::std::atomic<int> debug_graph_cnt{0};
          ::std::string filename = "instantiated_graph" + ::std::to_string(debug_graph_cnt++) + ".dot";
          cuda_safe_call(cudaGraphDebugDotPrint(graph, filename.c_str(), cudaGraphDebugDotFlags(0)));
        }

        size_t nnodes;
        size_t nedges;
        cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &nnodes));
#if _CCCL_CTK_AT_LEAST(13, 0)
        cuda_safe_call(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &nedges));
#else
        cuda_safe_call(cudaGraphGetEdges(graph, nullptr, nullptr, &nedges));
#endif

        auto [exec_graph, cache_hit] = ctx.async_resources().cached_graphs_query(nnodes, nedges, graph);

        auto* cache_stat = ctx.graph_get_cache_stat();
        if (cache_stat)
        {
          cache_stat->nnodes = nnodes;
          cache_stat->nedges = nedges;
          if (cache_hit)
          {
            cache_stat->update_cnt++;
          }
          else
          {
            cache_stat->instantiate_cnt++;
          }
        }

        // Make sure we launch after the "get" operations are done
        ctx_prereqs.sync_with_stream(ctx.get_backend(), support_stream);

        // Launch the graph
        cuda_safe_call(cudaGraphLaunch(*exec_graph, support_stream));

        // Release context resources after graph execution
        ctx.release_resources(support_stream);

        // Create an event that depends on the completion of previous operations in the stream
        event_list finalize_prereqs = parent_ctx.stream_to_event_list(support_stream, "finalized");
        return finalize_prereqs;
      }

      cudaGraph_t get_graph() const override
      {
        return graph;
      }

    private:
      cudaGraph_t graph = nullptr; // Graph containing conditional node (if conditional) or the entire graph otherwise
      bool nested_graph = false;
      // If we have a nested graph input and output node correspond to the nodes on which to enforce input or output
      // deps (they can be the same)
      cudaGraphNode_t input_node  = nullptr;
      cudaGraphNode_t output_node = nullptr;
    };

    // Grow the sparse context node vector to accommodate at least target_size entries.
    void grow_context_nodes(int target_size)
    {
      if (target_size < int(nodes.size()))
      {
        return;
      }

      size_t new_size = ::std::max(
        static_cast<size_t>(target_size),
        nodes.size() * node_hierarchy::default_growth_numerator / node_hierarchy::default_growth_denominator);
      nodes.resize(new_size);
    }

  public:
    impl()
    {
      // Stats are disabled by default, unless a non null value is passed to this env variable
      const char* display_graph_stats_str = getenv("CUDASTF_DISPLAY_GRAPH_STATS");
      display_graph_stats                 = (display_graph_stats_str && atoi(display_graph_stats_str) != 0);

      // Create the root node
      push(::cuda::std::source_location::current(), true /* is_root */);
    }

    ~impl()
    {
      print_cache_stats_summary();
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Non movable
    impl(impl&&) noexcept            = delete;
    impl& operator=(impl&&) noexcept = delete;

    /**
     * @brief Helper to get async handle from pool or create new one
     */
    template <typename ContextType>
    async_resources_handle get_async_handle(const ContextType& parent_ctx)
    {
      auto& handle_stack = async_handles[parent_ctx.default_exec_place()];
      if (handle_stack.empty())
      {
        return async_resources_handle{}; // Create new one when pool is empty
      }
      else
      {
        auto handle = mv(handle_stack.top());
        handle_stack.pop();
        return handle;
      }
    }

    /**
     * @brief Create a new nested level
     *
     * head_offset is the offset of thread's current top context (-1 if none)
     */
    void push(const ::cuda::std::source_location& loc,
              bool is_root                                     = false,
              [[maybe_unused]] const push_while_config& config = push_while_config{})
    {
      auto lock = acquire_exclusive_lock();

      // If we are creating the root context, we do not try to get some
      // uninitialized thread-local value.
      int head_offset = is_root ? -1 : get_head_offset();

      // Select the offset of the new node
      int node_offset = node_tree.get_avail_entry();

      // Grow nodes if needed
      if (int(nodes.size()) <= node_offset)
      {
        grow_context_nodes(node_offset + 1);
      }

      // Ensure the node offset was unused
      _CCCL_ASSERT(!nodes[node_offset], "inconsistent state");

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
      if (config.conditional_handle != nullptr)
      {
        _CCCL_ASSERT(head_offset != -1, "push_while cannot be used as root context - use push() for root");
      }
#endif

      if (head_offset == -1)
      {
        // If there is no current context, this is the root context and we use a stream_ctx
        // Create stream context node
        nodes[node_offset] = ::std::make_unique<stream_ctx_node>();

        // root of the context
        root_offset = node_offset;
      }
      else
      {
        // Keep track of parenthood
        node_tree.set_parent(head_offset, node_offset);

        // Create graph context node
        auto& parent_node = nodes[head_offset];
        auto& parent_ctx  = parent_node->ctx;
        auto handle       = get_async_handle(parent_ctx);

        // Create graph context node with optional conditional support
        nodes[node_offset] = ::std::make_unique<graph_ctx_node>(parent_node.get(), mv(handle), loc, config);
      }

      // Handle stats and update head
      if (display_graph_stats)
      {
        auto& new_node    = *nodes[node_offset];
        new_node.callsite = loc;
      }

      // Update the current context head
      set_head_offset(node_offset);
    }

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
    void push_while(cudaGraphConditionalHandle* phandle_out,
                    unsigned int default_launch_value      = 0,
                    unsigned int flags                     = cudaGraphCondAssignDefault,
                    const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
    {
      _CCCL_ASSERT(phandle_out != nullptr, "push_while requires non-null conditional handle output parameter");

      // Create config object and call push.
      push_while_config config(phandle_out, cudaGraphCondTypeWhile, default_launch_value, flags);
      push(loc, false, config);
    }
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

    // This method assumes that the mutex is already acquired in exclusive mode
    void _pop_prologue()
    {
      int head_offset = get_head_offset();

      _CCCL_ASSERT(nodes.size() > 0, "Calling pop while no context was pushed");
      _CCCL_ASSERT(nodes[head_offset], "invalid state");

      auto& current_node = nodes[head_offset];

      // Automatically pop data if needed. This will destroy the logical data
      // created within the context that is being destroyed. Unless using "non
      // exportable data" this logical data was created as an alias of an
      // another logical data that was frozen in the parent context. The
      // unfreeze operation is delayed after finalizing the context, in the
      // pop_after_finalize stage.
      for (auto& d_impl : current_node->pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_before_finalize(head_offset);
      }
    }

    // This method assumes that the mutex is already acquired in exclusive mode
    // The event_list correspond to the prereqs after we have finalized (eg.
    // launch a graph in a stream, or the child node)
    void _pop_epilogue(event_list& finalize_prereqs)
    {
      int head_offset = get_head_offset();

      auto& current_node = nodes[head_offset];
      auto& current_ctx  = current_node->ctx;

      // Make the async resource handle reusable on this execution place
      auto& handle_stack = async_handles[current_ctx.default_exec_place()];
      handle_stack.push(mv(current_node->async_handle.value()));

      if (display_graph_stats)
      {
        // When a graph context is finalized, a CUDA graph is created, we here
        // retrieve some information about it, and relate it to the location in
        // sources of the context push() call.
        executable_graph_cache_stat* stat = current_ctx.graph_get_cache_stat();
        _CCCL_ASSERT(stat, "");

        const auto& loc = current_node->callsite;
        stats_map[loc][::std::make_pair(stat->nnodes, stat->nedges)] += *stat;
      }

      int parent_offset = node_tree.get_parent(head_offset);
      _CCCL_ASSERT(parent_offset != -1, "internal error: no parent ctx");

      auto& parent_node = nodes[parent_offset];
      auto& parent_ctx  = parent_node->ctx;

      // Now that the context has been finalized, we can unfreeze data (unless
      // another sibling context has pushed it too)
      for (auto& d_impl : current_node->pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_after_finalize(parent_offset, finalize_prereqs);
      }

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_node->clear_adapters)
      {
        if (current_node->alloc_adapters)
        {
          current_node->alloc_adapters->clear();
        }

        for (auto& a : current_node->retained_adapters)
        {
          a->clear();
        }
      }
      else
      {
        // Transmit adapters to the parent node
        if (current_node->alloc_adapters)
        {
          parent_node->retained_adapters.push_back(current_node->alloc_adapters);
        }
        for (auto& a : current_node->retained_adapters)
        {
          parent_node->retained_adapters.push_back(a);
        }
      }

      // Destroy the current node
      nodes[head_offset].reset();

      // Since we have destroyed the node in the tree of contexts, we can put
      // it offsets back in the list of available offsets so that it can be
      // reused.
      node_tree.discard_node(head_offset);

      // The parent becomes the current context of the thread
      set_head_offset(parent_offset);
    }

    /**
     * @brief Terminate the current nested level and get back to the previous one
     */
    void pop()
    {
      auto lock = acquire_exclusive_lock();

      _pop_prologue();

      // Polymorphic finalization - no conditionals needed!
      int head_offset    = get_head_offset();
      auto& current_node = *nodes[head_offset];

      // Use polymorphic dispatch for context-specific finalization
      event_list finalize_prereqs = current_node.finalize();

      // Release all resources acquired for the push now that we have executed the graph
      _pop_epilogue(finalize_prereqs);
    }

    // Offset of the root context
    int get_root_offset() const
    {
      return root_offset;
    }

    context& get_root_ctx()
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset], "invalid state");
      return nodes[root_offset]->ctx;
    }

    const context& get_root_ctx() const
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset], "invalid state");
      return nodes[root_offset]->ctx;
    }

    ::std::unique_ptr<ctx_node_base>& get_node(int offset)
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset], "invalid value");
      return nodes[offset];
    }

    const ::std::unique_ptr<ctx_node_base>& get_node(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset], "invalid value");
      return nodes[offset];
    }

    context& get_ctx(int offset)
    {
      return get_node(offset)->ctx;
    }

    const context& get_ctx(int offset) const
    {
      return get_node(offset)->ctx;
    }

    /* Get the offset of the top context for the current thread */
    int get_head_offset() const
    {
      auto it = head_map.find(::std::this_thread::get_id());
      if (it == head_map.end())
      {
        fprintf(stderr, "Error: context offset isn't set in this thread\n");
        abort();
      }
      return it->second;
    }

    /** True if the current thread has a head offset set (has entered the context). */
    bool has_head_set() const
    {
      return head_map.find(::std::this_thread::get_id()) != head_map.end();
    }

    // Record the offset of the current context in the calling thread
    void set_head_offset(int offset)
    {
      head_map[::std::this_thread::get_id()] = offset;
    }

    int get_parent_offset(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "invalid node offset for parent lookup");
      return node_tree.get_parent(offset);
    }

    const auto& get_children_offsets(int parent) const
    {
      _CCCL_ASSERT(parent != -1, "invalid parent offset for children lookup");
      return node_tree.get_children(parent);
    }

  private:
    void print_cache_stats_summary() const
    {
      if (!display_graph_stats || stats_map.size() == 0)
      {
        return;
      }

      fprintf(stderr, "Executable Graph Cache Statistics Summary\n");
      fprintf(stderr, "=========================================\n");

      // Convert unordered map to ordered map for sorting by filename and line number
      using location_key_t = ::std::tuple<::std::string, int, ::std::string>; // (filename, line, function)
      ::std::map<location_key_t, stored_type_t> sorted_stats_map;

      for (const auto& [location, stat_map] : stats_map)
      {
        sorted_stats_map[{location.file_name(), location.line(), location.function_name()}] = stat_map;
      }

      for (const auto& [loc_key, stat_map] : sorted_stats_map)
      {
        const auto& [filename, line, function] = loc_key;

        fprintf(stderr, "Call-Site: %s:%d (%s)\n", filename.c_str(), line, function.c_str());

        // Convert unordered map to vector and sort by (nodes, edges)
        ::std::vector<::std::pair<::std::pair<size_t, size_t>, executable_graph_cache_stat>> sorted_stats(
          stat_map.begin(), stat_map.end());

        ::std::sort(sorted_stats.begin(), sorted_stats.end(), [](const auto& a, const auto& b) {
          return (a.first.first < b.first.first) || // Sort by nodes
                 (a.first.first == b.first.first && a.first.second < b.first.second); // Then by edges
        });

        fprintf(stderr, "  Nodes  Edges  InstantiateCnt  UpdateCnt\n");
        fprintf(stderr, "  --------------------------------------\n");

        for (const auto& [key, stat] : sorted_stats)
        {
          fprintf(stderr, "  %5zu  %5zu  %13zu  %9zu\n", key.first, key.second, stat.instantiate_cnt, stat.update_cnt);
        }
        fprintf(stderr, "\n");
      }
    }

    // Actual state for each node (which organization is dictated by node_tree)
    ::std::vector<::std::unique_ptr<ctx_node_base>> nodes;

    // Hierarchy of the context nodes
    node_hierarchy node_tree;

    int root_offset = -1;

    ::std::unordered_map<::std::thread::id, int> head_map;

    // Handles to retain some asynchronous states. This saves previously
    // instantiated graphs, or stream pools for example. We have a stack of
    // handles per execution place.
    ::std::map<exec_place, ::std::stack<async_resources_handle>> async_handles;

    bool display_graph_stats;

    // Create a map indexed by source locations, the value stored are a map of stats indexed per (nnodes,nedges) pairs
    using stored_type_t =
      ::std::unordered_map<::std::pair<size_t, size_t>, executable_graph_cache_stat, hash<::std::pair<size_t, size_t>>>;
    ::std::unordered_map<::cuda::std::source_location,
                         stored_type_t,
                         reserved::source_location_hash,
                         reserved::source_location_equal>
      stats_map;

  public:
    /* RAII wrappers to get the lock either in read-only or write mode */
    ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
    {
      return ::std::shared_lock<::std::shared_mutex>(mutex);
    }

    ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
    {
      return ::std::unique_lock<::std::shared_mutex>(mutex);
    }

  private:
    mutable ::std::shared_mutex mutex;
  };

  stackable_ctx()
      : pimpl(::std::make_shared<impl>())
  {}

  auto& get_node(size_t offset)
  {
    return pimpl->get_node(offset);
  }

  int get_parent_offset(int offset) const
  {
    return pimpl->get_parent_offset(offset);
  }

  const auto& get_children_offsets(int parent) const
  {
    return pimpl->get_children_offsets(parent);
  }

  context& get_root_ctx()
  {
    return pimpl->get_root_ctx();
  }

  const context& get_root_ctx() const
  {
    return pimpl->get_root_ctx();
  }

  int get_root_offset() const
  {
    return pimpl->get_root_offset();
  }

  context& get_ctx(int offset)
  {
    return pimpl->get_ctx(offset);
  }

  const context& get_ctx(int offset) const
  {
    return pimpl->get_ctx(offset);
  }

  int get_head_offset() const
  {
    auto lock = pimpl->acquire_shared_lock();
    return pimpl->get_head_offset();
  }

  /** True if the current thread has a head offset set (has entered the context). */
  bool has_head_set() const
  {
    auto lock = pimpl->acquire_shared_lock();
    return pimpl->has_head_set();
  }

  void set_head_offset(int offset)
  {
    auto lock = pimpl->acquire_exclusive_lock();
    pimpl->set_head_offset(offset);
  }

  void push(const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
  {
    pimpl->push(loc);
  }

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  void push_while(cudaGraphConditionalHandle* phandle_out,
                  unsigned int default_launch_value      = 0,
                  unsigned int flags                     = cudaGraphCondAssignDefault,
                  const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
  {
    pimpl->push_while(phandle_out, default_launch_value, flags, loc);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

  void pop()
  {
    pimpl->pop();
  }

  // RAII guard classes are defined as standalone types after the stackable_ctx
  // class (in stackable_ctx.cuh) so that the class body stays focused on core
  // logic.  The using-declarations below preserve the stackable_ctx::guard
  // nested-name syntax for backward compatibility.
  class graph_scope_guard;
#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  class while_graph_scope_guard;
#endif

  [[nodiscard]] graph_scope_guard
  graph_scope(const ::cuda::std::source_location& loc = ::cuda::std::source_location::current());

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  [[nodiscard]] while_graph_scope_guard while_graph_scope(
    unsigned int default_launch_value       = 1,
    unsigned int flags                      = cudaGraphCondAssignDefault,
    const ::cuda::std::source_location& loc = ::cuda::std::source_location::current());

  [[nodiscard]] auto repeat_graph_scope(
    size_t count,
    unsigned int default_launch_value       = 1,
    unsigned int flags                      = cudaGraphCondAssignDefault,
    const ::cuda::std::source_location& loc = ::cuda::std::source_location::current());
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

  template <typename T>
  auto logical_data(shape_of<T> s)
  {
    auto lock = pimpl->acquire_shared_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(*this, head, true, get_root_ctx().logical_data(mv(s)), true);
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    auto lock = pimpl->acquire_shared_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(
      *this, head, true, get_root_ctx().template logical_data<T>(elements, more_sizes...), true);
  }

  template <typename T>
  auto logical_data_no_export(shape_of<T> s)
  {
    auto lock = pimpl->acquire_shared_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(*this, head, true, get_ctx(head).logical_data(mv(s)), false);
  }

  template <typename T, typename... Sizes>
  auto logical_data_no_export(size_t elements, Sizes... more_sizes)
  {
    auto lock = pimpl->acquire_shared_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(
      *this, head, true, get_ctx(head).template logical_data<T>(elements, more_sizes...), false);
  }

  stackable_logical_data<void_interface> token();

  template <typename... Pack>
  auto logical_data(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int head           = pimpl->get_head_offset();
    auto underlying_ld = get_ctx(head).logical_data(::std::forward<Pack>(pack)...);
    using T            = typename decltype(underlying_ld)::element_type;
    return stackable_logical_data<T>(*this, head, false, mv(underlying_ld), true);
  }

  // Validate access modes and auto-push data to the correct context depth.
  // When the same logical data appears multiple times in a task's deps, access
  // modes are combined before pushing so that the data is imported with the
  // correct mode (e.g. read + write => rw).
  template <typename... Pack>
  void validate_and_push([[maybe_unused]] int offset, const Pack&... pack) const
  {
    ::std::vector<::std::pair<int, access_mode>> combined_accesses;

    [[maybe_unused]] auto combine = [&combined_accesses](const auto& arg) {
      if constexpr (reserved::is_stackable_task_dep_v<::std::decay_t<decltype(arg)>>)
      {
        combine_access_mode(combined_accesses, arg.get_d().get_unique_id(), arg.get_access_mode());
      }
    };

    [[maybe_unused]] auto validate = [&combined_accesses, offset, this](const auto& arg) {
      if constexpr (reserved::is_stackable_task_dep_v<::std::decay_t<decltype(arg)>>)
      {
        arg.get_d().validate_access(offset, *this, lookup_combined_mode(combined_accesses, arg.get_d().get_unique_id()));
      }
    };

    (combine(pack), ...);
    (validate(pack), ...);
  }

public:
  template <typename ExecPlace,
            typename... Deps,
            ::std::enable_if_t<::std::is_base_of_v<exec_place, ::std::decay_t<ExecPlace>>, int> = 0>
  auto task(ExecPlace&& e_place, Deps&&... deps)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    return deferred_task_builder{*this, offset, ::std::move(e_place), ::std::forward<Deps>(deps)...};
  }

  // Note we here duplicate the code above to avoid locking issues (and not create a 3 line common impl)
  template <typename... Deps>
  auto task(Deps&&... deps)
  {
    auto lock    = pimpl->acquire_shared_lock();
    int offset   = get_head_offset();
    auto e_place = get_ctx(offset).default_exec_place();
    return deferred_task_builder{*this, offset, ::std::move(e_place), ::std::forward<Deps>(deps)...};
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    return get_ctx(offset).parallel_for(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto launch(Pack&&... pack)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    return get_ctx(offset).launch(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel(Pack&&... pack)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    return get_ctx(offset).cuda_kernel(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel_chain(Pack&&... pack)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    return get_ctx(offset).cuda_kernel_chain(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }
#endif

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    return get_ctx(offset).host_launch(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }

  auto fence()
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    if (offset != get_root_offset())
    {
      fprintf(stderr, "Error: fence() not supported in nested contexts.\n");
      abort();
    }

    return get_ctx(offset).fence();
  }

  template <typename T>
  auto wait(::cuda::experimental::stf::stackable_logical_data<T>& ldata)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    if (offset != get_root_offset())
    {
      fprintf(stderr, "Error: wait() not supported in nested contexts.\n");
      abort();
    }

    return get_ctx(offset).wait(ldata.get_ld(offset));
  }

  auto get_dot()
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).get_dot();
  }

  template <typename... Pack>
  void push_affinity(Pack&&... pack) const
  {
    auto lock  = pimpl->acquire_shared_lock();
    int offset = get_head_offset();
    validate_and_push(offset, pack...);
    get_ctx(offset).push_affinity(reserved::resolve_dep(offset, ::std::forward<Pack>(pack))...);
  }

  void pop_affinity() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    get_ctx(offset).pop_affinity();
  }

  auto& current_affinity() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).current_affinity();
  }

  const exec_place& current_exec_place() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).current_exec_place();
  }

  auto& async_resources() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).async_resources();
  }

  auto dot_section(::std::string symbol) const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).dot_section(mv(symbol));
  }

  size_t task_count() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    return get_ctx(offset).task_count();
  }

  void finalize()
  {
    auto lock = pimpl->acquire_shared_lock();

    _CCCL_ASSERT(pimpl->get_head_offset() == pimpl->get_root_offset(),
                 "Can only finalize if there is no pending contexts");

    get_root_ctx().finalize();
  }

  ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
  {
    return pimpl->acquire_shared_lock();
  }

  ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
  {
    return pimpl->acquire_exclusive_lock();
  }

public:
  ::std::shared_ptr<impl> pimpl;
};
} // end namespace cuda::experimental::stf
