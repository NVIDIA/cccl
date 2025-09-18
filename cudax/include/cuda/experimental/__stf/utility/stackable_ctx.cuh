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
//! \brief Stackable context and logical data to nest contexts

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
#include <iostream>
#include <shared_mutex>
#include <stack>
#include <thread>

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/__stf/internal/task.cuh"
#include "cuda/experimental/__stf/utility/hash.cuh"
#include "cuda/experimental/__stf/utility/source_location.cuh"
#include "cuda/experimental/stf.cuh"

//! \brief Stackable Context Design Overview
//!
//! The stackable context allows nesting CUDA STF contexts to create hierarchical task graphs.
//! This enables complex workflows where tasks can be organized in a tree-like structure.
//!
//! Key concepts:
//! - **Context Stack**: Nested contexts form a stack where each level can have its own task graph
//! - **Data Movement**: Logical data can be imported ("pushed") between context levels automatically
//!
//! Usage pattern:
//! ```
//! stackable_ctx sctx;
//! auto data = sctx.logical_data(...);
//!
//! sctx.push();  // Enter nested context
//! data.push(access_mode::rw);  // Import data into nested context
//! // ... work with data in nested context ...
//! sctx.pop();   // Exit nested context, execute graph
//! ```
//!
//! By default, a task using a logical data in a nested context will
//! automatically issue a `push` in a `rw` mode. Advanced users can still push in
//! read-only mode to ensure data can be used concurrently from different nested
//! contexts.

namespace cuda::experimental::stf
{

namespace reserved
{

#if _CCCL_CTK_AT_LEAST(12, 4)
// This kernel is used by the update_cond method to update the conditional handle. The device function passed as an
// argument returns a boolean value which defines the new value of the conditional handle.
template <typename CondFunc, typename... Args>
__global__ void condition_update_kernel(cudaGraphConditionalHandle conditional_handle, CondFunc cond_func, Args... args)
{
  // Direct call to the user's condition function - no lambda nesting
  bool result = cond_func(args...);
  cudaGraphSetConditional(conditional_handle, result);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

} // end namespace reserved

template <typename T>
class stackable_logical_data;

template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep;

//! \brief Configuration class for push_while operations
//!
//! This class encapsulates the parameters needed for conditional graph nodes.
//! When CUDA 12.4+ is not available, it becomes effectively empty.
struct push_while_config
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  cudaGraphConditionalHandle* conditional_handle     = nullptr;
  enum cudaGraphConditionalNodeType conditional_type = cudaGraphCondTypeWhile;
  unsigned int defaultLaunchValue                    = 1;
  unsigned int flags                                 = cudaGraphCondAssignDefault;

  push_while_config() = default;

  push_while_config(cudaGraphConditionalHandle* handle,
                    enum cudaGraphConditionalNodeType type = cudaGraphCondTypeWhile,
                    unsigned int launch_value              = 1,
                    unsigned int condition_flags           = cudaGraphCondAssignDefault)
      : conditional_handle(handle)
      , conditional_type(type)
      , defaultLaunchValue(launch_value)
      , flags(condition_flags)
  {}

#else
  // Empty implementation for pre-CUDA 12.4
  push_while_config() = default;

  // Constructor that ignores parameters for compatibility
  template <typename... Args>
  push_while_config(Args&&...)
  {}
#endif
};

namespace reserved
{

template <typename T>
struct is_stackable_task_dep : ::std::false_type
{};

template <typename T, typename ReduceOp, bool Init>
struct is_stackable_task_dep<stackable_task_dep<T, ReduceOp, Init>> : ::std::true_type
{};

template <typename T>
inline constexpr bool is_stackable_task_dep_v = is_stackable_task_dep<T>::value;

// This helper converts stackable_task_dep to the underlying task_dep. If we
// have a stackable_logical_data A, A.read() is indeed a stackable_task_dep,
// which we can pass to stream_ctx/graph_ctx constructs by extracting the
// underlying task_dep.
template <typename U>
decltype(auto) to_task_dep(U&& u)
{
  if constexpr (is_stackable_task_dep_v<::std::decay_t<U>>)
  {
    return ::std::forward<U>(u).underlying_dep();
  }
  else
  {
    return ::std::forward<U>(u);
  }
}

} // end namespace reserved

//! \brief Base class with a virtual pop method to enable type erasure
//!
//! This is used to implement the automatic call to pop() on logical data when
//! a context node is popped, as we need to keep a vector of logical data that
//! were imported without knowing their type.
//!
class stackable_logical_data_impl_state_base
{
public:
  virtual ~stackable_logical_data_impl_state_base()                                            = default;
  virtual void pop_before_finalize(int ctx_offset) const                                       = 0;
  virtual void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) const = 0;
};

//! \brief This class defines a context that behaves as a context which can have nested subcontexts (implemented as
//! local CUDA graphs)
class stackable_ctx
{
public:
  //! Store metadata about task dependencies to automatically push data before
  //! the task is started.
  //!
  //! This is needed to implement add_deps, where dependencies are discovered
  //! incrementally and where importing a piece of data in a write-only mode, and
  //! then the same data in read mode would result in incorrect behaviour, while
  //! we expect to import it in rw mode in this scenario.

  // Information about a deferred argument that needs add_deps at task execution time
  struct deferred_arg_info
  {
    int logical_data_id;
    access_mode combined_access_mode;
    // Store function that adds dependency to underlying task (validation already done)
    ::std::function<void()> add_dependency_to_task;
  };

  // Deferred task builder using stored lambda approach
  template <typename... InitialPack>
  class deferred_task_builder
  {
  public:
    stackable_ctx& sctx_;
    int offset_;

    // Store the original parameter pack for task recreation
    ::std::tuple<InitialPack...> initial_pack_;

    // Store additional dependencies with captured operations
    struct additional_dep_info
    {
      int logical_data_id;
      access_mode mode;
      // Store just the essential operations as simple lambdas
      ::std::function<void(stackable_ctx&, int, access_mode)> validate_access_op;
      ::std::function<task_dep_untyped(access_mode)> create_task_dep_op;
    };
    ::std::vector<additional_dep_info> additional_deps_;

    // Store the concrete task (base class)
    mutable ::std::optional<::cuda::experimental::stf::task> concrete_task_;

    // Optional symbol to be applied to the underlying task when concretized
    ::std::optional<::std::string> symbol_;

    deferred_task_builder(stackable_ctx& sctx, int offset, InitialPack&&... pack)
        : sctx_(sctx)
        , offset_(offset)
        , initial_pack_(::std::forward<InitialPack>(pack)...)
    {}

    // Add more dependencies via add_deps
    template <typename... Deps>
    auto& add_deps(Deps&&... deps)
    {
      auto store_dep = [this](const auto& dep) {
        static_assert(reserved::is_stackable_task_dep_v<::std::decay_t<decltype(dep)>>,
                      "add_deps in stackable context only accepts stackable task dependencies");

        // Store metadata and create operations
        additional_dep_info info;
        info.logical_data_id = dep.get_d().get_unique_id();
        info.mode            = dep.get_access_mode();

        // Create operations with direct member function calls
        auto logical_data       = dep.get_d();
        info.validate_access_op = [logical_data](stackable_ctx& sctx, int offset, access_mode mode) mutable {
          logical_data.validate_access(offset, sctx, mode);
        };

        info.create_task_dep_op = [logical_data, this](access_mode mode) mutable -> task_dep_untyped {
          // Create dependency with proper context offset for deferred processing
          return logical_data.get_dep_with_mode(mode).to_task_dep_with_offset(this->offset_);
        };

        additional_deps_.push_back(::std::move(info));
      };
      (store_dep(deps), ...);
      return *this;
    }

  private:
    // Concretize the deferred task - process all dependencies and create the final concrete task
    template <typename TaskAction>
    auto concretize_deferred_task(TaskAction&& action) const
    {
      return ::std::apply(
        [this, &action](auto&&... initial_args) {
          // Combine access modes for all dependencies (initial + additional)
          ::std::map<int, access_mode> combined_modes;

          // Process initial arguments
          if constexpr (sizeof...(initial_args) > 0)
          {
            auto process_initial = [&](const auto& arg) {
              static_assert(reserved::is_stackable_task_dep_v<::std::decay_t<decltype(arg)>>,
                            "Initial task arguments in stackable context must be stackable task dependencies");
              int id             = arg.get_d().get_unique_id();
              combined_modes[id] = combined_modes[id] | arg.get_access_mode();
            };
            (process_initial(initial_args), ...);
          }

          // Process additional dependencies from add_deps
          for (const auto& info : additional_deps_)
          {
            combined_modes[info.logical_data_id] = combined_modes[info.logical_data_id] | info.mode;
          }

          // Validate access for ALL dependencies (initial + additional) with combined modes

          // First validate initial arguments with their combined access modes
          if constexpr (sizeof...(initial_args) > 0)
          {
            auto validate_initial = [&](const auto& arg) {
              int id             = arg.get_d().get_unique_id();
              auto combined_mode = combined_modes[id];
              // Validate initial argument with combined mode
              arg.get_d().validate_access(offset_, sctx_, combined_mode);
            };
            (validate_initial(initial_args), ...);
          }

          // Then validate additional dependencies from add_deps
          for (const auto& info : additional_deps_)
          {
            auto combined_mode = combined_modes[info.logical_data_id];
            // Use the stored operation - needed for automatic push pattern (stackable2.cu style)
            info.validate_access_op(sctx_, offset_, combined_mode);
          }

          // Create task with original arguments
          auto task = [&]() {
            // For virtual contexts, we need to use the real context for execution
            int exec_offset = sctx_.resolve_to_real_context(offset_);
            if constexpr (sizeof...(initial_args) > 0)
            {
              // Use real context offset for execution but keep offset_ for dependencies
              return sctx_.get_ctx(exec_offset).task(initial_args.to_task_dep_with_offset(offset_)...);
            }
            else
            {
              return sctx_.get_ctx(exec_offset).task();
            }
          }();

          // Add the additional dependencies from add_deps calls to the underlying task
          for (const auto& info : additional_deps_)
          {
            auto combined_mode = combined_modes[info.logical_data_id];

            // Use the stored operation to create and add the dependency
            task.add_deps(info.create_task_dep_op(combined_mode));
          }

          // Apply symbol if it was set
          if (symbol_.has_value())
          {
            task.set_symbol(*symbol_);
          }

          // Extract the base task from unified_task using the new get_base_task() method
          concrete_task_ = task.get_base_task();

          // Execute the provided action with the fully constructed task
          return action(task);
        },
        initial_pack_);
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
      virtual void finalize() = 0;
      virtual void cleanup()  = 0;
      virtual bool is_virtual() const
      {
        return false;
      }

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

      context ctx;
      cudaStream_t support_stream;
      // A wrapper to forward allocations from a node to its parent node (none is used at the root level)
      ::std::shared_ptr<stream_adapter> alloc_adapters;

      // This keeps track of the logical data that were pushed in this ctx node
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> pushed_data;

      // Where was the push() called ?
      _CUDA_VSTD::source_location callsite;

      // The async resource handle used in this context
      ::std::optional<async_resources_handle> async_handle;

      // Collection of events to start the context (based on the freeze
      // operations to get data imported into the context)
      event_list ctx_prereqs;

    protected:
      // If we want to keep the state of some logical data implementations until this node is popped
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> retained_data;
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

      void finalize() override
      {
        ctx.finalize();
      }

      void cleanup() override
      {
        // Stream context cleanup is mostly handled by finalize()
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
                     const _CUDA_VSTD::source_location& loc,
                     [[maybe_unused]] const push_while_config& config = push_while_config{})
      {
        // Graph context nodes create and set up the internal CUDA graph
        _CCCL_ASSERT(parent_node != nullptr, "Graph context must have a valid parent");

        auto& parent_ctx         = parent_node->ctx;
        cudaGraph_t parent_graph = parent_node->get_graph();

        /* The parent is either a stream_ctx, or a graph itself. If this is a graph, we either add a new child graph, or
         * a conditional node. */

        if (parent_graph)
        {
          nested_graph = true;
#if _CCCL_CTK_AT_LEAST(12, 4)
          if (config.conditional_handle != nullptr)
          {
            // We will add a conditional node to an existing graph, we do not create a new graph.
            fprintf(stderr, "PUSHING CTX : graph context parent => create cond graph node\n");
            graph = parent_graph;
          }
          else
#endif // _CCCL_CTK_AT_LEAST(12, 4)
          {
            fprintf(stderr, "PUSHING CTX : graph context parent => create child graph node\n");

            cuda_safe_call(cudaGraphCreate(&graph, 0));

            cudaGraphNode_t n;
            cuda_safe_call(cudaGraphAddChildGraphNode(&n, parent_graph, nullptr, 0, graph));
          }
        }
        else
        {
          fprintf(stderr, "PUSHING CTX : stream context parent => create new graph\n");
          cuda_safe_call(cudaGraphCreate(&graph, 0));
        }

        // This is the graph which will be used by our STF context. If we need
        // to use a conditional node later, will will change that value.
        cudaGraph_t sub_graph = graph;

#if _CCCL_CTK_AT_LEAST(12, 4)
        if (config.conditional_handle != nullptr)
        {
          // Create the conditional handle and store it in the provided pointer
          cuda_safe_call(cudaGraphConditionalHandleCreate(
            config.conditional_handle, graph, config.defaultLaunchValue, config.flags));

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
        }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

        // Get a stream from previous context
        cudaStream_t stream = parent_ctx.pick_stream();
        support_stream      = stream; // Update our stream

        // Make sure the stream depends on completion of events to import data
        parent_node->ctx_prereqs.sync_with_stream(parent_ctx.get_backend(), stream);

        auto gctx = graph_ctx(sub_graph, stream, handle);

        // Set up context properties
        gctx.set_parent_ctx(parent_ctx);
        gctx.get_dot()->set_ctx_symbol("graph[" + ::std::string(loc.file_name()) + ":" + ::std::to_string(loc.line())
                                       + "(" + loc.function_name() + ")]");

        // Create stream adapter and update allocator
        alloc_adapters = ::std::make_shared<stream_adapter>(gctx, stream);
        gctx.update_uncached_allocator(alloc_adapters->allocator());

        // Set the context and save async handle
        ctx          = gctx;
        async_handle = mv(handle);
      }

      void finalize() override
      {
        // Use finalize_as_graph pattern for explicit graph management
        _CCCL_ASSERT(ctx.is_graph_ctx(), "graph_ctx_node must contain a graph context");

        ctx.to_graph_ctx().finalize_as_graph();

        // Debug: Print DOT output of the finalized graph
        if (getenv("CUDASTF_DEBUG_STACKABLE_DOT"))
        {
          static int debug_graph_cnt = 0; // Warning: not thread-safe
          ::std::string filename     = "stackable_graph_" + ::std::to_string(debug_graph_cnt++) + ".dot";
          cudaGraphDebugDotPrint(graph, filename.c_str(), cudaGraphDebugDotFlags(0));
          ::std::cout << "Debug: Stackable graph DOT output written to " << filename << ::std::endl;
        }

        // This was either a new graph (with a stream_ctx parent) or a nested
        // graph ctx node. If this was a nested context, we do not need to
        // instantiate and launch, but we need to enforce dependencies by
        // adding input deps
        if (nested_graph)
        {
          fprintf(stderr, "TODO nested contexts are WIP\n");
          abort();
          return;
        }

        // Since this is not a nested context, we had to create a new graph for
        // this, and we have to launch it after instantiating it.

        // Create executable graph with enhanced error reporting
#if _CCCL_CTK_AT_LEAST(11, 4)
        cudaGraphInstantiateParams instantiate_params = {};
        instantiate_params.flags                      = 0; // No special flags for now

        cudaError_t instantiate_error = cudaGraphInstantiateWithParams(&graph_exec, graph, &instantiate_params);

        if (instantiate_error != cudaSuccess)
        {
          cudaGraphNode_t error_node        = instantiate_params.errNode_out;
          cudaGraphInstantiateResult result = instantiate_params.result_out;

          ::std::cerr << "Error: Graph instantiation failed with error: " << cudaGetErrorString(instantiate_error)
                      << " " << static_cast<int>(result) << ::std::endl;

          if (error_node != nullptr)
          {
            cudaGraphNodeType node_type;
            cudaError_t node_type_error = cudaGraphNodeGetType(error_node, &node_type);
            if (node_type_error == cudaSuccess)
            {
              ::std::cerr << "Error: Failed at node of type: " << node_type << ::std::endl;
            }
            else
            {
              ::std::cerr << "Error: Failed at node (node type query failed: " << cudaGetErrorString(node_type_error)
                          << ")" << ::std::endl;
            }
          }
          else
          {
            fprintf(stderr, "error_node = nullptr\n");
          }

          if (result != cudaGraphInstantiateSuccess)
          {
            ::std::cerr << "Error: Instantiation result code: " << result << ::std::endl;
          }
          cuda_safe_call(instantiate_error); // This will throw/abort
        }
#else
        // Fallback to basic instantiation for older CUDA versions
        cuda_safe_call(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
#endif

        // Launch the graph
        cuda_safe_call(cudaGraphLaunch(graph_exec, support_stream));
      }

      void cleanup() override
      {
        // TODO if nested: transfer resources to parent context ?
        if (nested_graph)
        {
          // TODO transfer resources if doable (or better)

          // TODO do we need to destroy some graph ?

          fprintf(stderr, "TODO: implement cleanup() for nested graph ctx nodes\n");
          return;
        }

        // Release context resources after graph execution
        ctx.release_resources(support_stream);

        // Cleanup executable graph
        if (graph_exec != nullptr)
        {
          cuda_safe_call(cudaGraphExecDestroy(graph_exec));
          graph_exec = nullptr;
        }

        // Cleanup parent graph if it exists (conditional handle is automatically cleaned up)
        _CCCL_ASSERT(graph != nullptr, "graph should have been created");
        cuda_safe_call(cudaGraphDestroy(graph));
        graph = nullptr;
      }

      cudaGraph_t get_graph() const override
      {
        return graph;
      }

    private:
      cudaGraphExec_t graph_exec = nullptr;
      cudaGraph_t graph = nullptr; // Graph containing conditional node (if conditional) or the entire graph otherwise
      bool nested_graph = false;
    };

    // Configuration constants
    static constexpr int initial_node_pool_size       = 16;
    static constexpr size_t growth_factor_numerator   = 3;
    static constexpr size_t growth_factor_denominator = 2;

    // Centralized method to grow context nodes to a certain size
    void grow_context_nodes(int target_size,
                            size_t factor_numerator   = growth_factor_numerator,
                            size_t factor_denominator = growth_factor_denominator)
    {
      if (target_size < int(nodes.size()))
      {
        return; // Already large enough
      }

      size_t new_size =
        ::std::max(static_cast<size_t>(target_size), nodes.size() * factor_numerator / factor_denominator);
      nodes.resize(new_size);
    }

    /* To describe the hierarchy of contexts, and the hierarchy of stackable
     * logical data which should match the structure of the context hierarchy,
     * this class describes a tree using a vector of offsets. Every node has an
     * offset, and we keep track of the parent offset of each node, as well as
     * its children. */
    class node_hierarchy
    {
    public:
      node_hierarchy()
      {
        // May grow up later if more contexts are needed
        constexpr int initialize_size = initial_node_pool_size;
        parent.resize(initialize_size);
        children.resize(initialize_size);
        free_list.reserve(initialize_size);

        for (int i = 0; i < initialize_size; i++)
        {
          // The order of the nodes does not really matter, but it may be
          // easier to follow if we get nodes in a reasonable order (sequence
          // from 0 ...)
          free_list.push_back(initialize_size - 1 - i);
        }
      }

      // Try to find an offset which is not used
      int get_avail_entry()
      {
        // XXX implement growth mechanism
        // remember size of parent, push new items in the free list ? (grow())
        _CCCL_ASSERT(!free_list.empty(), "no slot available");

        int result = free_list.back();
        free_list.pop_back();

        // the node should be unused
        _CCCL_ASSERT(children[result].empty(), "invalid state");
        parent[result] = -1;

        return result;
      }

      // When a node of the tree is destroyed, we can reuse its offset by
      // putting it back in the list of available offsets which can describe a
      // node.
      void discard_node(int offset)
      {
        nvtx_range r("discard_node");

        // Remove this child from its parent (if any)
        int p = parent[offset];
        if (p != -1)
        {
          // Use efficient erase-remove idiom instead of rebuilding vector
          auto& parent_children = children[p];
          auto it               = ::std::find(parent_children.begin(), parent_children.end(), offset);
          _CCCL_ASSERT(it != parent_children.end(), "invalid hierarchy state");
          parent_children.erase(it);
        }

        children[offset].clear();
        parent[offset] = -1;

        // Make this offset available again
        free_list.push_back(offset);
      }

      void set_parent(int parent_offset, int child_offset)
      {
        parent[child_offset] = parent_offset;
        children[parent_offset].push_back(child_offset);
      }

      int get_parent(int offset) const
      {
        _CCCL_ASSERT(offset < int(parent.size()), "node offset exceeds parent array size");
        return parent[offset];
      }

      const auto& get_children(int offset) const
      {
        _CCCL_ASSERT(offset < int(children.size()), "node offset exceeds children array size");
        return children[offset];
      }

      size_t depth(int offset) const
      {
        int p = get_parent(offset);
        if (p == -1)
        {
          return 0;
        }

        return 1 + depth(p);
      }

    private:
      // Offset of the node's parent : -1 if none. Only valid for entries not in free-list.
      ::std::vector<int> parent;

      // If a node has children, indicate their offset here
      ::std::vector<::std::vector<int>> children;

      // Available offsets to create new nodes
      ::std::vector<int> free_list;
    };

  public:
    impl()
    {
      // Stats are disabled by default, unless a non null value is passed to this env variable
      const char* display_graph_stats_str = getenv("CUDASTF_DISPLAY_GRAPH_STATS");
      display_graph_stats                 = (display_graph_stats_str && atoi(display_graph_stats_str) != 0);

      // Create the root node
      push(_CUDA_VSTD::source_location::current(), true /* is_root */);
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
    void push(const _CUDA_VSTD::source_location& loc,
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
      _CCCL_ASSERT(!nodes[node_offset].has_value(), "inconsistent state");

#if _CCCL_CTK_AT_LEAST(12, 4)
      // Additional validation for push_while (when conditional_handle is provided)
      if (config.conditional_handle != nullptr)
      {
        int parent_depth = is_root ? -1 : int(node_tree.depth(head_offset));

        // push_while cannot be used as root context - must have a parent
        _CCCL_ASSERT(head_offset != -1, "push_while cannot be used as root context - use push() for root");

        // push_while is not for nested contexts yet - parent must be depth 0 (simple hierarchy)
        _CCCL_ASSERT(parent_depth == 0, "push_while can only be used with depth 0 parent - not for nested contexts");
      }
#endif

      if (head_offset == -1)
      {
        // If there is no current context, this is the root context and we use a stream_ctx
        // Create stream context node
        nodes[node_offset].emplace(::std::make_unique<stream_ctx_node>());

        // root of the context
        root_offset = node_offset;
      }
      else
      {
        // In the current implementation, depth > 1 is using a no-op for push/pop
        ///   _CCCL_ASSERT(parent_depth == 0, "invalid state");
        // Keep track of parenthood
        node_tree.set_parent(head_offset, node_offset);

        // Create graph context node
        auto& parent_node = nodes[head_offset].value();
        auto& parent_ctx  = parent_node->ctx;
        auto handle       = get_async_handle(parent_ctx);

        // Create graph context node with optional conditional support
        nodes[node_offset].emplace(::std::make_unique<graph_ctx_node>(parent_node.get(), mv(handle), loc, config));
      }

      // Handle stats and update head
      if (display_graph_stats)
      {
        auto& new_node    = *nodes[node_offset].value();
        new_node.callsite = loc;
      }

      // Update the current context head
      set_head_offset(node_offset);
    }

#if _CCCL_CTK_AT_LEAST(12, 4)
    void push_while(cudaGraphConditionalHandle* phandle_out,
                    unsigned int defaultLaunchValue       = 0,
                    unsigned int flags                    = cudaGraphCondAssignDefault,
                    const _CUDA_VSTD::source_location loc = _CUDA_VSTD::source_location::current())
    {
      _CCCL_ASSERT(phandle_out != nullptr, "push_while requires non-null conditional handle output parameter");

      // Create config object and call push.
      push_while_config config(phandle_out, cudaGraphCondTypeWhile, defaultLaunchValue, flags);
      push(loc, false, config);
    }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

    // This method assumes that the mutex is already acquired in exclusive mode
    void _pop_prologue()
    {
      int head_offset = get_head_offset();

      _CCCL_ASSERT(nodes.size() > 0, "Calling pop while no context was pushed");
      _CCCL_ASSERT(nodes[head_offset].has_value(), "invalid state");

      auto& current_node = nodes[head_offset].value();

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
    void _pop_epilogue()
    {
      int head_offset = get_head_offset();

      auto& current_node = nodes[head_offset].value();
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

      // To create prereqs that depend on this finalize() stage, we get the
      // stream used in this context, and insert events in it.
      cudaStream_t stream = current_node->support_stream;

      int parent_offset = node_tree.get_parent(head_offset);
      _CCCL_ASSERT(parent_offset != -1, "internal error: no parent ctx");

      auto& parent_node = nodes[parent_offset].value();
      auto& parent_ctx  = parent_node->ctx;

      // Create an event that depends on the completion of previous operations in the stream
      event_list finalize_prereqs = parent_ctx.stream_to_event_list(stream, "finalized");

      // Now that the context has been finalized, we can unfreeze data (unless
      // another sibling context has pushed it too)
      for (auto& d_impl : current_node->pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_after_finalize(parent_offset, finalize_prereqs);
      }

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_node->alloc_adapters)
      {
        current_node->alloc_adapters->clear();
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
      auto& current_node = *nodes[head_offset].value();

      // Use polymorphic dispatch for context-specific finalization
      current_node.finalize();
      current_node.cleanup();

      // Release all resources acquired for the push now that we have executed the graph
      _pop_epilogue();
    }

    // Offset of the root context
    int get_root_offset() const
    {
      return root_offset;
    }

    context& get_root_ctx()
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset].has_value(), "invalid state");
      return nodes[root_offset].value()->ctx;
    }

    const context& get_root_ctx() const
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset].has_value(), "invalid state");
      return nodes[root_offset].value()->ctx;
    }

    ::std::unique_ptr<ctx_node_base>& get_node(int offset)
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset].has_value(), "invalid value");
      return nodes[offset].value();
    }

    const ::std::unique_ptr<ctx_node_base>& get_node(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset].has_value(), "invalid value");
      return nodes[offset].value();
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
      _CCCL_ASSERT(it != head_map.end(), "ctx offset isn't set in that thread");
      return (it != head_map.end()) ? it->second : -1;
    }

    // Record the offset of the current context in the calling thread
    void set_head_offset(int offset)
    {
      head_map[::std::this_thread::get_id()] = offset;
    }

    // Helper method to traverse virtual contexts and find the real execution context
    int resolve_to_real_context(int start_offset) const
    {
      int current_offset = start_offset;
      while (is_virtual_context(current_offset))
      {
        current_offset = get_parent_offset(current_offset);
      }
      return current_offset;
    }

    // Get the current head offset and resolve it to the real context in one call
    int get_head_real_offset() const
    {
      return resolve_to_real_context(get_head_offset());
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

    bool is_virtual_context(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "invalid context offset");
      _CCCL_ASSERT(offset < int(nodes.size()), "context offset out of bounds");
      _CCCL_ASSERT(nodes[offset].has_value(), "context node doesn't exist");
      return nodes[offset].value()->is_virtual();
    }

  private:
    void print_logical_data_summary() const
    {
      traverse_nodes([this](int offset) {
        auto& ctx = get_ctx(offset);
        // fprintf(stderr, "[context %d (%s)] logical data summary:\n", offset, ctx.to_string().c_str());
        //   ctx.print_logical_data_summary();
      });
    }

    /* Recursively apply a function over every node of the context tree */
    template <typename Func>
    void traverse_nodes(Func&& func) const
    {
      ::std::stack<int> node_stack;
      node_stack.push(root_offset);

      while (!node_stack.empty())
      {
        int offset = node_stack.top();
        node_stack.pop();

        // Call the provided function on the current node
        func(offset);

        // Push children to stack (reverse order to maintain left-to-right order)
        const auto& children = get_children_offsets(offset);
        for (auto it = children.rbegin(); it != children.rend(); ++it)
        {
          node_stack.push(*it);
        }
      }
    }

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
    ::std::vector<::std::optional<::std::unique_ptr<ctx_node_base>>> nodes;

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
    ::std::unordered_map<_CUDA_VSTD::source_location,
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

    /* Methods to explicitly lock/unlock the mutex in a read-only or write way */
    void lock_shared() const
    {
      mutex.lock_shared();
    }

    void unlock_shared() const
    {
      mutex.unlock_shared();
    }

    void lock_exclusive()
    {
      mutex.lock();
    }

    void unlock_exclusive()
    {
      mutex.unlock();
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

  bool is_virtual_context(int offset) const
  {
    return pimpl->is_virtual_context(offset);
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
    return pimpl->get_head_offset();
  }

  int resolve_to_real_context(int offset) const
  {
    return pimpl->resolve_to_real_context(offset);
  }

  int get_head_real_offset() const
  {
    return pimpl->get_head_real_offset();
  }

  void set_head_offset(int offset)
  {
    pimpl->set_head_offset(offset);
  }

  void push(const _CUDA_VSTD::source_location loc = _CUDA_VSTD::source_location::current())
  {
    pimpl->push(loc);
  }

#if _CCCL_CTK_AT_LEAST(12, 4)
  void push_while(cudaGraphConditionalHandle* phandle_out,
                  unsigned int defaultLaunchValue       = 0,
                  unsigned int flags                    = cudaGraphCondAssignDefault,
                  const _CUDA_VSTD::source_location loc = _CUDA_VSTD::source_location::current())
  {
    pimpl->push_while(phandle_out, defaultLaunchValue, flags, loc);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  void pop()
  {
    pimpl->pop();
  }

  //! \brief RAII wrapper for automatic push/pop management (lock_guard style)
  //!
  //! This class provides automatic scope management for nested contexts,
  //! following the same semantics as std::lock_guard.
  //! The constructor calls push() and the destructor calls pop().
  //!
  //! Usage (direct constructor style like std::lock_guard):
  //! \code
  //! {
  //!   stackable_ctx::graph_scope_guard scope{ctx};
  //!   // nested context operations...
  //!   // pop() called automatically when scope goes out of scope
  //! }
  //! \endcode
  //!
  //! Usage (factory method style):
  //! \code
  //! {
  //!   auto scope = ctx.graph_scope();
  //!   // nested context operations...
  //! }
  //! \endcode
  class graph_scope_guard
  {
  public:
    using context_type = stackable_ctx;

    explicit graph_scope_guard(stackable_ctx& ctx,
                               const _CUDA_VSTD::source_location& loc = _CUDA_VSTD::source_location::current())
        : ctx_(ctx)
    {
      ctx_.push(loc);
    }

    ~graph_scope_guard()
    {
      ctx_.pop();
    }

    // Non-copyable, non-movable (like std::lock_guard)
    graph_scope_guard(const graph_scope_guard&)            = delete;
    graph_scope_guard& operator=(const graph_scope_guard&) = delete;
    graph_scope_guard(graph_scope_guard&&)                 = delete;
    graph_scope_guard& operator=(graph_scope_guard&&)      = delete;

  private:
    stackable_ctx& ctx_;
  };

#if _CCCL_CTK_AT_LEAST(12, 4)
  //! \brief RAII guard for while loop contexts with conditional graphs
  //!
  //! This guard automatically creates a while loop context using push_while() on construction
  //! and calls pop() on destruction. It provides access to the conditional handle created.
  class while_graph_scope_guard
  {
  public:
    using context_type = stackable_ctx;

    explicit while_graph_scope_guard(
      stackable_ctx& ctx,
      unsigned int defaultLaunchValue        = 0,
      unsigned int flags                     = cudaGraphCondAssignDefault,
      const _CUDA_VSTD::source_location& loc = _CUDA_VSTD::source_location::current())
        : ctx_(ctx)
    {
      ctx_.push_while(&conditional_handle_, defaultLaunchValue, flags, loc);
    }

    ~while_graph_scope_guard()
    {
      ctx_.pop();
    }

    //! \brief Get the conditional handle for controlling the while loop
    //! \return The conditional handle value
    cudaGraphConditionalHandle cond_handle() const
    {
      return conditional_handle_;
    }

    template <typename... Deps>
    class condition_update_scope
    {
    public:
      condition_update_scope(stackable_ctx& ctx, cudaGraphConditionalHandle handle, Deps... deps)
          : ctx_(ctx)
          , handle_(handle)
          , tdeps(mv(deps)...)
      {}

      // Helper to extract data_t from dependency types
      template <typename T>
      using data_t_of = typename T::data_t;

      template <typename CondFunc>
      void operator->*(CondFunc&& cond_func)
      {
        /* Build a cuda kernel from the deps then pass it a lambda that will
         * configure a call to the condition_update_kernel kernel */
        ::std::apply(
          [this](auto&&... deps) {
            return this->ctx_.cuda_kernel(deps...);
          },
          tdeps)
            ->*[cond_func = mv(cond_func), h = handle_](data_t_of<Deps>... args) {
                  return cuda_kernel_desc{
                    reserved::condition_update_kernel<CondFunc, data_t_of<Deps>...>, 1, 1, 0, h, cond_func, args...};
                };
      }

    private:
      stackable_ctx& ctx_;
      cudaGraphConditionalHandle handle_;
      ::std::tuple<::std::decay_t<Deps>...> tdeps;
    };

    //! \brief Helper for updating while loop condition using a device lambda
    //!
    //! Creates a helper object that supports the fluent ->* interface.
    //! The lambda should return true to continue the loop, false to exit.
    //!
    //! Example usage:
    //! ```cpp
    //! auto while_guard = ctx.while_graph_scope();
    //!
    //! // Simple condition check
    //! while_guard.update_cond(counter.read())->*[] __device__(auto counter) {
    //!   return *counter > 0; // Continue while counter > 0
    //! };
    //!
    //! // Complex condition with multiple data
    //! while_guard.update_cond(residual.read(), max_iter.read())->*[tol] __device__(auto residual, auto iter) {
    //!   bool converged = (*residual < tol);
    //!   bool max_reached = (*iter >= 1000);
    //!   return !converged && !max_reached; // Continue until converged or max iterations
    //! };
    //! ```
    template <typename... Args>
    auto update_cond(Args&&... args)
    {
      return condition_update_scope(ctx_, cond_handle(), args...); // Simple copy instead of forwarding
    }

    // Non-copyable, non-movable
    while_graph_scope_guard(const while_graph_scope_guard&)            = delete;
    while_graph_scope_guard& operator=(const while_graph_scope_guard&) = delete;
    while_graph_scope_guard(while_graph_scope_guard&&)                 = delete;
    while_graph_scope_guard& operator=(while_graph_scope_guard&&)      = delete;

  private:
    stackable_ctx& ctx_;
    cudaGraphConditionalHandle conditional_handle_{};
  };
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  //! \brief Create RAII scope that automatically handles push/pop
  //!
  //! Creates a graph_scope_guard object that calls push() on construction and pop() on destruction.
  //! The [[nodiscard]] attribute ensures the returned object is stored (not discarded),
  //! as discarding it would immediately call the destructor and pop() prematurely.
  //!
  //! \param loc Source location for debugging (defaults to call site)
  //! \return graph_scope_guard object that manages the nested context lifetime
  [[nodiscard]] auto graph_scope(const _CUDA_VSTD::source_location& loc = _CUDA_VSTD::source_location::current())
  {
    return graph_scope_guard(*this, loc);
  }

#if _CCCL_CTK_AT_LEAST(12, 4)
  //! \brief Create RAII scope for while loop contexts with conditional graphs
  //!
  //! Creates a while_graph_scope_guard object that calls push_while() on construction and pop() on destruction.
  //! The [[nodiscard]] attribute ensures the returned object is stored (not discarded),
  //! as discarding it would immediately call the destructor and pop() prematurely.
  //!
  //! Example usage:
  //! ```cpp
  //! stackable_ctx ctx;
  //! {
  //!   auto while_guard = ctx.while_graph_scope();
  //!   // Tasks added here are part of the while loop body
  //!   auto handle = while_guard.cond_handle();
  //!   // Use handle to control while loop execution
  //! } // Automatic pop() when while_guard goes out of scope
  //! ```
  //!
  //! \param defaultLaunchValue Default launch value for the conditional node (default: 1)
  //! \param flags Conditional flags for the while loop (default: cudaGraphCondAssignDefault)
  //! \param loc Source location for debugging (defaults to call site)
  //! \return while_graph_scope_guard object that manages the while context lifetime and provides access to the
  //! conditional handle
  [[nodiscard]] auto while_graph_scope(unsigned int defaultLaunchValue        = 1,
                                       unsigned int flags                     = cudaGraphCondAssignDefault,
                                       const _CUDA_VSTD::source_location& loc = _CUDA_VSTD::source_location::current())
  {
    return while_graph_scope_guard(*this, defaultLaunchValue, flags, loc);
  }

  //! \brief Create RAII scope for repeat loops with automatic counter management
  //!
  //! Creates a repeat_graph_scope_guard object that automatically manages the loop counter
  //! and conditional logic. The [[nodiscard]] attribute ensures the returned object is stored,
  //! as discarding it would immediately call the destructor and end the scope prematurely.
  //!
  //! This is a higher-level abstraction over while_graph_scope that automatically handles:
  //! - Counter initialization and management
  //! - Loop termination condition
  //! - Integration with the while graph system
  //!
  //! Example usage:
  //! ```cpp
  //! stackable_ctx ctx;
  //! auto data = ctx.logical_data(...);
  //!
  //! {
  //!   auto guard = ctx.repeat_graph_scope(10);
  //!   // Tasks added here will run 10 times
  //!   ctx.parallel_for(data.shape(), data.rw())->*[] __device__(size_t i, auto d) {
  //!     d(i) += 1.0;
  //!   };
  //! } // Automatic cleanup when guard goes out of scope
  //! ```
  //!
  //! \param count Number of iterations to repeat
  //! \param defaultLaunchValue Default launch value for the conditional node (default: 1)
  //! \param flags Conditional flags for the while loop (default: cudaGraphCondAssignDefault)
  //! \param loc Source location for debugging (defaults to call site)
  //! \return repeat_graph_scope_guard object that manages the repeat loop lifetime
  [[nodiscard]] auto repeat_graph_scope(
    size_t count,
    unsigned int defaultLaunchValue        = 1,
    unsigned int flags                     = cudaGraphCondAssignDefault,
    const _CUDA_VSTD::source_location& loc = _CUDA_VSTD::source_location::current());
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  template <typename T>
  auto logical_data(shape_of<T> s)
  {
    auto lock = pimpl->acquire_shared_lock();

    // fprintf(stderr, "initialize from shape.\n");
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

    // fprintf(stderr, "initialize from shape.\n");
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

    int head = pimpl->get_head_offset();
    // fprintf(stderr, "initialize from value.\n");
    auto underlying_ld = get_ctx(head).logical_data(::std::forward<Pack>(pack)...);
    using T            = typename decltype(underlying_ld)::element_type;
    return stackable_logical_data<T>(*this, head, false, mv(underlying_ld), true);
  }

  // This is the function that should be called before starting a task to push
  // data at the proper depth, or automatically push them at the appropriate
  // depth if necessary. If this happens, we may update the task_dep objects to
  // reflect the actual logical data that needs to be used.
  template <typename... Pack>
  void process_pack([[maybe_unused]] int offset, const Pack&... pack) const
  {
    // This is a map of logical data, and the combined access modes
    ::std::vector<::std::pair<int, access_mode>> combined_accesses;

    // Lambda to combine argument access modes
    [[maybe_unused]] auto combine_access_modes = [&combined_accesses](const auto& arg) {
      if constexpr (reserved::is_stackable_task_dep_v<::std::decay_t<decltype(arg)>>)
      {
        int id        = arg.get_d().get_unique_id();
        access_mode m = arg.get_access_mode();

        auto it = ::std::find_if(
          combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
            return entry.first == id;
          });

        if (it != combined_accesses.end())
        {
          it->second = it->second | m; // Update if found
        }
        else
        {
          combined_accesses.emplace_back(id, m); // Insert if not found
        }
      }
      // Do nothing for non-stackable types
    };

    // Lambda to process individual arguments
    [[maybe_unused]] auto process_argument = [&combined_accesses, offset, this](const auto& arg) {
      if constexpr (reserved::is_stackable_task_dep_v<::std::decay_t<decltype(arg)>>)
      {
        // If the stackable logical data appears in multiple deps of the same
        // task, we need to combine access modes to push the data automatically
        // with an appropriate mode.
        int id  = arg.get_d().get_unique_id();
        auto it = ::std::find_if(
          combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
            return entry.first == id;
          });
        _CCCL_ASSERT(it != combined_accesses.end(), "internal error");
        access_mode combined_m = it->second;

        // If the logical data was not at the appropriate level, we may
        // automatically push it. In this case, we need to update the logical data
        // referenced in the task_dep object to point to the correct context level.
        arg.get_d().validate_access(offset, *this, combined_m);
        {
          // Update the underlying task_dep to reference the correct logical_data
          // after automatic push. This uses the existing update_data mechanism
          // which is designed for in-place mutation in immediate processing contexts.
          arg.underlying_dep().update_data(arg.get_d().get_ld(offset));
        }
      }
      // Do nothing for non-stackable types
    };

    // First pass: combine access modes for all stackable dependencies
    (combine_access_modes(pack), ...);

    // Second pass: process each argument with combined access modes
    (process_argument(pack), ...);
  }

public:
  // Wrapper task that wraps the underlying task and handles deferred data pushes
  //
  // There is a significant complication with dynamic tasks where we can add
  // deps dynamically because one may have a write-only access on a logical
  // data, and then a read-only access on the same logical data which will be
  // transformed to a rw access: it is thus not possible to eagerly import
  // logical data from a parent context, and we have to entirely defer the
  // creation of the task until we know all dependencies.
  template <typename UnderlyingTask>
  class stackable_wrapper_task
  {
  public:
    stackable_wrapper_task(
      stackable_ctx& sctx, int offset, ::std::vector<deferred_arg_info> deferred_deps, UnderlyingTask underlying_task)
        : sctx_(sctx)
        , offset_(offset)
        , deferred_deps_(::std::move(deferred_deps))
        , underlying_task_(::std::move(underlying_task))
    {}

    // Handle add_deps - defer processing until task execution
    //
    // NOTE: We cannot immediately add dependencies to the underlying task here because:
    // 1. The logical data may need to be pushed to the correct context level first
    // 2. Multiple add_deps calls with the same logical data need access mode combining
    // 3. Data validation and pushing must happen before the underlying task can use it
    //
    // Instead, we collect all dependencies and process them in operator->* or start()
    template <typename... Deps>
    auto& add_deps(Deps&&... deps)
    {
      // Process each dependency - deferred until task execution
      (add_single_dep(::std::forward<Deps>(deps)), ...);
      return *this;
    }

    // Forward operator->* for task execution - process all deferred deps first, then execute
    template <typename F>
    auto operator->*(F&& f)
    {
      process_deferred_dependencies();
      // Execute the underlying task (dependencies were already set up when task was created)
      return underlying_task_->*::std::forward<F>(f);
    }

    // Forward other task methods to the underlying task
    auto& set_symbol(const ::std::string& symbol)
    {
      underlying_task_.set_symbol(symbol);
      return *this;
    }

    auto get_symbol() const
    {
      return underlying_task_.get_symbol();
    }

    // Forward get<T>(index) method for accessing task dependencies by index
    template <typename T>
    auto get(size_t index) const
    {
      return underlying_task_.template get<T>(index);
    }

    // Forward start() method to the underlying task
    auto& start()
    {
      process_deferred_dependencies();
      underlying_task_.start();
      return *this;
    }

    // Forward end() method to the underlying task
    auto& end()
    {
      underlying_task_.end();
      return *this;
    }

  private:
    // Process all deferred dependencies - add to underlying task (validation already done)
    void process_deferred_dependencies()
    {
      if (!deferred_processed_)
      {
        for (const auto& deferred_dep : deferred_deps_)
        {
          // Add dependency to underlying task (validation was already done at add_deps time)
          deferred_dep.add_dependency_to_task();
        }
        deferred_processed_ = true;
      }
    }

    // Handle a single dependency in add_deps - merge with existing deferred deps
    template <typename Dep>
    void add_single_dep(Dep&& dep)
    {
      // In stackable context, add_deps only receives stackable task dependencies
      static_assert(reserved::is_stackable_task_dep_v<::std::decay_t<Dep>>,
                    "add_deps in stackable context only accepts stackable task dependencies");

      auto& stackable_dep = dep;
      int id              = stackable_dep.get_d().get_unique_id();
      access_mode mode    = stackable_dep.get_access_mode();

      // Find existing entry for this logical data ID and merge access modes
      auto it = ::std::find_if(deferred_deps_.begin(), deferred_deps_.end(), [id](const deferred_arg_info& info) {
        return info.logical_data_id == id;
      });

      if (it != deferred_deps_.end())
      {
        // Merge access modes: write | read = rw, etc.
        access_mode old_mode     = it->combined_access_mode;
        it->combined_access_mode = it->combined_access_mode | mode;

        // Update the add_dependency function to use the new combined mode
        auto logical_data         = stackable_dep.get_d();
        access_mode combined_mode = it->combined_access_mode;

        // Defer the add_deps call to the underlying task (validation done in concretize_deferred_task)
        it->add_dependency_to_task = [this, logical_data, combined_mode]() mutable {
          // Create appropriate task_dep with stored context offset (not get_head_offset()!)
          auto task_dep_ref = logical_data.get_dep_with_mode(combined_mode).to_task_dep_with_offset(offset_);
          underlying_task_.add_deps(task_dep_ref);
        };
      }
      else
      {
        // Create new deferred processing info
        deferred_arg_info info;
        info.logical_data_id      = id;
        info.combined_access_mode = mode;

        // Capture just the lightweight stackable logical data
        auto logical_data = stackable_dep.get_d();

        // Defer the add_deps call to the underlying task (validation done in concretize_deferred_task)
        info.add_dependency_to_task = [this, logical_data, mode]() mutable {
          // Create appropriate task_dep with stored context offset (not get_head_offset()!)
          auto task_dep_ref = logical_data.get_dep_with_mode(mode).to_task_dep_with_offset(offset_);
          underlying_task_.add_deps(task_dep_ref);
        };

        deferred_deps_.push_back(::std::move(info));
      }

      // Reset the processed flag since we have new/updated dependencies
      deferred_processed_ = false;
    }

    stackable_ctx& sctx_;
    int offset_;
    ::std::vector<deferred_arg_info> deferred_deps_; // All deferred stackable deps
    UnderlyingTask underlying_task_;
    mutable bool deferred_processed_ = false; // Flag to ensure deferred deps are processed only once
  };

  template <typename... Pack>
  auto task(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();

    // Do not build the task now, we defer this when all dependencies are know when we start the task
    return deferred_task_builder{*this, offset, ::std::forward<Pack>(pack)...};
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);

    // For virtual contexts, we need to use the real context for execution
    return get_ctx(get_head_real_offset()).parallel_for(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);

    // For virtual contexts, we need to use the real context for execution
    return get_ctx(get_head_real_offset()).cuda_kernel(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel_chain(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);

    // For virtual contexts, we need to use the real context for execution
    return get_ctx(get_head_real_offset()).cuda_kernel_chain(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }
#endif

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);

    // For virtual contexts, we need to use the real context for execution
    return get_ctx(get_head_real_offset()).host_launch(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  auto fence()
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).fence();
  }

  template <typename T>
  auto wait(::cuda::experimental::stf::stackable_logical_data<T>& ldata)
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).wait(ldata.get_ld(exec_offset));
  }

  auto get_dot()
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).get_dot();
  }

  template <typename... Pack>
  void push_affinity(Pack&&... pack) const
  {
    auto lock = pimpl->acquire_shared_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);

    // For virtual contexts, we need to use the real context for execution
    get_ctx(get_head_real_offset()).push_affinity(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  void pop_affinity() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    get_ctx(exec_offset).pop_affinity();
  }

  auto& current_affinity() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).current_affinity();
  }

  const exec_place& current_exec_place() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).current_exec_place();
  }

  auto& async_resources() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).async_resources();
  }

  auto dot_section(::std::string symbol) const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).dot_section(mv(symbol));
  }

  size_t task_count() const
  {
    auto lock = pimpl->acquire_shared_lock();

    int exec_offset = get_head_real_offset();
    return get_ctx(exec_offset).task_count();
  }

  void finalize()
  {
    auto lock = pimpl->acquire_shared_lock();

    _CCCL_ASSERT(pimpl->get_head_offset() == pimpl->get_root_offset(),
                 "Can only finalize if there is no pending contexts");

    get_root_ctx().finalize();
  }

  void print_logical_data_summary() const
  {
    auto lock = pimpl->acquire_shared_lock();

    pimpl->print_logical_data_summary();
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

//! Logical data type used in a stackable_ctx context type.
//!
//! It should behaves exactly like a logical_data with additional API to import
//! it across nested contexts.
template <typename T>
class stackable_logical_data
{
public:
  /// @brief Alias for `T` - matches logical_data<T> convention
  using element_type = T;

private:
  class impl
  {
    // We separate the impl and the state so that if the stackable logical data
    // gets destroyed before the stackable context gets destroyed, we can save
    // this state in the context, in a vector of type-erased retained states
    class state : public stackable_logical_data_impl_state_base
    {
    public:
      state(stackable_ctx _sctx)
          : sctx(mv(_sctx))
      {}

      // This method is called when we pop the stackable_logical_data before we
      // have called finalize() on the nested context. This destroys the
      // logical data that was created in the nested context.
      virtual void pop_before_finalize(int ctx_offset) const override
      {
        // either data node is valid at the same offset (if the logical data
        // wasn't destroyed), or its parent must be valid.
        // int parent_offset = node_tree.parent[ctx_offset];
        int parent_offset = sctx.get_parent_offset(ctx_offset);
        _CCCL_ASSERT(parent_offset != -1, "");

        // check the parent data is valid
        _CCCL_ASSERT(data_nodes[parent_offset].has_value(), "");

        auto& parent_dnode = data_nodes[parent_offset].value();

        // Maybe the logical data was already destroyed if the stackable
        // logical data was destroyed before ctx pop, and that the data state
        // was retained. In this case, the data_node object was already
        // cleared and there is no need to do it here.
        if (data_nodes[ctx_offset].has_value())
        {
          access_mode frozen_mode = get_frozen_mode(parent_offset);
          if ((frozen_mode == access_mode::rw) && (data_nodes[ctx_offset].value().effective_mode == access_mode::read))
          {
            fprintf(stderr,
                    "Warning : no write access on data pushed with a write mode (may be suboptimal) (symbol %s)\n",
                    symbol.empty() ? "(no symbol)" : symbol.c_str());
          }

          _CCCL_ASSERT(!data_nodes[ctx_offset].value().frozen_ld.has_value(), "internal error");
          data_nodes[ctx_offset].reset();
        }

        // For virtual contexts, we skipped the get_cnt++ increment in push(),
        // so we should also skip the get_cnt-- decrement here to maintain balance
        if (sctx.is_virtual_context(ctx_offset))
        {
          return;
        }

        // Unfreezing data will create a dependency which we need to track to
        // display a dependency between the context and its parent in DOT.
        // We do this operation before finalizing the context.
        _CCCL_ASSERT(parent_dnode.frozen_ld.has_value(), "internal error");
        parent_dnode.get_cnt--;

        sctx.get_node(ctx_offset)
          ->ctx.get_dot()
          ->ctx_add_output_id(parent_dnode.frozen_ld.value().unfreeze_fake_task_id());
      }

      // Unfreeze the logical data after the context has been finalized.
      virtual void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) const override
      {
        nvtx_range r("stackable_logical_data::pop_after_finalize");

        _CCCL_ASSERT(data_nodes[parent_offset].has_value(), "");
        auto& dnode = data_nodes[parent_offset].value();

        _CCCL_ASSERT(dnode.frozen_ld.has_value(), "internal error");

        dnode.unfreeze_prereqs.merge(finalize_prereqs);

        // Only unfreeze if there are no other subcontext still using it
        _CCCL_ASSERT(dnode.get_cnt >= 0, "get_cnt should never be negative");
        if (dnode.get_cnt == 0)
        {
          dnode.frozen_ld.value().unfreeze(dnode.unfreeze_prereqs);
          dnode.frozen_ld.reset();
        }
      }

      int get_unique_id() const
      {
        _CCCL_ASSERT(data_root_offset != -1, "");
        _CCCL_ASSERT(data_nodes[data_root_offset].has_value(), "");

        // Get the ID of the base logical data
        return get_data_node(data_root_offset).ld.get_unique_id();
      }

      bool is_read_only() const
      {
        return read_only;
      }

      class data_node
      {
      public:
        data_node(logical_data<T> ld)
            : ld(mv(ld))
        {}

        ~data_node()
        {
          if (!frozen_ld.has_value())
          {
            return;
          }
        }

        // Get the access mode used to freeze data
        access_mode get_frozen_mode() const
        {
          _CCCL_ASSERT(frozen_ld.has_value(), "cannot query frozen mode : not frozen");
          return frozen_ld.value().get_access_mode();
        }

        void set_symbol(const ::std::string& symbol)
        {
          ld.set_symbol(symbol);
        }

        logical_data<T> ld;

        // Frozen counterpart of ld (if any)
        ::std::optional<frozen_logical_data<T>> frozen_ld;

        event_list unfreeze_prereqs;

        // Once frozen, count number of calls to get
        mutable int get_cnt;

        // Keep track of actual data accesses, so that we can detect if we
        // eventually did not need to freeze a data in write mode, for example.
        access_mode effective_mode = access_mode::none;
      };

      auto& get_data_node(int offset)
      {
        _CCCL_ASSERT(offset != -1, "invalid value");
        _CCCL_ASSERT(data_nodes[offset].has_value(), "invalid value");
        return data_nodes[offset].value();
      }

      const auto& get_data_node(int offset) const
      {
        _CCCL_ASSERT(offset != -1, "invalid value");
        _CCCL_ASSERT(data_nodes[offset].has_value(), "invalid value");
        return data_nodes[offset].value();
      }

      template <typename Func>
      void traverse_data_nodes(Func&& func) const
      {
        ::std::stack<int> node_stack;
        node_stack.push(data_root_offset);

        while (!node_stack.empty())
        {
          int offset = node_stack.top();
          node_stack.pop();

          // Call the provided function on the current node
          func(offset);

          // Push children to stack (reverse order to maintain left-to-right order)
          const auto& children = sctx.get_children_offsets(offset);
          for (auto it = children.rbegin(); it != children.rend(); ++it)
          {
            if (was_imported(*it))
            {
              node_stack.push(*it);
            }
          }
        }
      }

      void set_symbol(::std::string symbol_)
      {
        auto ctx_lock = sctx.acquire_exclusive_lock();
        symbol        = mv(symbol_);
        traverse_data_nodes([this](int offset) {
          get_data_node(offset).ld.set_symbol(this->symbol);
        });
      }

      int get_data_root_offset() const
      {
        return data_root_offset;
      }

      bool was_imported(int offset) const
      {
        _CCCL_ASSERT(offset != -1, "");

        if (offset >= int(data_nodes.size()))
        {
          return false;
        }

        return data_nodes[offset].has_value();
      }

      // Mark how a construct accessed this data node, so that we may detect if
      // we were overly cautious when freezing data in RW mode. This would
      // prevent concurrent accesses from different contexts, and may require
      // to push data in read only, if appropriate.
      void mark_access(int offset, access_mode m)
      {
        // For virtual contexts, traverse up to find the real context
        int actual_offset = offset;
        while (sctx.is_virtual_context(actual_offset))
        {
          actual_offset = sctx.get_parent_offset(actual_offset);
        }

        _CCCL_ASSERT(actual_offset != -1 && data_nodes[actual_offset].has_value(),
                     "Failed to find data node for mark_access");
        data_nodes[actual_offset].value().effective_mode |= m;
      }

      bool is_frozen(int offset) const
      {
        _CCCL_ASSERT(data_nodes[offset].has_value(), "");
        return data_nodes[offset].value().frozen_ld.has_value();
      }

      access_mode get_frozen_mode(int offset) const
      {
        _CCCL_ASSERT(is_frozen(offset), "");
        return data_nodes[offset].value().frozen_ld.value().get_access_mode();
      }

      ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
      {
        return ::std::shared_lock<::std::shared_mutex>(mutex);
      }

      ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
      {
        return ::std::unique_lock<::std::shared_mutex>(mutex);
      }

      friend impl;

    private:
      // Centralized method to grow data nodes to a certain size
      void grow_data_nodes(int target_size, size_t factor_numerator = 3, size_t factor_denominator = 2)
      {
        if (target_size < int(data_nodes.size()))
        {
          return; // Already large enough
        }

        size_t new_size =
          ::std::max(static_cast<size_t>(target_size), data_nodes.size() * factor_numerator / factor_denominator);
        data_nodes.resize(new_size);
      }

      mutable stackable_ctx sctx;

      mutable ::std::vector<::std::optional<data_node>> data_nodes;

      // If the logical data was created at a level that is not directly the
      // root of the context, we remember this offset
      // size_t offset_depth = 0;
      int data_root_offset;

      ::std::string symbol;

      // Indicate whether it is allowed to access this logical data with
      // write() or rw() access
      bool read_only = false;

      // We can call the stackable_logical_data destructor before popping the
      // context. In this case, the state must be retained to unfreeze when
      // appropriate.
      bool was_destroyed = false;

      mutable ::std::shared_mutex mutex;
    };

  public:
    impl() = default;
    impl(stackable_ctx sctx_,
         int target_offset,
         bool ld_from_shape,
         logical_data<T> ld,
         bool can_export,
         data_place where = data_place::invalid())
        : sctx(mv(sctx_))
    {
      impl_state = ::std::make_shared<state>(sctx);

      // TODO pass this offset directly rather than a boolean for more flexibility ? (e.g. creating a ctx of depth 2,
      // export at depth 1, not 0 ...)
      int data_root_offset         = can_export ? sctx.get_root_offset() : target_offset;
      impl_state->data_root_offset = data_root_offset;

      // Save the logical data at the base level
      if (data_root_offset >= int(impl_state->data_nodes.size()))
      {
        impl_state->grow_data_nodes(data_root_offset + 1);
      }
      _CCCL_ASSERT(!impl_state->data_nodes[data_root_offset].has_value(), "");

      impl_state->data_nodes[data_root_offset].emplace(ld);

      // fprintf(stderr, "Creating ld with ctx offset %d and root offset %d\n", target_offset, data_root_offset);

      // If necessary, import data recursively until we reach the target depth.
      // We first find the path from the target to the root and we push along this path
      if (target_offset != data_root_offset)
      {
        // Recurse from the target offset to the root offset
        ::std::stack<int> path;
        int current = target_offset;
        while (current != data_root_offset)
        {
          path.push(current);

          current = sctx.get_parent_offset(current);
          _CCCL_ASSERT(current != -1, "");
        }

        // push along the path
        while (!path.empty())
        {
          int offset = path.top();
          push(offset, ld_from_shape ? access_mode::write : access_mode::rw, where);

          path.pop();
        }
      }
    }

    // stackable_logical_data::impl::~impl
    ~impl()
    {
      // Nothing to clean up if moved or default-constructed
      if (!impl_state)
      {
        return;
      }

      auto ctx_lock = sctx.acquire_exclusive_lock(); // to protect retained_data

      int data_root_offset = impl_state->get_data_root_offset();

      _CCCL_ASSERT(impl_state->data_nodes[data_root_offset].has_value(), "");

      impl_state->was_destroyed = true;

      // TODO: Implement early cleanup of leaf nodes for better resource management
      // For now, we remove the last node but should traverse all leaves
      impl_state->data_nodes.pop_back();

      // Ensure we don't destroy the state too early by retaining its state
      // (with a shared_ptr) in all children of the data_root_offset if they
      // are valid
      // We do not retain it in the data_root_offset because  is not frozen
      // in this context.
      const auto& root_children = sctx.get_children_offsets(data_root_offset);
      for (auto c : root_children)
      {
        if (c < int(impl_state->data_nodes.size()) && impl_state->data_nodes[c].has_value())
        {
          // Transfer shared_ptr ownership to child context's retained_data vector.
          // The child context will keep impl_state alive until it's popped.
          sctx.get_node(c)->retain_data(impl_state);
        }
      }

      impl_state = nullptr;
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Non movable
    impl(impl&&) noexcept            = delete;
    impl& operator=(impl&&) noexcept = delete;

    ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
    {
      return impl_state->acquire_shared_lock();
    }

    ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
    {
      return impl_state->acquire_exclusive_lock();
    }

    const auto& get_ld(int offset) const
    {
      // For virtual contexts, traverse up to find the context with imported data
      int data_offset = resolve_to_data_context(offset);

      _CCCL_ASSERT(data_offset != -1 && impl_state->was_imported(data_offset),
                   "Failed to find imported data for virtual context");
      return impl_state->get_data_node(data_offset).ld;
    }

    auto& get_ld(int offset)
    {
      // For virtual contexts, traverse up to find the context with imported data
      int data_offset = resolve_to_data_context(offset);

      _CCCL_ASSERT(data_offset != -1 && impl_state->was_imported(data_offset),
                   "Failed to find imported data for virtual context");
      return impl_state->get_data_node(data_offset).ld;
    }

    int get_data_root_offset() const
    {
      return impl_state->get_data_root_offset();
    }

    int get_unique_id() const
    {
      return impl_state->get_unique_id();
    }

    // Helper method to traverse virtual contexts and find the context with imported data
    int resolve_to_data_context(int start_offset) const
    {
      int current_offset = start_offset;
      while (current_offset != -1 && !impl_state->was_imported(current_offset))
      {
        if (sctx.is_virtual_context(current_offset))
        {
          current_offset = sctx.get_parent_offset(current_offset);
        }
        else
        {
          break; // Non-virtual context without data - this shouldn't happen
        }
      }
      return current_offset;
    }

    // Helper method to traverse virtual contexts and find the real execution context
    int resolve_to_real_context(int start_offset) const
    {
      int current_offset = start_offset;
      while (sctx.is_virtual_context(current_offset))
      {
        current_offset = sctx.get_parent_offset(current_offset);
      }
      return current_offset;
    }

    /* Import data into the ctx at this offset */
    void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
    {
      int parent_offset = sctx.get_parent_offset(ctx_offset);

      // Base case: if this is root context (no parent), data should already exist
      if (parent_offset == -1)
      {
        _CCCL_ASSERT(ctx_offset < int(impl_state->data_nodes.size()) && impl_state->data_nodes[ctx_offset].has_value(),
                     "Root context must already have data");
        return;
      }

      if (ctx_offset >= int(impl_state->data_nodes.size()))
      {
        impl_state->grow_data_nodes(ctx_offset + 1);
      }

      if (impl_state->data_nodes[ctx_offset].has_value())
      {
        // Data already exists - ensure existing mode is compatible, no upgrades possible
        auto& existing_node = impl_state->data_nodes[ctx_offset].value();
        _CCCL_ASSERT(access_mode_is_compatible(existing_node.effective_mode, m), "Cannot change existing access mode");
        return;
      }

      // Ancestor compatibility is now handled by recursive push calls

      // Check if parent has data, if not push with max required mode
      access_mode max_required_parent_mode =
        (m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw;

      if (!impl_state->data_nodes[parent_offset].has_value())
      {
        // RECURSIVE CALL: Ensure parent has data first
        push(parent_offset, max_required_parent_mode, where);
      }

      _CCCL_ASSERT(impl_state->data_nodes[parent_offset].has_value(), "parent data should be available here");

      auto& to_node   = sctx.get_node(ctx_offset);
      auto& from_node = sctx.get_node(parent_offset);

      context& to_ctx   = to_node->ctx;
      context& from_ctx = from_node->ctx;

      auto& from_data_node = impl_state->data_nodes[parent_offset].value();

      if (where.is_invalid())
      {
        // use the default place
        where = from_ctx.default_exec_place().affine_data_place();
      }

      _CCCL_ASSERT(!where.is_invalid(), "Invalid data place");

      // Freeze the logical data of the parent node if it wasn't yet
      if (!from_data_node.frozen_ld.has_value())
      {
        from_data_node.frozen_ld = from_ctx.freeze(from_data_node.ld, m, where, false /* not a user freeze */);
        from_data_node.get_cnt   = 0;
      }
      else
      {
        // Data is already frozen - this is an IMPLICIT push
        // For implicit pushes, use conservative mode: write/rw unless specifically read-only
        access_mode existing_frozen_mode = from_data_node.frozen_ld.value().get_access_mode();

        // Check if we need to upgrade the frozen mode for implicit push
        if (!access_mode_is_compatible(existing_frozen_mode, m))
        {
          fprintf(stderr,
                  "Error: Incompatible access mode - existing frozen mode %s conflicts with requested mode %s\n",
                  access_mode_string(existing_frozen_mode),
                  access_mode_string(m));
          abort();
        }
      }

      _CCCL_ASSERT(from_data_node.frozen_ld.has_value(), "");
      auto& frozen_ld = from_data_node.frozen_ld.value();

      // FAKE IMPORT : use the stream needed to support the (graph) ctx
      cudaStream_t stream = to_node->support_stream;

      // Ensure there is a copy of the data in the data place, we keep a
      // reference count of each context using this frozen data so that we only
      // unfreeze once possible.
      ::std::pair<T, event_list> get_res = frozen_ld.get(where);
      auto ld                            = to_ctx.logical_data(get_res.first, where);
      from_data_node.get_cnt++;

      to_node->ctx_prereqs.merge(mv(get_res.second));

      if (!impl_state->symbol.empty())
      {
        ld.set_symbol(impl_state->symbol);
      }

      // The inner context depends on the freeze operation, so we ensure DOT
      // displays these dependencies from the freeze in the parent context to
      // the child context itself.
      to_ctx.get_dot()->ctx_add_input_id(frozen_ld.freeze_fake_task_id());

      // Keep track of data that were pushed in this context.  This will be
      // used to pop data automatically when nested contexts are popped.
      to_node->track_pushed_data(impl_state);

      // Create the node at the requested offset based on the logical data we
      // have just created from the data frozen in its parent.
      impl_state->data_nodes[ctx_offset].emplace(mv(ld));
    }

    /* Pop one level down */
    void pop_before_finalize(int ctx_offset) const
    {
      impl_state->pop_before_finalize(ctx_offset);
    }

    void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) const
    {
      impl_state->pop_after_finalize(parent_offset, finalize_prereqs);
    }

    void set_symbol(::std::string symbol)
    {
      impl_state->set_symbol(mv(symbol));
    }

    auto get_symbol() const
    {
      return impl_state->symbol;
    }

    // The write-back mechanism here refers to the write-back of the data at the bottom of the stack (user visible)
    void set_write_back(bool flag)
    {
      _CCCL_ASSERT(!impl_state->data_nodes.empty(), "invalid value");
      impl_state->data_nodes[0].value().ld.set_write_back(flag);
    }

    // Indicate that this logical data will only be used in a read-only mode
    // now. Implicit data push will therefore be done in a read-only mode,
    // which allows concurrent read accesses from  different contexts (ie. from
    // multiple CUDA graphs)
    void set_read_only(bool flag = true)
    {
      impl_state->read_only = flag;
    }

    bool is_read_only() const
    {
      return impl_state->is_read_only();
    }

    void mark_access(int offset, access_mode m)
    {
      return impl_state->mark_access(offset, m);
    }

    bool was_imported(int offset) const
    {
      return impl_state->was_imported(offset);
    }

    bool is_frozen(int offset) const
    {
      return impl_state->is_frozen(offset);
    }

    // Get the access mode used to freeze at a given offset
    access_mode get_frozen_mode(int offset) const
    {
      return impl_state->get_frozen_mode(offset);
    }

    int get_ctx_head_offset() const
    {
      return sctx.get_head_offset();
    }

  private:
    template <typename, typename, bool>
    friend class stackable_task_dep;

    // Note: mutable required for validate_access() calls from const methods
    // Consider refactoring to make validation logic const-correct
    mutable stackable_ctx sctx; // in which stackable context was this created ?

    ::std::shared_ptr<state> impl_state;
  };

public:
  stackable_logical_data() = default;

  /* Create a logical data in the stackable ctx : in order to make it possible
   * to export all the way down to the root context, we create the logical data
   * in the root, and import them. */
  template <typename... Args>
  stackable_logical_data(stackable_ctx sctx, int ctx_offset, bool ld_from_shape, logical_data<T> ld, bool can_export)
      : pimpl(::std::make_shared<impl>(sctx, ctx_offset, ld_from_shape, mv(ld), can_export))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>, "");
    static_assert(::std::is_move_assignable_v<stackable_logical_data>, "");
  }

  int get_data_root_offset() const
  {
    return pimpl->get_data_root_offset();
  }

  const auto& get_ld(int offset) const
  {
    return pimpl->get_ld(offset);
  }

  auto& get_ld(int offset)
  {
    return pimpl->get_ld(offset);
  }

  int get_unique_id() const
  {
    return pimpl->get_unique_id();
  }

  void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
  {
    pimpl->push(ctx_offset, m, mv(where));
  }

  void push(access_mode m, data_place where = data_place::invalid()) const
  {
    int ctx_offset = pimpl->get_ctx_head_offset();
    pimpl->push(ctx_offset, m, mv(where));
  }

  // Helpers
  template <typename... Pack>
  auto read(Pack&&... pack) const
  {
    using U = rw_type_of<T>;
    return stackable_task_dep<U, ::std::monostate, false>(
      *this, get_ld(get_data_root_offset()).read(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto write(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld(get_data_root_offset()).write(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto rw(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld(get_data_root_offset()).rw(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto reduce(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld(get_data_root_offset()).reduce(::std::forward<Pack>(pack)...));
  }

  // Helper to create dependency with specific access mode - avoids cascade of if-else
  template <typename... Pack>
  auto get_dep_with_mode(access_mode mode, Pack&&... pack)
  {
    switch (mode)
    {
      case access_mode::read:
        return read(::std::forward<Pack>(pack)...);
      case access_mode::write:
        return write(::std::forward<Pack>(pack)...);
      default: // access_mode::rw or combined modes
        return rw(::std::forward<Pack>(pack)...);
    }
  }

  auto shape() const
  {
    return get_ld(get_data_root_offset()).shape();
  }

  auto& set_symbol(::std::string symbol)
  {
    pimpl->set_symbol(mv(symbol));
    return *this;
  }

  void set_write_back(bool flag)
  {
    pimpl->set_write_back(flag);
  }

  void set_read_only(bool flag = true)
  {
    pimpl->set_read_only(flag);
  }

  bool is_read_only() const
  {
    return pimpl->is_read_only();
  }

  auto get_symbol() const
  {
    return pimpl->get_symbol();
  }

  auto get_impl()
  {
    return pimpl;
  }

  auto get_impl() const
  {
    return pimpl;
  }

  // Test whether it is valid to access this stackable_logical_data with a
  // given access mode, and automatically push data at the proper context depth
  // if necessary.
  //
  // Returns true if the task_dep needs an update
  bool validate_access(int ctx_offset, const stackable_ctx& sctx, access_mode m) const
  {
    // Grab the lock of the data, note that we are already holding the context lock in read mode
    auto lock = pimpl->acquire_exclusive_lock();

    _CCCL_ASSERT(m != access_mode::none && m != access_mode::relaxed, "Unsupported access mode in nested context");

    _CCCL_ASSERT(!is_read_only() || m == access_mode::read, "read only data cannot be modified");

    if (get_data_root_offset() == ctx_offset)
    {
      return false;
    }

    // If the stackable logical data is already available at the appropriate depth, we
    // simply need to ensure we don't make an illegal access (eg. writing a
    // read only variable)
    if (pimpl->was_imported(ctx_offset))
    {
      // Always validate access modes - find the actual parent with frozen data
      int parent_offset = sctx.get_parent_offset(ctx_offset);

      // For virtual contexts, traverse up to find the context with actual frozen data
      while (parent_offset != -1 && sctx.is_virtual_context(parent_offset))
      {
        parent_offset = sctx.get_parent_offset(parent_offset);
      }

      if (parent_offset != -1 && pimpl->is_frozen(parent_offset))
      {
        access_mode parent_frozen_mode = pimpl->get_frozen_mode(parent_offset);
        fprintf(
          stderr,
          "DEBUG: validate_access - parent frozen mode: %s, requesting mode: %s, ctx_offset: %d, parent_offset: %d\n",
          access_mode_string(parent_frozen_mode),
          access_mode_string(m),
          ctx_offset,
          parent_offset);
        if (!access_mode_is_compatible(parent_frozen_mode, m))
        {
          fprintf(stderr,
                  "Error: Invalid access mode transition - parent frozen with %s, requesting %s\n",
                  access_mode_string(parent_frozen_mode),
                  access_mode_string(m));
          abort();
        }
      }

      // To potentially detect if we were overly cautious when importing data
      // in rw mode, we record how we access it in this construct
      pimpl->mark_access(ctx_offset, m);

      // We need to update because the current ctx offset was not the base offset
      return true;
    }

    // If we reach this point, this means we need to automatically push data

    // The access mode will be very conservative for these implicit accesses
    access_mode push_mode =
      is_read_only() ? access_mode::read
                     : ((m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw);

    // Recurse from the target offset to its first imported(pushed) parent
    ::std::stack<int> path;
    int current = ctx_offset;
    while (!pimpl->was_imported(current))
    {
      path.push(current);

      current = sctx.get_parent_offset(current);
      _CCCL_ASSERT(current != -1, "");
    }

    // use the affine data place for the current default place
    // For virtual contexts, use the parent's execution place
    int exec_ctx_offset = pimpl->resolve_to_real_context(ctx_offset);
    auto where          = sctx.get_ctx(exec_ctx_offset).default_exec_place().affine_data_place();

    // push along the path
    while (!path.empty())
    {
      int offset = path.top();
      pimpl->push(offset, push_mode, where);
      path.pop();
    }

    // To potentially detect if we were overly cautious when importing data
    // in rw mode, we record how we access it in this construct
    pimpl->mark_access(ctx_offset, m);

    return true;
  }

private:
  ::std::shared_ptr<impl> pimpl;
};

inline stackable_logical_data<void_interface> stackable_ctx::token()
{
  int head = pimpl->get_head_offset();
  return stackable_logical_data<void_interface>(*this, head, true, get_root_ctx().token(), true);
}

//! Task dependency for a stackable logical data
template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep
{
public:
  // STF-compatible typedefs (required by parallel_for_scope and other STF templates)
  using data_t      = T;
  using dep_type    = T;
  using op_and_init = ::std::pair<reduce_op, ::std::bool_constant<initialize>>;
  using op_type     = reduce_op;
  enum : bool
  {
    does_work = !::std::is_same_v<reduce_op, ::std::monostate>
  };

  stackable_task_dep(stackable_logical_data<T> _d, task_dep<T, reduce_op, initialize> _dep)
      : d(mv(_d))
      , dep(mv(_dep))
  {}

  // Implicit conversion to task_dep
  operator task_dep<T, reduce_op, initialize>&()
  {
    auto& sctx = d.get_impl()->sctx;
    int offset = sctx.get_head_offset();
    d.validate_access(offset, sctx, get_access_mode());
    return dep;
  }

  // Implicit conversion to task_dep
  operator const task_dep<T, reduce_op, initialize>&() const
  {
    auto& sctx = d.get_impl()->sctx;
    int offset = sctx.get_head_offset();
    d.validate_access(offset, sctx, get_access_mode());
    return dep;
  }

  const stackable_logical_data<T>& get_d() const
  {
    return d;
  }

  // Convert to task_dep using explicit context offset (for deferred processing)
  task_dep<T, reduce_op, initialize>& to_task_dep_with_offset(int context_offset)
  {
    auto& sctx = d.get_impl()->sctx;
    d.validate_access(context_offset, sctx, get_access_mode());

    // Create new task_dep using the logical_data at the specified context offset
    // to ensure correct access to pushed data in nested contexts
    auto& context_ld = d.get_ld(context_offset);

    switch (get_access_mode())
    {
      case access_mode::read:
        dep = context_ld.read();
        break;
      case access_mode::write:
        dep = context_ld.write();
        break;
      default: // access_mode::rw or combined modes
        dep = context_ld.rw();
        break;
    }

    return dep;
  }

  // Const version for use in const contexts (like concretize_deferred_task)
  const task_dep<T, reduce_op, initialize>& to_task_dep_with_offset(int context_offset) const
  {
    auto& sctx = d.get_impl()->sctx;
    d.validate_access(context_offset, sctx, get_access_mode());

    // Create new task_dep using the logical_data at the specified context offset
    // to ensure correct access to pushed data in nested contexts
    // Note: Need non-const access to create task dependencies, even from const method
    auto& context_ld = const_cast<stackable_logical_data<T>&>(d).get_ld(context_offset);

    switch (get_access_mode())
    {
      case access_mode::read:
        dep = context_ld.read();
        break;
      case access_mode::write:
        dep = context_ld.write();
        break;
      default: // access_mode::rw or combined modes
        dep = context_ld.rw();
        break;
    }

    return dep;
  }

  // Provide non-const and const accessors.
  auto& underlying_dep()
  {
    // `*this` is a stackable_task_dep. Upcast to the base subobject.
    return dep;
  }

  const auto& underlying_dep() const
  {
    return dep;
  }

  access_mode get_access_mode() const
  {
    return dep.get_access_mode();
  }

private:
  stackable_logical_data<T> d;
  data_place dplace;
  mutable task_dep<T, reduce_op, initialize> dep;
};

#ifdef UNITTESTED_FILE
#  ifdef __CUDACC__
namespace reserved
{

template <typename T>
static __global__ void kernel_set(T* addr, T val)
{
  printf("SETTING ADDR %p at %d\n", addr, val);
  *addr = val;
}

template <typename T>
static __global__ void kernel_add(T* addr, T val)
{
  *addr += val;
}

template <typename T>
static __global__ void kernel_check_value(T* addr, T val)
{
  printf("CHECK %d EXPECTED %d\n", *addr, val);
  if (*addr != val)
  {
    ::cuda::std::terminate();
  }
}

} // namespace reserved

UNITTEST("stackable fence")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.fence();
  ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.pop();
  ctx.finalize();
};

UNITTEST("stackable host_launch")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.host_launch(lA.read())->*[](auto a) {
    _CCCL_ASSERT(a(0) == 42, "invalid value");
  };
  ctx.pop();
  ctx.finalize();
};

UNITTEST("graph_scope basic RAII")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test basic RAII behavior - scope automatically calls push/pop
  {
    auto scope = ctx.graph_scope(); // push() called here
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
    };
    // pop() called automatically when scope goes out of scope
  }

  ctx.finalize();
};

UNITTEST("graph_scope direct constructor style")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test direct constructor style (like std::lock_guard)
  {
    stackable_ctx::graph_scope_guard scope{ctx}; // Direct constructor, push() called here
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 24);
    };
    // pop() called automatically when scope goes out of scope
  }

  ctx.finalize();
};

UNITTEST("graph_scope nested scopes")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  auto lB = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test nested scopes work correctly using direct constructor style
  {
    stackable_ctx::graph_scope_guard outer_scope{ctx}; // outer push()
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 10);
    };

    {
      stackable_ctx::graph_scope_guard inner_scope{ctx}; // inner push() (nested)
      lB.push(access_mode::write, data_place::current_device());
      ctx.task(lB.write())->*[](cudaStream_t stream, auto b) {
        reserved::kernel_set<<<1, 1, 0, stream>>>(b.data_handle(), 20);
      };
      // inner pop() called automatically here
    }

    // Verify outer scope still works after inner scope closed
    ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 10);
    };
    // outer pop() called automatically here
  }

  ctx.finalize();
};

UNITTEST("graph_scope multiple sequential scopes")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test multiple sequential scopes
  {
    auto scope1 = ctx.graph_scope();
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 100);
    };
  } // pop() for scope1

  {
    auto scope2 = ctx.graph_scope();
    ctx.task(lA.rw())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_add<<<1, 1, 0, stream>>>(a.data_handle(), 23);
    };
  } // pop() for scope2

  {
    auto scope3 = ctx.graph_scope();
    ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 123);
    };
  } // pop() for scope3

  ctx.finalize();
};

inline void test_graph_scope_with_tmp()
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  {
    auto scope = ctx.graph_scope();

    // Create temporary data in nested context
    auto temp = ctx.logical_data(shape_of<slice<int>>(1024));

    ctx.parallel_for(lA.shape(), temp.write(), lA.read())->*[](size_t i, auto temp, auto a) {
      // Copy data and modify
      temp(i) = a(i) * 2;
    };

    ctx.parallel_for(lA.shape(), lA.write(), temp.read())->*[](size_t i, auto a, auto temp) {
      // Copy back
      a(i) = temp(i) + 1;
    };

    // temp automatically cleaned up when scope ends
  }

  ctx.finalize();
}

UNITTEST("graph_scope with temporary data")
{
  test_graph_scope_with_tmp();
};

inline void test_graph_scope()
{
  stackable_ctx ctx;

  // Initialize array similar to stackable2.cu
  int array[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i] = 1 + i * i;
  }

  auto lA = ctx.logical_data(array).set_symbol("A");

  // Test iterative pattern: {tmp = a, a++; tmp*=2; a+=tmp} using graph_scope RAII
  for (size_t iter = 0; iter < 3; iter++) // Use fewer iterations for faster testing
  {
    auto graph = ctx.graph_scope(); // RAII: automatic push/pop

    auto tmp = ctx.logical_data(lA.shape()).set_symbol("tmp");

    ctx.parallel_for(tmp.shape(), tmp.write(), lA.read())->*[] __device__(size_t i, auto tmp, auto a) {
      tmp(i) = a(i);
    };

    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };

    ctx.parallel_for(tmp.shape(), tmp.rw())->*[] __device__(size_t i, auto tmp) {
      tmp(i) *= 2;
    };

    ctx.parallel_for(lA.shape(), tmp.read(), lA.rw())->*[] __device__(size_t i, auto tmp, auto a) {
      a(i) += tmp(i);
    };

    // ctx.pop() is called automatically when 'graph' goes out of scope
  }

  ctx.finalize();
}

UNITTEST("graph_scope iterative pattern")
{
  test_graph_scope();
};

#  endif // __CUDACC__
#endif // UNITTESTED_FILE

#if _CCCL_CTK_AT_LEAST(12, 4)
//! \brief RAII guard for repeat loops with automatic counter management
//!
//! This class provides RAII semantics for repeat loops, automatically managing
//! the counter and conditional logic. The loop body is executed in the scope
//! between construction and destruction.
//!
//! It encapsulates the common pattern of creating a counter-based while loop
//! in CUDA STF and automatically handles:
//! - Creating and initializing a loop counter
//! - Setting up the while graph scope
//! - Decrementing the counter and controlling the loop continuation
//!
//! Example usage:
//! ```cpp
//! stackable_ctx ctx;
//! auto data = ctx.logical_data(...);
//!
//! {
//!   auto guard = ctx.repeat_graph_scope(10);
//!   // Tasks added here will run 10 times
//!   ctx.parallel_for(data.shape(), data.rw())->*[] __device__(size_t i, auto d) {
//!     d(i) += 1.0;
//!   };
//! } // Automatic scope cleanup
//! ```
class repeat_graph_scope_guard
{
public:
  template <typename CounterType>
  static void init_counter_value(stackable_ctx& ctx, CounterType counter, size_t count)
  {
    ctx.parallel_for(box(1), counter.write())->*[count] __device__(size_t, auto counter) {
      *counter = count;
    };
  }

  template <typename CounterType>
  static void setup_condition_update(stackable_ctx::while_graph_scope_guard& while_guard, CounterType counter)
  {
    while_guard.update_cond(counter.read())->*[] __device__(auto counter) {
      (*counter)--;
      return (*counter > 0);
    };
  }

  explicit repeat_graph_scope_guard(
    stackable_ctx& ctx,
    size_t count,
    unsigned int defaultLaunchValue = 1,
    unsigned int flags              = cudaGraphCondAssignDefault)
      : ctx_(ctx)
  {
    // Create counter logical data BEFORE starting while loop context
    auto counter_shape = shape_of<scalar_view<size_t>>();
    counter_           = ctx_.logical_data(counter_shape);

    // Initialize counter to the specified count
    init_counter_value(ctx_, counter_, count);

    // only create the while guard now - this starts the while loop context
    while_guard_.emplace(ctx_, defaultLaunchValue, flags, _CUDA_VSTD::source_location::current());

    // Set up the condition update logic
    setup_condition_update(*while_guard_, counter_);
  }

  // Non-copyable, non-movable
  repeat_graph_scope_guard(const repeat_graph_scope_guard&)            = delete;
  repeat_graph_scope_guard& operator=(const repeat_graph_scope_guard&) = delete;
  repeat_graph_scope_guard(repeat_graph_scope_guard&&)                 = delete;
  repeat_graph_scope_guard& operator=(repeat_graph_scope_guard&&)      = delete;

private:
  stackable_ctx& ctx_;
  ::std::optional<stackable_ctx::while_graph_scope_guard> while_guard_;
  stackable_logical_data<scalar_view<size_t>> counter_;
};

// Implementation of repeat_graph_scope method - defined here after repeat_graph_scope_guard class is complete
inline auto stackable_ctx::repeat_graph_scope(
  size_t count, unsigned int defaultLaunchValue, unsigned int flags, const _CUDA_VSTD::source_location& loc)
{
  // Note: loc parameter is provided for API consistency but not currently used in repeat_graph_scope_guard
  (void) loc; // Suppress unused parameter warning
  return repeat_graph_scope_guard(*this, count, defaultLaunchValue, flags);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

} // end namespace cuda::experimental::stf
