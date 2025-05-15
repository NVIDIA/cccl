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
 * @brief Stackable context and logical data to nest contexts
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

#include <shared_mutex>
#include <stack>
#include <thread>

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/__stf/utility/hash.cuh"
#include "cuda/experimental/__stf/utility/source_location.cuh"
#include "cuda/experimental/stf.cuh"

/**
 * TODO insert a big comment explaining the design and how to reason about this !
 */

namespace cuda::experimental::stf
{

template <typename T>
class stackable_logical_data;

template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep;

namespace reserved
{

template <typename T>
struct is_stackable_task_dep : ::std::false_type
{};

template <typename T, typename ReduceOp, bool Init>
struct is_stackable_task_dep<stackable_task_dep<T, ReduceOp, Init>> : ::std::true_type
{};

// This helper converts stackable_task_dep to the underlying task_dep. If we
// have a stackable_logical_data A, A.read() is indeed a stackable_task_dep,
// which we can pass to stream_ctx/graph_ctx constructs by extracting the
// underlying task_dep.
//
template <typename U>
decltype(auto) to_task_dep(U&& u)
{
  if constexpr (is_stackable_task_dep<::std::decay_t<U>>::value)
  {
    return ::std::forward<U>(u).underlying_dep();
  }
  else
  {
    return ::std::forward<U>(u);
  }
}

} // end namespace reserved

/**
 * @brief Base class with a virtual pop method to enable type erasure
 *
 * This is used to implement the automatic call to pop() on logical data when a
 * context node is popped.
 */
class stackable_logical_data_impl_state_base
{
public:
  virtual ~stackable_logical_data_impl_state_base()                                            = default;
  virtual void pop_before_finalize(int ctx_offset) const                                       = 0;
  virtual void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) const = 0;
};

/**
 * @brief This class defines a context that behaves as a context which can have nested subcontexts (implemented as local
 * CUDA graphs)
 */
class stackable_ctx
{
public:
  class impl
  {
    friend class stackable_ctx;

  private:
    /*
     * State of each nested context
     */
    struct ctx_node
    {
      ctx_node(context ctx, cudaStream_t support_stream, ::std::shared_ptr<stream_adapter> alloc_adapters)
          : ctx(mv(ctx))
          , support_stream(mv(support_stream))
          , alloc_adapters(mv(alloc_adapters))
          , refcnt(1)
      {}

      ctx_node(ctx_node&&) noexcept            = default;
      ctx_node& operator=(ctx_node&&) noexcept = default;

      // To avoid prematurely destroying data created in a nested context, we need to hold a reference to them
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

      // When we call ctx.push() in a nested way, we do not actually create
      // nested CUDA graphs, so the nested push/pop are no-ops, but we keep
      // track of the number of users of a context node to properly implement
      // the push/pop semantic
      int refcnt;

      // The async resource handle used in this context
      ::std::optional<async_resources_handle> async_handle;

    private:
      // If we want to keep the state of some logical data implementations until this node is popped
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> retained_data;
    };

    /* To describe the hierarchy of contexts, and the hierarchy of stackable
     * logical data which should match the structure of the context hierarchy, this
     * class describes a tree using a vector of offsets. Every node has an offset,
     * and we keep track of the parent offset of each node, as well as its
     * children. */
    class node_hierarchy
    {
    public:
      node_hierarchy()
      {
        // May grow up later if more contexts are needed
        int initialize_size = 16;
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
        _CCCL_ASSERT(free_list.size() > 0, "no slot available");

        int res = free_list.back();
        free_list.pop_back();

        parent[res] = -1;
        // the node should be unused
        _CCCL_ASSERT(children[res].size() == 0, "invalid state");

        return res;
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
          bool found = false;
          ::std::vector<int> new_children;
          new_children.reserve(children[p].size() - 1);
          for (auto c : children[p])
          {
            if (c == offset)
            {
              found = true;
            }
            else
            {
              new_children.push_back(c);
            }
          }

          // Ensure we did find the node in it's parent's children
          _CCCL_ASSERT(found, "invalid hierarchy state");
          ::std::swap(children[p], new_children);
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
        _CCCL_ASSERT(offset < int(parent.size()), "");
        return parent[offset];
      }

      const auto& get_children(int offset) const
      {
        _CCCL_ASSERT(offset < int(children.size()), "");
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
     * @brief Create a new nested level
     *
     * head_offset is the offset of thread's current top context (-1 if none)
     */
    void push(const _CUDA_VSTD::source_location& loc, bool is_root = false)
    {
      auto lock = get_write_lock();

      // If we are creating the root context, we do not try to get some
      // uninitialized thread-local value.
      int head_offset  = is_root ? -1 : get_head_offset();
      int parent_depth = is_root ? -1 : int(node_tree.depth(head_offset));

      if (parent_depth >= 1)
      {
        // If there is already a parent context node that is a CUDA graph,
        // the push/pop section will use the same context node because there
        // is currently no such thing as a nested CUDA graphs (child graph
        // would not help much) and this simple strategy is good enough.
        //
        // There is no need to update the current head of the thread, but we
        // increment the reference count of the context node.
        _CCCL_ASSERT(nodes[head_offset].has_value(), "invalid hierarchy");
        auto& parent_node = nodes[head_offset].value();

        // The parent node will perform all constructs in this push/pop scope.
        parent_node.refcnt++;

        return;
      }

      // Select the offset of the new node
      int node_offset = node_tree.get_avail_entry();

      if (int(nodes.size()) <= node_offset)
      {
        nodes.resize(node_offset + 1); // TODO round or resize to node_tree size ?
      }

      // Ensure the node offset was unused
      _CCCL_ASSERT(!nodes[node_offset].has_value(), "inconsistent state");

      // If there is no current context, this is the root context and we use a stream_ctx
      if (head_offset == -1)
      {
        nodes[node_offset].emplace(stream_ctx(), nullptr, nullptr);

        // root of the context
        root_offset = node_offset;
      }
      else
      {
        // In the current implementation, depth > 1 is using a no-op for push/pop
        _CCCL_ASSERT(parent_depth == 0, "invalid state");
        // Keep track of parenthood
        node_tree.set_parent(head_offset, node_offset);

        _CCCL_ASSERT(nodes[head_offset].has_value(), "invalid hierarchy");
        auto& parent_node = nodes[head_offset].value();

        // Get a stream from previous context (we haven't pushed the new one yet)
        cudaStream_t stream = parent_node.ctx.pick_stream();

        // We use an optional to either use an existing handle from the stack
        // of handles associated to that execution place, or create one if
        // there is none available, but we do not want to create a new async
        // handle if we are not going to use it.
        ::std::optional<async_resources_handle> handle;

        auto& handle_stack = async_handles[parent_node.ctx.default_exec_place()];
        if (handle_stack.empty())
        {
          handle.emplace();
        }
        else
        {
          handle = mv(handle_stack.top());
          handle_stack.pop();
        }

        auto gctx = graph_ctx(stream, handle.value());

        // Useful for tools
        gctx.set_parent_ctx(parent_node.ctx);
        gctx.get_dot()->set_ctx_symbol("graph[" + ::std::string(loc.file_name()) + ":" + ::std::to_string(loc.line())
                                       + "(" + loc.function_name() + ")]");

        auto wrapper = ::std::make_shared<stream_adapter>(gctx, stream);

        gctx.update_uncached_allocator(wrapper->allocator());

        nodes[node_offset].emplace(gctx, stream, wrapper);

        // Save the async handle to reuse it later if necessary
        nodes[node_offset]->async_handle = mv(handle);

        if (display_graph_stats)
        {
          nodes[node_offset]->callsite = loc;
        }
      }

      // Update the current context head of the thread
      set_head_offset(node_offset);
    }

    /**
     * @brief Terminate the current nested level and get back to the previous one
     */
    void pop()
    {
      auto lock       = get_write_lock();
      int head_offset = get_head_offset();

      // fprintf(stderr, "stackable_ctx::pop() depth() was %ld\n", depth());
      _CCCL_ASSERT(nodes.size() > 0, "Calling pop while no context was pushed");
      _CCCL_ASSERT(nodes[head_offset].has_value(), "invalid state");

      auto& current_node = nodes[head_offset].value();

      // If we have nested push/pop statements, the inner ctx.push() are just
      // no-ops, we only really pop the context once all push/pop scopes using
      // this ctx node are done.
      current_node.refcnt--;
      if (current_node.refcnt > 0)
      {
        return;
      }

      const char* display_mem_stats_env = getenv("CUDASTF_DISPLAY_MEM_STATS");
      if (display_mem_stats_env && atoi(display_mem_stats_env) != 0)
      {
        fprintf(stderr, "PRINT DATA SUMMARY ctx %s\n", current_node.ctx.get_dot()->get_ctx_symbol().c_str());
        // current_node.ctx.print_logical_data_summary();
      }

      // Automatically pop data if needed. This will destroy the logical data
      // created within the context that is being destroyed. Unless using "non
      // exportable data" this logical data was created as an alias of an
      // another logical data that was frozen in the parent context. The
      // unfreeze operation is delayed after finalizing the context, in the
      // pop_after_finalize stage.
      for (auto& d_impl : current_node.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_before_finalize(head_offset);
      }

      // Ensure everything is finished in the context
      current_node.ctx.finalize();

      // Make the async resource handle reusable on this execution place
      auto& handle_stack = async_handles[current_node.ctx.default_exec_place()];
      handle_stack.push(mv(current_node.async_handle.value()));

      if (display_graph_stats)
      {
        // When a graph context is finalized, a CUDA graph is created, we here
        // retrieve some information about it, and relate it to the location in
        // sources of the context push() call.
        executable_graph_cache_stat* stat = current_node.ctx.graph_get_cache_stat();
        _CCCL_ASSERT(stat, "");

        const auto& loc = current_node.callsite;
        stats_map[loc][::std::make_pair(stat->nnodes, stat->nedges)] += *stat;
      }

      // To create prereqs that depend on this finalize() stage, we get the
      // stream used in this context, and insert events in it.
      cudaStream_t stream = current_node.support_stream;

      int parent_offset = node_tree.get_parent(head_offset);
      _CCCL_ASSERT(parent_offset != -1, "internal error: no parent ctx");

      auto& parent_node = nodes[parent_offset].value();

      // Create an event that depends on the completion of previous operations in the stream
      event_list finalize_prereqs = parent_node.ctx.stream_to_event_list(stream, "finalized");

      // Now that the context has been finalized, we can unfreeze data (unless
      // another sibling context has pushed it too)
      for (auto& d_impl : current_node.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_after_finalize(parent_offset, finalize_prereqs);
      }

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_node.alloc_adapters)
      {
        current_node.alloc_adapters->clear();
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

    // Offset of the root context
    int get_root_offset() const
    {
      return root_offset;
    }

    auto& get_root_ctx()
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset].has_value(), "invalid state");
      return nodes[root_offset].value().ctx;
    }

    const auto& get_root_ctx() const
    {
      _CCCL_ASSERT(root_offset != -1, "invalid state");
      _CCCL_ASSERT(nodes[root_offset].has_value(), "invalid state");
      return nodes[root_offset].value().ctx;
    }

    ctx_node& get_node(int offset)
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset].has_value(), "invalid value");
      return nodes[offset].value();
    }

    const ctx_node& get_node(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "invalid value");
      _CCCL_ASSERT(offset < int(nodes.size()), "invalid value");
      _CCCL_ASSERT(nodes[offset].has_value(), "invalid value");
      return nodes[offset].value();
    }

    auto& get_ctx(int offset)
    {
      return get_node(offset).ctx;
    }

    const auto& get_ctx(int offset) const
    {
      return get_node(offset).ctx;
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

    int get_parent_offset(int offset) const
    {
      _CCCL_ASSERT(offset != -1, "");
      return node_tree.get_parent(offset);
    }

    const auto& get_children_offsets(int parent) const
    {
      _CCCL_ASSERT(parent != -1, "");
      return node_tree.get_children(parent);
    }

  private:
    void print_logical_data_summary() const
    {
      traverse_nodes([this](int offset) {
        auto& ctx = get_ctx(offset);
        fprintf(stderr, "[context %d (%s)] logical data summary:\n", offset, ctx.to_string().c_str());
        //  ctx.print_logical_data_summary();
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
    ::std::vector<::std::optional<ctx_node>> nodes;

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
    ::std::shared_lock<::std::shared_mutex> get_read_lock() const
    {
      return ::std::shared_lock<::std::shared_mutex>(mutex);
    }

    ::std::unique_lock<::std::shared_mutex> get_write_lock()
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

  auto& get_root_ctx()
  {
    return pimpl->get_root_ctx();
  }

  const auto& get_root_ctx() const
  {
    return pimpl->get_root_ctx();
  }

  int get_root_offset() const
  {
    return pimpl->get_root_offset();
  }

  auto& get_ctx(int offset)
  {
    return pimpl->get_ctx(offset);
  }

  const auto& get_ctx(int offset) const
  {
    return pimpl->get_ctx(offset);
  }

  int get_head_offset() const
  {
    return pimpl->get_head_offset();
  }

  void set_head_offset(int offset)
  {
    pimpl->set_head_offset(offset);
  }

  void push(const _CUDA_VSTD::source_location loc = _CUDA_VSTD::source_location::current())
  {
    pimpl->push(loc);
  }

  void pop()
  {
    pimpl->pop();
  }

  template <typename T>
  auto logical_data(shape_of<T> s)
  {
    auto lock = pimpl->get_read_lock();

    // fprintf(stderr, "initialize from shape.\n");
    int head = pimpl->get_head_offset();
    return stackable_logical_data(*this, head, true, get_root_ctx().logical_data(mv(s)), true);
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    auto lock = pimpl->get_read_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(
      *this, head, true, get_root_ctx().template logical_data<T>(elements, more_sizes...), true);
  }

  template <typename T>
  auto logical_data_no_export(shape_of<T> s)
  {
    auto lock = pimpl->get_read_lock();

    // fprintf(stderr, "initialize from shape.\n");
    int head = pimpl->get_head_offset();
    return stackable_logical_data(*this, head, true, get_ctx(head).logical_data(mv(s)), false);
  }

  template <typename T, typename... Sizes>
  auto logical_data_no_export(size_t elements, Sizes... more_sizes)
  {
    auto lock = pimpl->get_read_lock();

    int head = pimpl->get_head_offset();
    return stackable_logical_data(
      *this, head, true, get_ctx(head).template logical_data<T>(elements, more_sizes...), false);
  }

  stackable_logical_data<void_interface> token();

  template <typename... Pack>
  auto logical_data(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int head = pimpl->get_head_offset();
    // fprintf(stderr, "initialize from value.\n");
    return stackable_logical_data(*this, head, false, get_ctx(head).logical_data(::std::forward<Pack>(pack)...), true);
  }

  // Helper function to process a single argument
  template <typename T1>
  void process_argument(int, const T1&, ::std::vector<::std::pair<int, access_mode>>&) const
  {
    // Do nothing for non-stackable_task_dep
  }

  template <typename T1, typename reduce_op, bool initialize>
  void process_argument(int ctx_offset,
                        const stackable_task_dep<T1, reduce_op, initialize>& dep,
                        ::std::vector<::std::pair<int, access_mode>>& combined_accesses) const
  {
    // If the stackable logical data appears in multiple deps of the same
    // task, we need to combine access modes to push the data automatically
    // with an appropriate mode.
    int id  = dep.get_d().get_unique_id();
    auto it = ::std::find_if(
      combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
        return entry.first == id;
      });
    _CCCL_ASSERT(it != combined_accesses.end(), "internal error");
    access_mode combined_m = it->second;

    // If the logical data was not at the appropriate level, we may
    // automatically push it. In this case, we need to update the logical data
    // referred stored in the task_dep object.
    bool need_update = dep.get_d().validate_access(ctx_offset, *this, combined_m);
    //    if (need_update)
    {
      // The underlying dep eg. obtained when calling l.read() was resulting in
      // an task_dep_untyped where the logical data was the one at the "top of
      // the stack". Since a push method was done automatically, the logical
      // data that needs to be used was incorrect, and we update it.
      dep.underlying_dep().update_data(dep.get_d().get_ld(ctx_offset));
    }
  }

  template <typename T1>
  void combine_argument_access_modes(const T1&, ::std::vector<::std::pair<int, access_mode>>&) const
  {
    // nothing
  }

  template <typename T1, typename reduce_op, bool initialize>
  void combine_argument_access_modes(const stackable_task_dep<T1, reduce_op, initialize>& dep,
                                     ::std::vector<::std::pair<int, access_mode>>& combined_accesses) const
  {
    int id        = dep.get_d().get_unique_id();
    access_mode m = dep.get_access_mode();

    // Now we go through the vector and update it if necessary, or add to the vector
    auto it = ::std::find_if(
      combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
        return entry.first == id;
      });

    if (it != combined_accesses.end())
    {
      it->second = it->second | m; // Merge access modes
    }
    else
    {
      combined_accesses.emplace_back(id, m); // Insert if not found
    }
  }

  // Process each argument passed to this construct to ensure we are accessing
  // data at the proper depth, or automatically push them at the appropriate
  // depth if necessary. If this happens, we may update the task_dep objects to
  // reflect the actual logical data that needs to be used.
  template <typename... Pack>
  void process_pack([[maybe_unused]] int offset, const Pack&... pack) const
  {
    // This is a map of logical data, and the combined access modes
    ::std::vector<::std::pair<int, access_mode>> combined_accesses;
    (combine_argument_access_modes(pack, combined_accesses), ...);

    (process_argument(offset, pack, combined_accesses), ...);
  }

  template <typename... Pack>
  auto task(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    return get_ctx(offset).task(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    return get_ctx(offset).parallel_for(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    return get_ctx(offset).cuda_kernel(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel_chain(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    return get_ctx(offset).cuda_kernel_chain(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }
#endif

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    return get_ctx(offset).host_launch(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  auto task_fence()
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).task_fence();
  }

  auto get_dot()
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).get_dot();
  }

  template <typename... Pack>
  void push_affinity(Pack&&... pack) const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    process_pack(offset, pack...);
    get_ctx(offset).push_affinity(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  void pop_affinity() const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    get_ctx(offset).pop_affinity();
  }

  auto& current_affinity() const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).current_affinity();
  }

  const exec_place& current_exec_place() const
  {
    auto lock  = pimpl->get_read_lock();
    int offset = get_head_offset();
    return get_ctx(offset).current_exec_place();
  }

  auto& async_resources() const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).async_resources();
  }

  auto dot_section(::std::string symbol) const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).dot_section(mv(symbol));
  }

  size_t task_count() const
  {
    auto lock = pimpl->get_read_lock();

    int offset = get_head_offset();
    return get_ctx(offset).task_count();
  }

  void finalize()
  {
    auto lock = pimpl->get_read_lock();

    _CCCL_ASSERT(pimpl->get_head_offset() == pimpl->get_root_offset(),
                 "Can only finalize if there is no pending contexts");

    get_root_ctx().finalize();
  }

  void print_logical_data_summary() const
  {
    auto lock = pimpl->get_read_lock();

    pimpl->print_logical_data_summary();
  }

  ::std::shared_lock<::std::shared_mutex> get_read_lock() const
  {
    return pimpl->get_read_lock();
  }

  ::std::unique_lock<::std::shared_mutex> get_write_lock()
  {
    return pimpl->get_write_lock();
  }

public:
  ::std::shared_ptr<impl> pimpl;
};

template <typename T>
class stackable_logical_data
{
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

        // Unfreezing data will create a dependency which we need to track to
        // display a dependency between the context and its parent in DOT.
        // We do this operation before finalizing the context.
        _CCCL_ASSERT(parent_dnode.frozen_ld.has_value(), "internal error");
        parent_dnode.get_cnt--;

        sctx.get_node(ctx_offset)
          .ctx.get_dot()
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
        auto ctx_lock = sctx.get_write_lock();
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
        _CCCL_ASSERT(data_nodes[offset].has_value(), "");
        data_nodes[offset].value().effective_mode |= m;
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

      ::std::shared_lock<::std::shared_mutex> get_read_lock() const
      {
        return ::std::shared_lock<::std::shared_mutex>(mutex);
      }

      ::std::unique_lock<::std::shared_mutex> get_write_lock()
      {
        return ::std::unique_lock<::std::shared_mutex>(mutex);
      }

      friend impl;

    private:
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
        impl_state->data_nodes.resize(data_root_offset + 1);
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
      // Maybe we moved it for example
      if (!impl_state)
      {
        return;
      }

      auto ctx_lock = sctx.get_write_lock(); // to protect retained_data

      int data_root_offset = impl_state->get_data_root_offset();

      _CCCL_ASSERT(impl_state->data_nodes[data_root_offset].has_value(), "");

      impl_state->was_destroyed = true;

      // TODO reset all leaves to destroy ld early
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
          // Save the shared_ptr into the children contexts using the data
          sctx.get_node(c).retain_data(impl_state);
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

    ::std::shared_lock<::std::shared_mutex> get_read_lock() const
    {
      return impl_state->get_read_lock();
    }

    ::std::unique_lock<::std::shared_mutex> get_write_lock()
    {
      return impl_state->get_write_lock();
    }

    const auto& get_ld(int offset) const
    {
      return impl_state->get_data_node(offset).ld;
    }

    auto& get_ld(int offset)
    {
      return impl_state->get_data_node(offset).ld;
    }

    int get_data_root_offset() const
    {
      return impl_state->get_data_root_offset();
    }

    int get_unique_id() const
    {
      return impl_state->get_unique_id();
    }

    /* Import data into the ctx at this offset */
    void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
    {
      int parent_offset = sctx.get_parent_offset(ctx_offset);
      _CCCL_ASSERT(parent_offset != -1, "");

      if (ctx_offset >= int(impl_state->data_nodes.size()))
      {
        impl_state->data_nodes.resize(ctx_offset + 1);
      }

      if (impl_state->data_nodes[ctx_offset].has_value())
      {
        // If the logical data was already imported in this context, we just ensure the existing import was compatible
        auto& existing_node = impl_state->data_nodes[ctx_offset].value();
        _CCCL_ASSERT(access_mode_is_compatible(existing_node.effective_mode, m), "invalid access mode");
        return;
      }
      _CCCL_ASSERT(impl_state->data_nodes[parent_offset].has_value(), "parent data must have been pushed");

      auto& to_node   = sctx.get_node(ctx_offset);
      auto& from_node = sctx.get_node(parent_offset);

      context& to_ctx   = to_node.ctx;
      context& from_ctx = from_node.ctx;

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
        // TODO check that the frozen mode is compatible !
      }

      _CCCL_ASSERT(from_data_node.frozen_ld.has_value(), "");
      auto& frozen_ld = from_data_node.frozen_ld.value();

      // FAKE IMPORT : use the stream needed to support the (graph) ctx
      cudaStream_t stream = to_node.support_stream;

      // Ensure there is a copy of the data in the data place, we keep a
      // reference count of each context using this frozen data so that we only
      // unfreeze once possible.
      T inst  = frozen_ld.get(where, stream);
      auto ld = to_ctx.logical_data(inst, where);
      from_data_node.get_cnt++;

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
      to_node.track_pushed_data(impl_state);

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
      _CCCL_ASSERT(impl_state->s.size() > 0, "invalid value");
      impl_state->s[0].set_write_back(flag);
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

    // TODO replace this mutable by a const ...
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

  // Test whether it is valid to access this stackable_logical_data with a
  // given access mode, and automatically push data at the proper context depth
  // if necessary.
  //
  // Returns true if the task_dep needs an update
  bool validate_access(int ctx_offset, const stackable_ctx& sctx, access_mode m) const
  {
    // Grab the lock of the data, note that we are already holding the context lock in read mode
    auto lock = pimpl->get_write_lock();

    _CCCL_ASSERT(m == access_mode::read || m == access_mode::rw || m == access_mode::write,
                 "Unsupported access mode in nested context");

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
#ifndef NDEBUG
      // Parent must be frozen, and current offset must not be frozen (otherwise a subcontext is accessing it)
      int parent_offset = sctx.get_parent_offset(ctx_offset);
      _CCCL_ASSERT(!pimpl->is_frozen(ctx_offset), "");
      _CCCL_ASSERT(pimpl->is_frozen(parent_offset), "");
      _CCCL_ASSERT(access_mode_is_compatible(pimpl->get_frozen_mode(parent_offset), m), "Invalid access mode");
#endif

      // To potentially detect if we were overly cautious when importing data
      // in rw mode, we record how we access it in this construct
      pimpl->mark_access(ctx_offset, m);

      // We need to update because the current ctx offset was not the base offset
      return true;
    }

    // If we reach this point, this means we need to automatically push data

    // The access mode will be very conservative for these implicit accesses
    access_mode push_mode =
      is_read_only() ? access_mode::read : ((m == access_mode::write) ? access_mode::write : access_mode::rw);

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
    auto where = sctx.get_ctx(ctx_offset).default_exec_place().affine_data_place();

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

/**
 * @brief Task dependency for a stackable logical data
 */
template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep
{
public:
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

UNITTEST("stackable task_fence")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.task_fence();
  ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 44);
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
  // ctx.host_launch(lA.read())->*[](auto a){ _CCCL_ASSERT(a(0) == 42, "invalid value"); };
  ctx.pop();
  ctx.finalize();
};

#  endif // __CUDACC__
#endif // UNITTESTED_FILE

} // end namespace cuda::experimental::stf
