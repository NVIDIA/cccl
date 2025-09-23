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
#include "cuda/experimental/__stf/stackable/conditional_nodes.cuh"
#include "cuda/experimental/__stf/stackable/stackable_ctx_impl.cuh"
#include "cuda/experimental/__stf/stackable/stackable_task_dep.cuh"
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

//! Logical data type used in a stackable_ctx context type.
//!
//! It should behaves exactly like a logical_data with additional API to import
//! it across nested contexts.
class stackable_logical_data_untyped
{
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
        template <typename T>
        data_node(logical_data<T> ld)
            : ld(mv(ld))
        {}

        data_node(logical_data_untyped ld)
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

        logical_data_untyped ld;

        // Frozen counterpart of ld (if any)
        ::std::optional<frozen_logical_data_untyped> frozen_ld;

        event_list unfreeze_prereqs;

        // Once frozen, count number of calls to get
        mutable int get_cnt;

        // Keep track of actual data accesses, so that we can detect if we
        // eventually did not need to freeze a data in write mode, for example.
        access_mode effective_mode = access_mode::none;
      };

      data_node& get_data_node(int offset)
      {
        _CCCL_ASSERT(offset != -1, "invalid value");
        _CCCL_ASSERT(data_nodes[offset].has_value(), "invalid value");
        return data_nodes[offset].value();
      }

      const data_node& get_data_node(int offset) const
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
        _CCCL_ASSERT(offset != -1 && data_nodes[offset].has_value(), "Failed to find data node for mark_access");
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

    template <typename T>
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
          push<T>(offset, ld_from_shape ? access_mode::write : access_mode::rw, where);

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
      _CCCL_ASSERT(offset != -1 && impl_state->was_imported(offset), "Failed to find imported data");
      return impl_state->get_data_node(offset).ld;
    }

    auto& get_ld(int offset)
    {
      _CCCL_ASSERT(offset != -1 && impl_state->was_imported(offset), "Failed to find imported data");
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
    template <typename T>
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
        push<T>(parent_offset, max_required_parent_mode, where);
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
      ::std::pair<T, event_list> get_res = frozen_ld.template get<T>(where);
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
  stackable_logical_data_untyped() = default;

  /* Create a logical data in the stackable ctx : in order to make it possible
   * to export all the way down to the root context, we create the logical data
   * in the root, and import them. */
  template <typename T>
  stackable_logical_data_untyped(
    stackable_ctx sctx, int ctx_offset, bool ld_from_shape, logical_data<T> ld, bool can_export)
      : pimpl(::std::make_shared<impl>(sctx, ctx_offset, ld_from_shape, mv(ld), can_export))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data_untyped>, "");
    static_assert(::std::is_move_assignable_v<stackable_logical_data_untyped>, "");
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

  template <typename T>
  void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
  {
    pimpl->template push<T>(ctx_offset, m, mv(where));
  }

  template <typename T>
  void push(access_mode m, data_place where = data_place::invalid()) const
  {
    int ctx_offset = pimpl->get_ctx_head_offset();
    pimpl->template push<T>(ctx_offset, m, mv(where));
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
  template <typename T>
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
      if (parent_offset != -1 && pimpl->is_frozen(parent_offset))
      {
        access_mode parent_frozen_mode = pimpl->get_frozen_mode(parent_offset);
        // Check access mode compatibility with parent's frozen mode
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
    auto where = sctx.get_ctx(ctx_offset).default_exec_place().affine_data_place();

    // push along the path
    while (!path.empty())
    {
      int offset = path.top();
      pimpl->push<T>(offset, push_mode, where);
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

template <typename T>
class stackable_logical_data : public stackable_logical_data_untyped
{
public:
  /// @brief Alias for `T` - matches logical_data<T> convention
  using element_type = T;

  stackable_logical_data() = default;

  stackable_logical_data(stackable_ctx sctx, int ctx_offset, bool ld_from_shape, logical_data<T> ld, bool can_export)
      : stackable_logical_data_untyped(sctx, ctx_offset, ld_from_shape, mv(ld), can_export)
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>, "");
    static_assert(::std::is_move_assignable_v<stackable_logical_data>, "");
  }

  bool validate_access(int ctx_offset, const stackable_ctx& sctx, access_mode m) const
  {
    return stackable_logical_data_untyped::template validate_access<T>(ctx_offset, sctx, m);
  }

  // Helpers
  template <typename... Pack>
  auto read(Pack&&... pack) const
  {
    using U = rw_type_of<T>;
    return stackable_task_dep<U, ::std::monostate, false>(*this, get_ld().read(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto write(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld().write(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto rw(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld().rw(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto reduce(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld().reduce(::std::forward<Pack>(pack)...));
  }

  auto shape() const
  {
    return get_ld().shape();
  }

  // Override set_symbol to return proper derived type
  stackable_logical_data<T>& set_symbol(::std::string symbol)
  {
    stackable_logical_data_untyped::set_symbol(mv(symbol));
    return *this;
  }

  // Type-safe get_ld() method that returns logical_data<T>& directly
  logical_data<T>& get_ld(int offset)
  {
    // Get the untyped logical_data from base class
    logical_data_untyped& untyped_ld = stackable_logical_data_untyped::get_ld(offset);

    // Since logical_data<T> inherits from logical_data_untyped with same size,
    // we can safely cast (verified by static_assert in logical_data.cuh)
    return static_cast<logical_data<T>&>(untyped_ld);
  }

  const logical_data<T>& get_ld(int offset) const
  {
    // Get the untyped logical_data from base class
    const logical_data_untyped& untyped_ld = stackable_logical_data_untyped::get_ld(offset);

    // Since logical_data<T> inherits from logical_data_untyped with same size,
    // we can safely cast (verified by static_assert in logical_data.cuh)
    return static_cast<const logical_data<T>&>(untyped_ld);
  }

  // Overload that uses current context offset for convenience
  logical_data<T>& get_ld()
  {
    return get_ld(get_data_root_offset());
  }

  const logical_data<T>& get_ld() const
  {
    return get_ld(get_data_root_offset());
  }
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
    d.template validate_access<T>(offset, sctx, get_access_mode());
    return dep;
  }

  // Implicit conversion to task_dep
  operator const task_dep<T, reduce_op, initialize>&() const
  {
    auto& sctx = d.get_impl()->sctx;
    int offset = sctx.get_head_offset();
    d.template validate_access<T>(offset, sctx, get_access_mode());
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
    d.template validate_access<T>(context_offset, sctx, get_access_mode());

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
    d.template validate_access<T>(context_offset, sctx, get_access_mode());

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

  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };

  {
    auto scope = ctx.graph_scope();

    // Create temporary data in nested context
    auto temp = ctx.logical_data(shape_of<slice<int>>(1024));

    ctx.parallel_for(lA.shape(), temp.write(), lA.read())->*[] __device__(size_t i, auto temp, auto a) {
      // Copy data and modify
      temp(i) = a(i) * 2;
    };

    ctx.parallel_for(lA.shape(), lA.write(), temp.read())->*[] __device__(size_t i, auto a, auto temp) {
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

UNITTEST("stackable task on exec_place::host()")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.task(exec_place::host(), lA.write())->*[](cudaStream_t stream, auto) {
    cuda_safe_call(cudaStreamSynchronize(stream));
  };
  ctx.finalize();
};

UNITTEST("stackable task with set_symbol and set_exec_place")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.task(lA.write()).set_symbol("task").set_exec_place(exec_place::host())->*[](cudaStream_t stream, auto) {
    cuda_safe_call(cudaStreamSynchronize(stream));
  };
  ctx.finalize();
};

#  endif // __CUDACC__
#endif // UNITTESTED_FILE

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
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
    unsigned int default_launch_value = 1,
    unsigned int flags                = cudaGraphCondAssignDefault)
      : ctx_(ctx)
  {
    // Create counter logical data BEFORE starting while loop context
    auto counter_shape = shape_of<scalar_view<size_t>>();
    counter_           = ctx_.logical_data(counter_shape);

    // Initialize counter to the specified count
    init_counter_value(ctx_, counter_, count);

    // only create the while guard now - this starts the while loop context
    while_guard_.emplace(ctx_, default_launch_value, flags, _CUDA_VSTD::source_location::current());

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
  size_t count, unsigned int default_launch_value, unsigned int flags, const _CUDA_VSTD::source_location& loc)
{
  // Note: loc parameter is provided for API consistency but not currently used in repeat_graph_scope_guard
  (void) loc; // Suppress unused parameter warning
  return repeat_graph_scope_guard(*this, count, default_launch_value, flags);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

} // end namespace cuda::experimental::stf
