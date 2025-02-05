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

#include "cuda/experimental/__stf/allocators/adapters.cuh"
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
// By default, return the argument as-is (perfect forwarding)
template <typename U>
decltype(auto) to_task_dep(U&& u)
{
  return ::std::forward<U>(u);
}

// Overload for stackable_task_dep (non-const version)
template <typename T, typename reduce_op, bool initialize>
task_dep<T, reduce_op, initialize>& to_task_dep(stackable_task_dep<T, reduce_op, initialize>& sdep)
{
  return sdep.underlying_dep();
}

// Overload for stackable_task_dep (const version)
template <typename T, typename reduce_op, bool initialize>
const task_dep<T, reduce_op, initialize>& to_task_dep(const stackable_task_dep<T, reduce_op, initialize>& sdep)
{
  return sdep.underlying_dep();
}

template <typename T, typename reduce_op, bool initialize>
task_dep<T, reduce_op, initialize> to_task_dep(stackable_task_dep<T, reduce_op, initialize>&& sdep)
{
  // Return by value or whatever makes sense:
  return ::std::move(sdep.underlying_dep());
}

} // end namespace reserved

/**
 * @brief Base class with a virtual pop method to enable type erasure
 *
 * This is used to implement the automatic call to pop() on logical data when a
 * context level is popped.
 */
class stackable_logical_data_impl_state_base
{
public:
  virtual ~stackable_logical_data_impl_state_base()         = default;
  virtual void pop_before_finalize() const = 0;
  virtual void pop_after_finalize() const                   = 0;
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
  private:
    /*
     * State of each nested context
     */
    struct per_level
    {
      per_level(context ctx, cudaStream_t support_stream, ::std::shared_ptr<stream_adapter> alloc_adapters)
          : ctx(mv(ctx))
          , support_stream(mv(support_stream))
          , alloc_adapters(mv(alloc_adapters))
      {}

      context ctx;
      cudaStream_t support_stream;
      // A wrapper to forward allocations from a level to the previous one (none is used at the root level)
      ::std::shared_ptr<stream_adapter> alloc_adapters;

      // This map keeps track of the logical data that were pushed in this level
      // key: logical data's unique id
      ::std::unordered_map<int, ::std::shared_ptr<stackable_logical_data_impl_state_base>> pushed_data;

      // If we want to keep the state of some logical data implementations until this level is popped
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> retained_data;
    };

  public:
    impl()
    {
      // Create the root level
      push();
    }

    ~impl() = default;

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Define move constructor and move assignment operator
    impl(impl&&) noexcept            = default;
    impl& operator=(impl&&) noexcept = default;

    /**
     * @brief Create a new nested level
     */
    void push()
    {
      // fprintf(stderr, "stackable_ctx::push() depth() was %ld\n", depth());

      // These resources are not destroyed when we pop, so we create it only if needed
      if (async_handles.size() < levels.size())
      {
        async_handles.emplace_back();
      }

      if (levels.size() == 0)
      {
        levels.emplace_back(stream_ctx(), nullptr, nullptr);
      }
      else
      {
        // Get a stream from previous context (we haven't pushed the new one yet)
        cudaStream_t stream = levels[depth()].ctx.pick_stream();

        auto gctx = graph_ctx(stream, async_handles.back());

        // Useful for tools
        gctx.set_parent_ctx(levels[depth()].ctx);
        gctx.get_dot()->set_ctx_symbol("stacked_ctx_" + ::std::to_string(levels.size()));

        auto wrapper = ::std::make_shared<stream_adapter>(gctx, stream);

        // FIXME : issue with the deinit phase
        gctx.update_uncached_allocator(wrapper->allocator());

        levels.emplace_back(gctx, stream, wrapper);
      }
    }

    /**
     * @brief Terminate the current nested level and get back to the previous one
     */
    void pop()
    {
      // fprintf(stderr, "stackable_ctx::pop() depth() was %ld\n", depth());
      _CCCL_ASSERT(levels.size() > 0, "Calling pop while no context was pushed");

      auto& current_level = levels.back();

      // Automatically pop data if needed
      for (auto& [key, d_impl] : current_level.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_before_finalize();
      }

      // Ensure everything is finished in the context
      current_level.ctx.finalize();

      for (auto& [key, d_impl] : current_level.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        // false indicates there is no need to update the pushed_data map to
        // automatically pop data when the context is popped because we are
        // already doing this now.
        d_impl->pop_after_finalize();
      }

      // TODO deal with pending unfreeze for destroyed data

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_level.alloc_adapters)
      {
        current_level.alloc_adapters->clear();
      }

      // Destroy the current level state
      levels.pop_back();
    }

    /**
     * @brief Get the nesting depth
     */
    size_t depth() const
    {
      return levels.size() - 1;
    }

    /**
     * @brief Returns a reference to the context at a specific level
     */
    auto& get_ctx(size_t level)
    {
      _CCCL_ASSERT(level < levels.size(), "invalid value");
      return levels[level].ctx;
    }

    /**
     * @brief Returns a const reference to the context at a specific level
     */
    const auto& get_ctx(size_t level) const
    {
      _CCCL_ASSERT(level < levels.size(), "invalid value");
      return levels[level].ctx;
    }

    cudaStream_t get_stream(size_t level) const
    {
      _CCCL_ASSERT(level < levels.size(), "invalid value");
      return levels[level].support_stream;
    }

    // void track_pushed_data(int data_id, ::std::shared_ptr<stackable_logical_data_impl_base> data_impl)
    void track_pushed_data(int data_id, const ::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
    {
      _CCCL_ASSERT(data_impl, "invalid value");
      levels[depth()].pushed_data[data_id] = data_impl;
    }

    void untrack_pushed_data(int data_id)
    {
      size_t erased = levels[depth()].pushed_data.erase(data_id);
      // We must have erased exactly one value (at least one otherwise it was already removed, and it must be pushed
      // only once (TODO check))
      _CCCL_ASSERT(erased == 1, "invalid value");
    }

    void retain_data(size_t level, ::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
    {
      _CCCL_ASSERT(level < levels.size(), "invalid value");
      levels[level].retained_data.push_back(mv(data_impl));
    }

  private:
    // State for each nested level
    ::std::vector<per_level> levels;

    // Handles to retain some asynchronous states, we maintain it separately
    // from levels because we keep its entries even when we pop a level
    ::std::vector<async_resources_handle> async_handles;
  };

  stackable_ctx()
      : pimpl(::std::make_shared<impl>())
  {}

  cudaStream_t get_stream(size_t level) const
  {
    return pimpl->get_stream(level);
  }

  const auto& get_ctx(size_t level) const
  {
    return pimpl->get_ctx(level);
  }

  auto& get_ctx(size_t level)
  {
    return pimpl->get_ctx(level);
  }

  const auto& operator()() const
  {
    return get_ctx(depth());
  }

  auto& operator()()
  {
    return get_ctx(depth());
  }

  void push()
  {
    pimpl->push();
  }

  void pop()
  {
    pimpl->pop();
  }

  size_t depth() const
  {
    return pimpl->depth();
  }

  template <typename T>
  auto logical_data(shape_of<T> s)
  {
    // fprintf(stderr, "initialize from shape.\n");
    return stackable_logical_data(*this, depth(), true, get_ctx(0).logical_data(mv(s)), true);
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    return stackable_logical_data(*this, depth(), true, get_ctx(0).template logical_data<T>(elements, more_sizes...), true);
  }

  template <typename T>
  auto logical_data_no_export(shape_of<T> s)
  {
    // fprintf(stderr, "initialize from shape.\n");
    return stackable_logical_data(*this, depth(), true, get_ctx(depth()).logical_data(mv(s)), false);
  }

  template <typename T, typename... Sizes>
  auto logical_data_no_export(size_t elements, Sizes... more_sizes)
  {
    return stackable_logical_data(*this, depth(), true, get_ctx(depth()).template logical_data<T>(elements, more_sizes...), false);
  }


  stackable_logical_data<void_interface> logical_token();

  template <typename... Pack>
  auto logical_data(Pack&&... pack)
  {
    // fprintf(stderr, "initialize from value.\n");
    return stackable_logical_data(*this, depth(), false, get_ctx(depth()).logical_data(::std::forward<Pack>(pack)...), true);
  }

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
  void retain_data(size_t level, ::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
  {
    _CCCL_ASSERT(pimpl, "uninitialized context");
    _CCCL_ASSERT(data_impl, "invalid value");
    pimpl->retain_data(level, mv(data_impl));
  }

  // Helper function to process a single argument
  template <typename T1>
  void process_argument(const T1&) const
  {
    // Do nothing for non-stackable_task_dep
  }

  template <typename T1, typename reduce_op, bool initialize>
  void process_argument(const stackable_task_dep<T1, reduce_op, initialize>& dep) const
  {
    // If the logical data was not at the appropriate level, we may
    // automatically push it. In this case, we need to update the logical data
    // referred stored in the task_dep object.
    bool need_update = dep.get_d().validate_access(*this, dep.get_access_mode());
    if (need_update)
    {
      // The underlying dep eg. obtained when calling l.read() was resulting in
      // an task_dep_untyped where the logical data was the one at the "top of
      // the stack". Since a push method was done automatically, the logical
      // data that needs to be used was incorrect, and we update it.
      dep.underlying_dep().update_data(dep.get_d().get_ld());
    }
  }

  // Process the parameter pack
  template <typename... Pack>
  void process_pack(const Pack&... pack) const
  {
    // fprintf(stderr, "process_pack begin.\n");
    (process_argument(pack), ...);
    //  fprintf(stderr, "process_pack end.\n");
  }

  template <typename... Pack>
  auto task(Pack&&... pack)
  {
    process_pack(pack...);
    return get_ctx(depth()).task(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    process_pack(pack...);
    return get_ctx(depth()).parallel_for(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel(Pack&&... pack)
  {
    process_pack(pack...);
    return get_ctx(depth()).cuda_kernel(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel_chain(Pack&&... pack)
  {
    process_pack(pack...);
    return get_ctx(depth()).cuda_kernel_chain(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }
#endif

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    process_pack(pack...);
    return get_ctx(depth()).host_launch(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  auto task_fence()
  {
    return get_ctx(depth()).task_fence();
  }

  template <typename... Pack>
  void push_affinity(Pack&&... pack) const
  {
    process_pack(pack...);
    get_ctx(depth()).push_affinity(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  void pop_affinity() const
  {
    get_ctx(depth()).pop_affinity();
  }

  auto& async_resources() const
  {
    return get_ctx(depth()).async_resources();
  }

  // void track_pushed_data(int data_id, ::std::shared_ptr<stackable_logical_data_impl_base> data_impl)
  void track_pushed_data(int data_id, const ::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
  {
    pimpl->track_pushed_data(data_id, data_impl);
  }

  void untrack_pushed_data(int data_id)
  {
    pimpl->untrack_pushed_data(data_id);
  }

  auto dot_section(::std::string symbol) const
  {
    return get_ctx(depth()).dot_section(mv(symbol));
  }

  void finalize()
  {
    // There must be only one level left
    _CCCL_ASSERT(depth() == 0, "All nested levels must have been popped");

    get_ctx(depth()).finalize();
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
    // this state in the context, in a vector of retained states
    class state : public stackable_logical_data_impl_state_base
    {
    public:
      state(stackable_ctx _sctx)
          : sctx(mv(_sctx))
      {}

      virtual void pop_before_finalize() const override
      {
        _CCCL_ASSERT(s.size() == sctx.depth() || s.size() == sctx.depth() + 1, "internal error");

        // Maybe the logical data was already destroyed if the stackable
        // logical data was destroyed before ctx pop, and that the data state
        // was retained. In this case, we don't remove an entry from the vector
        // of logical data.
        if (s.size() == sctx.depth() + 1)
        {
          // Remove aliased logical data because this wasn't done yet
          s.pop_back();
        }
      }

      virtual void pop_after_finalize() const override
      {
        // We are going to unfreeze the data, which is currently being used
        // in a (graph) ctx that uses this stream to launch the graph
        cudaStream_t stream = sctx.get_stream(depth());

        frozen_s.back().unfreeze(stream);

        // Remove frozen logical data
        frozen_s.pop_back();
      }

      size_t depth() const
      {
        return s.size() - 1 + offset_depth;
      }

      mutable stackable_ctx sctx;
      mutable ::std::vector<logical_data<T>> s;

      // When stacking data, we freeze data from the lower levels, these are
      // their frozen counterparts. This vector has one item less than the
      // vector of logical data.
      mutable ::std::vector<frozen_logical_data<T>> frozen_s;

      // If the logical data was created at a level that is not directly the root of the context, we remember this
      // offset
      size_t base_depth = 0;
      size_t offset_depth = 0;

      ::std::string symbol;
    };

  public:
    impl() = default;
    impl(stackable_ctx sctx_,
         size_t target_depth,
         bool ld_from_shape,
         logical_data<T> ld,
         bool can_export,
         data_place where = data_place::invalid)
        : sctx(mv(sctx_))
    {
      impl_state = ::std::make_shared<state>(sctx);

      impl_state->base_depth = target_depth;

      // TODO pass this offset directly rather than a boolean for more flexibility ? (e.g. creating a ctx of depth 2, export at depth 1, not 0 ...)
      impl_state->offset_depth = can_export?0:target_depth;

      // fprintf(stderr, "stackable_logical_data::impl %p - base depth %ld offset depth %ld can export ? %d\n", this, impl_state->base_depth, impl_state->offset_depth, can_export);

      // Save the logical data at the base level
      impl_state->s.push_back(ld);

      // If necessary, import data recursively until we reach the target depth
      for (size_t current_depth = impl_state->offset_depth + 1; current_depth <= target_depth; current_depth++)
      {
        push(ld_from_shape ? access_mode::write : access_mode::rw, where);
      }
    }

    // stackable_logical_data::impl::~impl
    ~impl()
    {
      if (!impl_state)
      {
        return;
      }

      auto& s       = impl_state->s;
      size_t s_size = s.size();

      // There is at least the logical data passed when we created the stackable_logical_data
      _CCCL_ASSERT(s_size > 0, "internal error");

      s.pop_back();

      // If there were at least 2 logical data before pop_back, there remains
      // at least one logical data, so we need to retain the impl.
      if (s_size > 1)
      {
        // This state will be retained until we pop the ctx enough times
        size_t offset_depth = impl_state->offset_depth;

        if (offset_depth < sctx.depth())
        {
            // If that was an exportable data, offset_depth = 0, it can be deleted
            // when the first stacked level is popped (ie. when the CUDA graph has
            // been executed)
            sctx.retain_data(offset_depth + 1, mv(impl_state));
        }
      }

      impl_state = nullptr;
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Define move constructor and move assignment operator
    impl(impl&&) noexcept            = default;
    impl& operator=(impl&&) noexcept = default;

    const auto& get_ld() const
    {
      return impl_state->s.back();
    }
    auto& get_ld()
    {
      return impl_state->s.back();
    }

    /* Push one level up (from the current data depth) */
    void push(access_mode m, data_place where = data_place::invalid) const
    {
      // fprintf(stderr, "stackable_logical_data::push() %s mode = %s\n", symbol.c_str(), access_mode_string(m));

      const size_t ctx_depth          = sctx.depth();
      const size_t current_data_depth = depth();

      // (current_data_depth + 1) is the data depth after pushing
      _CCCL_ASSERT(ctx_depth >= current_data_depth + 1, "Invalid depth");

      context& from_ctx = sctx.get_ctx(current_data_depth);
      context& to_ctx   = sctx.get_ctx(current_data_depth + 1);

      auto& s        = impl_state->s;
      auto& frozen_s = impl_state->frozen_s;

      // fprintf(stderr,
      //         "pushing data (%p) %ld[%s,%p]->%ld[%s,%p] (ctx depth %ld)\n",
      //         this,
      //         current_data_depth,
      //         from_ctx.to_string().c_str(),
      //         &from_ctx,
      //         current_data_depth + 1,
      //         to_ctx.to_string().c_str(),
      //         &to_ctx,
      //         ctx_depth);

      if (where == data_place::invalid)
      {
        // use the default place
        where = from_ctx.default_exec_place().affine_data_place();
      }

      _CCCL_ASSERT(where != data_place::invalid, "Invalid data place");

      // Freeze the logical data at the top
      logical_data<T>& from_data = s.back();
      frozen_logical_data<T> f   = from_ctx.freeze(from_data, m, mv(where));

      // Save the frozen data in a separate vector
      frozen_s.push_back(f);

      // FAKE IMPORT : use the stream needed to support the (graph) ctx
      cudaStream_t stream = sctx.get_stream(current_data_depth + 1);

      T inst  = f.get(where, stream);
      auto ld = to_ctx.logical_data(inst, where);

      if (!impl_state->symbol.empty())
      {
        // TODO reflect base/offset depths
        ld.set_symbol(impl_state->symbol + "." + ::std::to_string(current_data_depth + 1 - impl_state->base_depth));
      }

      // Keep track of data that were pushed in this context. Note that the ID
      // used is the ID of the logical data at this level. This will be used to
      // pop data automatically when nested contexts are popped.
      sctx.track_pushed_data(ld.get_unique_id(), impl_state);

      // Save the logical data created to keep track of the data instance
      // obtained with the get method of the frozen_logical_data object.
      s.push_back(mv(ld));
    }

    /* Pop one level down */
    void pop_before_finalize() const
    {
      impl_state->pop_before_finalize();
    }

    void pop_after_finalize() const
    {
      impl_state->pop_after_finalize();
    }

    size_t depth() const
    {
      return impl_state->depth();
    }

    // TODO move to state
    void set_symbol(::std::string symbol_)
    {
      impl_state->symbol = mv(symbol_);
      // TODO reflect base/offset depths
      impl_state->s.back().set_symbol(impl_state->symbol + "." + ::std::to_string(depth() - impl_state->base_depth));
    }

    auto get_symbol() const
    {
      return impl_state->symbol;
    }

    // TODO why making sctx private or why do we need to expose this at all ?
    auto& get_sctx()
    {
      return sctx;
    }

    size_t get_offset_depth() const {
        return impl_state->offset_depth;
    }

    // Get the access mode used to freeze at depth d
    // TODO move to state
    access_mode get_frozen_mode(size_t d) const
    {
      // fprintf(stderr, "get_frozen_mode d = %ld impl_state->offset_depth %ld, impl_state->frozen_s.size() %ld\n", d, impl_state->offset_depth, impl_state->frozen_s.size());
      _CCCL_ASSERT(d >= impl_state->offset_depth, "invalid value");
      _CCCL_ASSERT(d - impl_state->offset_depth < impl_state->frozen_s.size(), "invalid value");
      return impl_state->frozen_s[d - impl_state->offset_depth].get_access_mode();
    }

  private:
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
  stackable_logical_data(stackable_ctx sctx, bool ld_from_shape, size_t target_depth, logical_data<T> ld, bool can_export)
      : pimpl(::std::make_shared<impl>(sctx, ld_from_shape, target_depth, mv(ld), can_export))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>, "");
    static_assert(::std::is_move_assignable_v<stackable_logical_data>, "");
  }

  const auto& get_ld() const
  {
    return pimpl->get_ld();
  }
  auto& get_ld()
  {
    return pimpl->get_ld();
  }

  size_t depth() const
  {
    return pimpl->depth();
  }

  void push(access_mode m, data_place where = data_place::invalid) const
  {
    pimpl->push(m, mv(where));

#if 0
    // Keep track of data that were pushed in this context. Note that the ID
    // used is the ID of the logical data at this level.
    pimpl->get_sctx().track_pushed_data(get_ld().get_unique_id(), pimpl.get());
#endif
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

  auto shape() const
  {
    return get_ld().shape();
  }

  auto& set_symbol(::std::string symbol)
  {
    pimpl->set_symbol(mv(symbol));
    return *this;
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
  bool validate_access(const stackable_ctx& sctx, access_mode m) const
  {
    // fprintf(stderr,
    //         "Calling validate_access() on stackable_logical_data %p - symbol=%s - requested mode %s\n",
    //         pimpl.get(),
    //         get_symbol().c_str(),
    //         access_mode_string(m));

    size_t offset_depth = pimpl->get_offset_depth();

    // Are we trying to access a logical data that cannot be "exported" at this level ?
    _CCCL_ASSERT(sctx.depth() >= offset_depth, "Invalid value");

    // Fast path : do nothing if we are not in a stacked ctx
    if (sctx.depth() == offset_depth)
    {
      // fprintf(stderr, "validate : FAST PATH %s : sctx.depth() == offset_depth == %ld\n", get_symbol().c_str(), offset_depth);
      return false;
    }

    // TODO assert sctx == this->sctx

    _CCCL_ASSERT(m == access_mode::read || m == access_mode::rw || m == access_mode::write,
                 "Unsupported access mode in nested context");

    // If the stackable logical data is already at the appropriate depth, we
    // simply need to ensure we don't make an illegal access (eg. writing a
    // read only variable)
    size_t d = depth();

    // Data depth cannot be higher than context depth
    _CCCL_ASSERT(sctx.depth() >= d, "Invalid value");

    if (sctx.depth() == d)
    {
      _CCCL_ASSERT(d > 0, "data must have already been pushed");
      _CCCL_ASSERT(access_mode_is_compatible(pimpl->get_frozen_mode(d - 1), m), "Invalid access mode");
      // fprintf(stderr, "VALIDATED ACCESS ON stackable_logical_data %p - symbol=%s\n", pimpl.get(),
      //  get_symbol().c_str());
      return false;
    }

    // If we reach this point, this means we need to automatically push data

    // The access mode will be very conservative for these implicit accesses
    access_mode push_mode = (m == access_mode::write) ? access_mode::write : access_mode::rw;

    while (sctx.depth() > d)
    {
      // fprintf(stderr,
      //         "AUTOMATIC PUSH of data %p (symbol=%s) with mode %s at depth %d\n",
      //         pimpl.get(),
      //         get_symbol().c_str(),
      //         access_mode_string(push_mode),
      //         d);
      pimpl->push(push_mode, data_place::current_device());
      d++;
    }

    return true;
  }

private:
  ::std::shared_ptr<impl> pimpl;
};

inline stackable_logical_data<void_interface> stackable_ctx::logical_token()
{
  return stackable_logical_data<void_interface>(*this, depth(), true, get_ctx(0).logical_token(), true);
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
