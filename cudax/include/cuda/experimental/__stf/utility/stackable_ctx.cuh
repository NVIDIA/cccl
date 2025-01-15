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

namespace cuda::experimental::stf
{

template <typename T>
class stackable_logical_data;

/**
 * @brief Base class with a virtual pop method to enable type erasure
 *
 * This is used to implement the automatic call to pop() on logical data when a
 * context level is popped.
 */
class stackable_logical_data_impl_base
{
public:
  virtual ~stackable_logical_data_impl_base() = default;
  virtual void pop()                          = 0;
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
      per_level(context ctx, cudaStream_t support_stream, ::std::optional<stream_adapter> alloc_adapters)
          : ctx(mv(ctx))
          , support_stream(mv(support_stream))
          , alloc_adapters(mv(alloc_adapters))
      {}

      context ctx;
      cudaStream_t support_stream;
      // A wrapper to forward allocations from a level to the previous one (none is used at the root level)
      ::std::optional<stream_adapter> alloc_adapters;

      // This map keeps track of the logical data that were pushed in this level
      // key: logical data's unique id
      ::std::unordered_map<int, ::std::shared_ptr<stackable_logical_data_impl_base>> pushed_data;
    };

  public:
    impl()
    {
      // Create the root level
      push();
    }

    ~impl() = default;

    // Delete copy constructor and copy assignment operator
    impl(const impl&) = delete;
    impl& operator=(const impl&) = delete;

    // Define move constructor and move assignment operator
    impl(impl&&) noexcept = default;
    impl& operator=(impl&&) noexcept = default;

    /**
     * @brief Create a new nested level
     */
    void push()
    {
      // These resources are not destroyed when we pop, so we create it only if needed
      if (async_handles.size() < levels.size())
      {
        async_handles.emplace_back();
      }

      if (levels.size() == 0)
      {
        levels.emplace_back(stream_ctx(), nullptr, ::std::nullopt);
      }
      else
      {
        // Get a stream from previous context (we haven't pushed the new one yet)
        cudaStream_t stream = levels[depth()].ctx.pick_stream();

        auto gctx = graph_ctx(stream, async_handles.back());

        auto wrapper = stream_adapter(gctx, stream);
        // FIXME : issue with the deinit phase
        //   gctx.update_uncached_allocator(wrapper.allocator());

        levels.emplace_back(gctx, stream, wrapper);
      }
    }

    /**
     * @brief Terminate the current nested level and get back to the previous one
     */
    void pop()
    {
      _CCCL_ASSERT(levels.size() > 0, "Calling pop while no context was pushed");

      auto& current_level = levels.back();

      // Automatically pop data if needed
      for (auto& [key, d_impl] : current_level.pushed_data)
      {
        d_impl->pop();
      }

      // Ensure everything is finished in the context
      current_level.ctx.finalize();

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_level.alloc_adapters.has_value())
      {
        current_level.alloc_adapters.value().clear();
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
      return levels[level].ctx;
    }

    /**
     * @brief Returns a const reference to the context at a specific level
     */
    const auto& get_ctx(size_t level) const
    {
      return levels[level].ctx;
    }

    cudaStream_t get_stream(size_t level) const
    {
      return levels[level].support_stream;
    }

    void track_pushed_data(int data_id, ::std::shared_ptr<stackable_logical_data_impl_base> data_impl)
    {
      levels[depth()].pushed_data[data_id] = mv(data_impl);
    }

    void untrack_pushed_data(int data_id)
    {
      size_t erased = levels[depth()].pushed_data.erase(data_id);
      // We must have erased exactly one value (at least one otherwise it was already removed, and it must be pushed
      // only once (TODO check))
      _CCCL_ASSERT(erased == 1, "invalid value");
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

  template <typename... Pack>
  auto logical_data(Pack&&... pack)
  {
    return stackable_logical_data(*this, depth(), get_ctx(depth()).logical_data(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto task(Pack&&... pack)
  {
    return get_ctx(depth()).task(::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    return get_ctx(depth()).parallel_for(::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    return get_ctx(depth()).host_launch(::std::forward<Pack>(pack)...);
  }

  void track_pushed_data(int data_id, ::std::shared_ptr<stackable_logical_data_impl_base> data_impl)
  {
    pimpl->track_pushed_data(data_id, mv(data_impl));
  }

  void untrack_pushed_data(int data_id)
  {
    pimpl->untrack_pushed_data(data_id);
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
  class impl : public stackable_logical_data_impl_base
  {
  public:
    impl() = default;
    impl(stackable_ctx sctx, size_t depth, logical_data<T> ld)
        : base_depth(depth)
        , sctx(mv(sctx))
    {
      s.push_back(ld);
    }

    const auto& get_ld() const
    {
      check_level_mismatch();
      return s.back();
    }
    auto& get_ld()
    {
      check_level_mismatch();
      return s.back();
    }

    void push(access_mode m, data_place where = data_place::invalid)
    {
      // We have not pushed yet, so the current depth of the logical data is
      // the one before pushing
      context& from_ctx = sctx.get_ctx(depth());
      context& to_ctx   = sctx.get_ctx(depth() + 1);

      // Ensure this will match the depth of the context after pushing
      _CCCL_ASSERT(sctx.depth() == depth() + 1, "Invalid depth");

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
      cudaStream_t stream = sctx.get_stream(depth());

      T inst  = f.get(where, stream);
      auto ld = to_ctx.logical_data(inst, where);

      if (!symbol.empty())
      {
        ld.set_symbol(symbol + "." + ::std::to_string(depth() + 1));
      }

      s.push_back(mv(ld));
    }

    virtual void pop() override
    {
      // We are going to unfreeze the data, which is currently being used
      // in a (graph) ctx that uses this stream to launch the graph
      cudaStream_t stream = sctx.get_stream(depth());

      frozen_logical_data<T>& f = frozen_s.back();
      f.unfreeze(stream);

      // Remove frozen logical data
      frozen_s.pop_back();
      // Remove aliased logical data
      s.pop_back();
    }

    size_t depth() const
    {
      return s.size() - 1 + base_depth;
    }

    void set_symbol(::std::string symbol_)
    {
      symbol = mv(symbol_);
      s.back().set_symbol(symbol + "." + ::std::to_string(depth()));
    }

    auto& get_sctx()
    {
      return sctx;
    }

  private:
    void check_level_mismatch() const
    {
      if (depth() != sctx.depth())
      {
        fprintf(stderr, "Warning: mismatch between ctx level %ld and data level %ld\n", sctx.depth(), depth());
      }
    }

    mutable ::std::vector<logical_data<T>> s;

    // When stacking data, we freeze data from the lower levels, these are
    // their frozen counterparts. This vector has one item less than the
    // vector of logical data.
    mutable ::std::vector<frozen_logical_data<T>> frozen_s;

    // If the logical data was created at a level that is not directly the root of the context, we remember this
    // offset
    size_t base_depth = 0;
    stackable_ctx sctx; // in which stackable context was this created ?

    ::std::string symbol;
  };

public:
  stackable_logical_data() = default;

  template <typename... Args>
  stackable_logical_data(stackable_ctx sctx, size_t depth, logical_data<T> ld)
      : pimpl(::std::make_shared<impl>(mv(sctx), depth, mv(ld)))
  {}

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

  void push(access_mode m, data_place where = data_place::invalid)
  {
    pimpl->push(m, mv(where));

    // Keep track of data that were pushed in this context. Note that the ID
    // used is the ID of the logical data at this level.
    pimpl->get_sctx().track_pushed_data(get_ld().get_unique_id(), pimpl);
  }

  void pop()
  {
    // We remove the data from the map before popping it to have the id of the
    // logical data. Doing so will prevent the automatic call to pop() when the
    // context level gets popped.
    pimpl->get_sctx().untrack_pushed_data(get_ld().get_unique_id());

    pimpl->pop();
  }

  // Helpers
  template <typename... Pack>
  auto read(Pack&&... pack) const
  {
    return get_ld().read(::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto write(Pack&&... pack)
  {
    return get_ld().write(::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto rw(Pack&&... pack)
  {
    return get_ld().rw(::std::forward<Pack>(pack)...);
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

private:
  ::std::shared_ptr<impl> pimpl;
};

} // end namespace cuda::experimental::stf
