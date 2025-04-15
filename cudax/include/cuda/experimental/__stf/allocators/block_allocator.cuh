//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Facilities to define specific allocators
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

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/places/places.cuh>

#include <mutex>

namespace cuda::experimental::stf
{

class backend_ctx_untyped;

/**
 * @brief Interface for block allocator
 *
 * This class provides an interface for a block allocator. It defines methods for allocation, deallocation,
 * initialization, and printing of information.
 */
class block_allocator_interface
{
public:
  /**
   * @brief Default constructor
   */
  block_allocator_interface() = default;

  /**
   * @brief Virtual destructor
   */
  virtual ~block_allocator_interface() = default;

  /**
   * @brief Asynchronous allocation of memory
   *
   * This method asynchronously allocates `*s` bytes on the specified memory node.
   * Upon success, `*ptr` will contain the address of the allocated buffer.
   * If the allocation fails, `*s` will contain a negative value.
   *
   * @param ctx Untyped backend context
   * @param memory_node Memory node where the allocation should occur
   * @param s Pointer to the size of the allocation
   * @param prereqs List of events that should finish before the allocation starts
   * @return void* Pointer to the allocated memory
   */
  virtual void*
  allocate(backend_ctx_untyped&, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) = 0;

  /**
   * @brief Deallocation of memory
   *
   * This method deallocates memory previously allocated by the allocate method.
   *
   * @param ctx Untyped backend context
   * @param memory_node Memory node where the deallocation should occur
   * @param prereqs List of events that should finish before the deallocation starts
   * @param ptr Pointer to the memory to be deallocated
   * @param sz Size of the memory to be deallocated
   */
  virtual void
  deallocate(backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) = 0;

  /**
   * @brief Deinitialization of the allocator
   * @return List of events indicating that this was deinitialized
   */
  virtual event_list deinit(backend_ctx_untyped&) = 0;

  /**
   * @brief String representation of the allocator
   *
   * This method returns a string representation of the allocator.
   *
   * @return std::string String representation of the allocator
   */
  virtual ::std::string to_string() const = 0;

  /**
   * @brief Print information about the allocator
   *
   * This method prints information about the allocator to stderr.
   */
  virtual void print_info() const
  {
    const auto s = to_string();
    fprintf(stderr, "No additional info for allocator of kind \"%.*s\".\n", static_cast<int>(s.size()), s.data());
  }
};

/**
 * @brief Type-erased counterpart to block_allocator
 */
class block_allocator_untyped
{
public:
  template <typename T>
  friend class block_allocator;

  block_allocator_untyped() = default;

  block_allocator_untyped(::std::shared_ptr<block_allocator_interface> ptr)
      : pimpl(mv(ptr))
  {}

  template <typename ctx_t>
  block_allocator_untyped(ctx_t& ctx, ::std::shared_ptr<block_allocator_interface> ptr)
      : pimpl(mv(ptr))
  {
    ctx.attach_allocator(pimpl);
  }

  void* allocate(backend_ctx_untyped& bctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs)
  {
    return pimpl->allocate(bctx, memory_node, s, prereqs);
  }

  void deallocate(backend_ctx_untyped& bctx, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz)
  {
    return pimpl->deallocate(bctx, memory_node, prereqs, ptr, sz);
  }

  event_list deinit(backend_ctx_untyped& bctx)
  {
    return pimpl->deinit(bctx);
  }

  ::std::string to_string() const
  {
    return pimpl->to_string();
  }

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

private:
  ::std::shared_ptr<block_allocator_interface> pimpl;
};

/**
 * @brief Handle to an allocator that is attached to a context
 *
 * When a block_allocator is constructed, it is attached to the context passed
 * as the first argument. The allocator will be detached when the context is
 * finalized (ie. deinit will be called).
 */
template <typename T>
class block_allocator : public block_allocator_untyped
{
public:
  template <typename ctx_t, typename... Args>
  block_allocator(ctx_t& ctx, Args&&... args)
      : block_allocator_untyped(ctx, ::std::make_shared<T>(::std::forward<Args>(args)...))
  {}

  block_allocator(::std::shared_ptr<block_allocator_interface> ptr)
      : block_allocator_untyped(mv(ptr))
  {}
};

} // end namespace cuda::experimental::stf
