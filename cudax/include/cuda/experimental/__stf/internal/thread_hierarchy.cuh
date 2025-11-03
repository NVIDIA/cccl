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
 * @brief The thread hierarchy class describes how threads are organized in a kernel
 *
 * It can be used to describe a hierarchy of threads, the size of the different
 * levels, and whether they can synchronize or not, for example.
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

#include <cuda/experimental/__stf/internal/cooperative_group_system.cuh>
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/internal/slice.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/cyclic_shape.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief This describes a hierarchy of threads used to implement a launch construct.
 *
 * It corresponds to a thread_hierarchy_spec which was mapped on the execution
 * place, and provides simple mechanisms at different levels in the hierarchy :
 * - getting the rank and the size of the calling thread in the hierarchy
 * - synchronizing all threads in a specific level
 * - getting a local storage attached to a specific level.
 *
 * This class is intended to be passed by value from the host to CUDA kernels
 * so it does not contain pointers or indirection (except in the implementation
 * of system-wide barriers)
 */
template <auto... spec>
class thread_hierarchy
{
  // Depth of this hierarchy (each level has two spec values, `sync` and `width`)
  static constexpr size_t depth = [](auto x, auto y, auto...) {
    // Also run some checks
    static_assert(::std::is_same_v<decltype(x), bool>, "You must use bool for the odd arguments of thread_hierarchy.");
    static_assert(::std::is_same_v<decltype(y), size_t>,
                  "You must use size_t for the even arguments of thread_hierarchy.");
    // Two spec parameters per depth level
    return sizeof...(spec) / 2;
  }(spec..., false, size_t(0));

  // Fetch the inner thread_hierarchy (peels off two template arguments)
  template <auto... subspec>
  struct inner_t;
  template <auto x, auto y, auto... subspec>
  struct inner_t<x, y, subspec...>
  {
    using type = thread_hierarchy<subspec...>;
  };

public:
  // For tag-types
  thread_hierarchy() = default;

  template <auto...>
  friend class thread_hierarchy;

  // Construct from an outer thread hierarchy (peel off one level)
  template <bool outer_sync, size_t outer_width>
  _CCCL_HOST_DEVICE thread_hierarchy(const thread_hierarchy<outer_sync, outer_width, spec...>& outer)
      : devid(outer.devid)
      , launch_config(outer.launch_config)
      , cg_system(outer.cg_system)
      , device_tmp(outer.device_tmp)
      , system_tmp(outer.system_tmp)
  {
    for (size_t i : each(0, depth))
    {
      level_sizes[i] = outer.level_sizes[i + 1];
      mem_sizes[i]   = outer.mem_sizes[i + 1];
    }
  }

  /**
   *  This takes an interpreted_execution_policy which is the mapping of a spec
   *  on the hardware, and generates a thread_hierarchy object that can be passed
   *  to kernels as an argument
   */
  thread_hierarchy(int devid, interpreted_execution_policy<spec...>& p)
      : devid(devid)
  {
    launch_config = p.get_config();

    // If we may synchronize across multiple devices.
    cg_system = p.cg_system;

    size_t i = 0;
    for (auto& l : p.get_levels())
    {
      level_sizes[i] = l.width();
      assert(l.width() > 0);
      mem_sizes[i] = l.get_mem();
      i++;
    }
  }

  /**
   * @brief Get the statically-specified width at a specific level
   * @param level The level
   * @return The width (0 if width is dynamic)
   */
  static inline constexpr size_t static_width(size_t level)
  {
    size_t data[] = {spec...};
    return data[1 + 2 * level];
  }

  _CCCL_HOST_DEVICE const ::std::array<size_t, 3>& get_config() const
  {
    return launch_config;
  }

  // Rank from the root
  _CCCL_HOST_DEVICE size_t rank([[maybe_unused]] int level, [[maybe_unused]] int root_level) const
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (int tid = threadIdx.x; int bid = blockIdx.x;
       // config : ndevs    = launch_config[0]
       //          nblocks  = launch_config[1]
       //          nthreads = launch_config[2]
       const int nblocks  = launch_config[1];
       const int nthreads = launch_config[2];

       int global_id = tid + bid * nthreads + devid * nblocks * nthreads;

       // An entry at level l represents a total of width[level] * width[level+1] ... * width[depth - 1] threads.
       size_t level_effective_size      = 1;
       size_t root_level_effective_size = 1;

       if constexpr (depth > 0) {
         for (size_t l = level + 1; l < depth; l++)
         {
           level_effective_size *= level_sizes[l];
         }

         for (size_t l = root_level + 1; l < depth; l++)
         {
           root_level_effective_size *= level_sizes[l];
         }
       }

       return (global_id % root_level_effective_size)
       / level_effective_size;),
      (return 0;))
    _CCCL_UNREACHABLE();
  }

  _CCCL_HOST_DEVICE size_t size([[maybe_unused]] int level, [[maybe_unused]] int root_level) const
  {
    if constexpr (depth == 0)
    {
      return 1;
    }

    assert(root_level < level);
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (return 1;),
      (
        // Could have been : return level_prod_size[level]/level_prod_size[root_level];
        // Divisions are probably more expensive.
        size_t s = 1; for (int l = root_level; l < level; l++) { s *= level_sizes[l + 1]; }

        return s;))
    _CCCL_UNREACHABLE();
  }

  _CCCL_HOST_DEVICE size_t size(int level = int(depth) - 1) const
  {
    return size(level, -1);
  }

  _CCCL_HOST_DEVICE size_t rank(int level = int(depth) - 1) const
  {
    return rank(level, -1);
  }

  _CCCL_HOST_DEVICE void sync([[maybe_unused]] int level = 0)
  {
    assert(level >= 0);
    assert(level < depth);

    NV_IF_TARGET(
      NV_IS_DEVICE,
      (
        // Check that it is legal to synchronize (use a static assertion in the future)
        assert(may_sync(level));

        // We compute the products of level_sizes and config in reversed order.

        // This is the number of threads to synchronize
        size_t target_size = 1;
        for (int l = level; l < depth; l++) { target_size *= level_sizes[l]; }

        // We then compute the number of threads for each of the different
        // scopes (system, device, blocks) ... and compare with this number of
        // threads

        // Test the different config levels
        size_t block_scope_size = launch_config[2];
        if (target_size == block_scope_size) {
          cooperative_groups::this_thread_block().sync();
          return;
        }

        size_t device_scope_size = block_scope_size * launch_config[1];
        if (target_size == device_scope_size) {
          cooperative_groups::this_grid().sync();
          return;
        }

        size_t ndevs             = launch_config[0];
        size_t system_scope_size = device_scope_size * ndevs;
        if (target_size == system_scope_size) {
          cg_system.sync(devid, ndevs);
          return;
        }

        // Unsupported configuration
        assert(0);))
  }

  template <typename T, typename... Others>
  auto remove_first_tuple_element(const ::std::tuple<T, Others...>& t)
  {
    return ::std::make_tuple(::std::get<Others>(t)...);
  }

  template <typename shape_t, typename P, typename... sub_partitions>
  _CCCL_HOST_DEVICE auto apply_partition(const shape_t& s, const ::std::tuple<P, sub_partitions...>& t) const
  {
    auto s0         = P::apply(s, pos4(rank(0)), dim4(size(0)));
    auto sans_first = make_tuple_indexwise<sizeof...(sub_partitions)>([&](auto index) {
      return ::std::get<index + 1>(t);
    });
    if constexpr (sizeof...(sub_partitions))
    {
      return inner().apply_partition(s0, sans_first);
    }
    else
    {
      return s0;
    }
  }

  // Default partitioner tuple<blocked, blocked...., blocked, cyclic>
  template <typename shape_t>
  _CCCL_HOST_DEVICE auto apply_partition(const shape_t& s) const
  {
    if constexpr (depth == 1)
    {
      return cyclic_partition::apply(s, pos4(rank(0)), dim4(size(0)));
    }
    else
    {
      auto s0 = blocked_partition::apply(s, pos4(rank(0)), dim4(size(0)));
      return inner().apply_partition(s0);
    }
  }

  /**
   * @brief Get the inner thread hierarchy (starting one level down)
   *
   * @return `thread_hierarchy` instantiated with `spec` sans the first two arguments
   */
  _CCCL_HOST_DEVICE auto inner() const
  {
    return typename inner_t<spec...>::type(*this);
  }

  template <typename T>
  _CCCL_HOST_DEVICE slice<T> storage(int level)
  {
    assert(level >= 0);
    assert(level < depth);

    // We compute the products of level_sizes and config in reversed order.

    // This is the number of threads to synchronize
    size_t target_size = 1;
    for (int l = level; l < depth; l++)
    {
      target_size *= level_sizes[l];
    }

    // We then compute the number of threads for each of the different
    // scopes (system, device, blocks) ... and compare with this number of
    // threads

    NV_IF_TARGET(
      NV_IS_DEVICE,
      (
        size_t nelems = mem_sizes[level] / sizeof(T);

        // Test the different config levels
        size_t block_scope_size = launch_config[2];
        if (target_size == block_scope_size) {
          // Use dynamic shared memory
          extern __shared__ T dyn_buffer[];
          return make_slice(&dyn_buffer[0], nelems);
        }

        size_t device_scope_size = block_scope_size * launch_config[1];
        if (target_size == device_scope_size) {
          // Use device memory
          return make_slice(static_cast<T*>(device_tmp), nelems);
        }

        size_t ndevs             = launch_config[0];
        size_t system_scope_size = device_scope_size * ndevs;
        if (target_size == system_scope_size) {
          // Use system memory (managed memory)
          return make_slice(static_cast<T*>(system_tmp), nelems);
        }))

    // Unsupported configuration : memory must be a scope boundaries
    assert(!"Unsupported configuration : memory must be a scope boundaries");
    return make_slice(static_cast<T*>(nullptr), 0);
  }

  void set_device_tmp(void* addr)
  {
    device_tmp = addr;
  }

  void set_system_tmp(void* addr)
  {
    system_tmp = addr;
  }

  void set_devid(int d)
  {
    devid = d;
  }

private:
  // On which device is this running ? (or -1 if host or ignored)
  int devid = -1;

  // ndevs * nblocks * nthreads
  ::std::array<size_t, 3> launch_config = {};

  reserved::cooperative_group_system cg_system;

  // width of each level
  ::std::array<size_t, depth> level_sizes = {};

  // What is the memory associated to each level ?
  ::std::array<size_t, depth> mem_sizes = {};

  // Address of the per-device buffer in device memory, if any
  void* device_tmp = nullptr;

  // Base address for system-wide buffers (managed memory), if any
  void* system_tmp = nullptr;

  /* Special case to work-around spec = <> */
  template <size_t Idx>
  _CCCL_HOST_DEVICE bool may_sync_impl(int) const
  {
    return false;
  }

  template <size_t Idx, bool sync, size_t w, auto... remaining>
  _CCCL_HOST_DEVICE bool may_sync_impl(int level) const
  {
    static_assert(Idx < sizeof...(spec) / 2);
    return (level == Idx) ? sync : may_sync_impl<Idx + 1, remaining...>(level);
  }

  /* Evaluate at runtime if we can call sync on the specified level */
  _CCCL_HOST_DEVICE bool may_sync(int level) const
  {
    return may_sync_impl<0, spec...>(level);
  }
};

#ifdef UNITTESTED_FILE
#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
namespace reserved
{
template <auto... spec>
__global__ void unit_test_thread_hierarchy(thread_hierarchy<spec...> h)
{
  assert(h.size(0) == 2);
  assert(h.size(1) == 2 * 32);

  // Rank within the whole system
  assert(h.rank(1, -1) == 32 + threadIdx.x + blockIdx.x * blockDim.x);
  assert(h.rank(1) == 32 + threadIdx.x + blockIdx.x * blockDim.x);

  assert(h.rank(0, -1) == 1); // device
  assert(h.rank(1, 0) == threadIdx.x + blockIdx.x * blockDim.x);
}
} // end namespace reserved

UNITTEST("thread hierarchy indexing")
{
  interpreted_execution_policy<false, size_t(0), false, size_t(0)> p;

  p.add_level({::std::make_pair(hw_scope::device, 2UL)});
  p.add_level({::std::make_pair(hw_scope::block, 8UL), ::std::make_pair(hw_scope::thread, 4UL)});

  int dev_id = 1;
  auto h     = thread_hierarchy<false, size_t(0), false, size_t(0)>(dev_id, p);

  static_assert(h.static_width(0) == 0);
  static_assert(h.static_width(1) == 0);

  auto config = p.get_config();
  reserved::unit_test_thread_hierarchy<<<config[1], config[2]>>>(h);

  cuda_safe_call(cudaDeviceSynchronize());
};

namespace reserved
{
template <auto... spec>
__global__ void unit_test_thread_hierarchy_sync(thread_hierarchy<spec...> h)
{
  assert(h.rank(1, 0) == threadIdx.x);
  assert(h.rank(0, -1) == blockIdx.x);
  h.sync(1);
  h.sync(0);
}
} // end namespace reserved

UNITTEST("thread hierarchy sync")
{
  interpreted_execution_policy<true, size_t(0), true, size_t(0)> p;

  // 2 levels
  p.add_level({::std::make_pair(hw_scope::block, 8UL)});
  p.set_level_sync(0, true);
  p.add_level({::std::make_pair(hw_scope::thread, 4UL)});
  p.set_level_sync(1, true);

  size_t dev_id = 1;
  auto h        = thread_hierarchy(dev_id, p);

  auto config = p.get_config();

  void* args[] = {&h};
  cuda_safe_call(cudaLaunchCooperativeKernel(
    (void*) reserved::unit_test_thread_hierarchy_sync<true, size_t(0), true, size_t(1)>,
    config[1],
    config[2],
    args,
    0,
    0));

  cuda_safe_call(cudaDeviceSynchronize());
};

namespace reserved
{
template <auto... spec>
__global__ void unit_test_thread_hierarchy_inner_sync(thread_hierarchy<spec...> h)
{
  h.inner().sync();
  auto h_inner = h.inner();
  assert(h_inner.size() == 4);
}
} // end namespace reserved

UNITTEST("thread hierarchy inner sync")
{
  interpreted_execution_policy<false, size_t(0), true, size_t(0)> p;

  // 2 levels
  p.add_level({::std::make_pair(hw_scope::block, 8UL)});
  p.set_level_sync(0, false);
  p.add_level({::std::make_pair(hw_scope::thread, 4UL)});
  p.set_level_sync(1, true);

  int dev_id = 1;
  auto h     = thread_hierarchy<false, size_t(0), true, size_t(0)>(dev_id, p);

  auto config = p.get_config();
  reserved::unit_test_thread_hierarchy_inner_sync<false, size_t(0), true, size_t(0)><<<config[1], config[2]>>>(h);

  cuda_safe_call(cudaDeviceSynchronize());
};

#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
