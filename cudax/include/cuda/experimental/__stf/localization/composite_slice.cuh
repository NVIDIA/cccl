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
 * @brief Implementation of the localized_array class which is used to allocate a piece
 * of data that is dispatched over multiple data places
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
#include <cuda/experimental/__stf/utility/memory.cuh>
#include <cuda/experimental/__stf/utility/traits.cuh>

#include <list>
#include <random>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief An allocator that takes a mapping function to dispatch an allocation over multiple data places.
 *
 * This is the mechanism used to implement the data_place of a grid of execution places.
 */
class localized_array
{
  struct metadata
  {
    metadata(int dev_, size_t size_, size_t offset_)
        : alloc_handle{}
        , dev(dev_)
        , size(size_)
        , offset(offset_)
    {}

    CUmemGenericAllocationHandle alloc_handle;
    int dev;
    size_t size;
    size_t offset;
  };

public:
  // ::std::function<pos4(size_t)> delinearize : translate the index in a buffer into a position in the data
  // TODO pass mv(place)
  template <typename F>
  localized_array(exec_place_grid grid,
                  get_executor_func_t mapper,
                  F&& delinearize,
                  size_t total_size,
                  size_t elemsize,
                  dim4 data_dims)
      : grid(mv(grid))
      , mapper(mv(mapper))
      , total_size_bytes(total_size * elemsize)
      , data_dims(data_dims)
      , elemsize(elemsize)
  {
    // Regardless of the grid, we allow all devices to access that localized array
    const int ndevs = cuda_try<cudaGetDeviceCount>();
    CUdevice dev    = cuda_try<cuCtxGetDevice>();

    /* Check whether the current device supports UVA */
    int supportsVMM = cuda_try<cuDeviceGetAttribute>(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
    //        fprintf(stderr, "VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED ? %d\n", supportsVMM);
    EXPECT(supportsVMM == 1, "Cannot create a localized_array object on this machine because it does not support VMM.");

    /* Get allocation granularity */

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location            = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = dev};

    size_t alloc_granularity_bytes = cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    //        fprintf(stderr, "GRANULARITY = %ld KB\n", alloc_granularity_bytes / 1024);

    // To make our life simpler for now: we assume that we only allocate full blocks
    block_size_bytes = alloc_granularity_bytes;

    vm_total_size_bytes =
      ((total_size_bytes + alloc_granularity_bytes - 1) / alloc_granularity_bytes) * alloc_granularity_bytes;

    // Number of pages to assign (note that we will try to make less allocations in practice by grouping pages)
    size_t nblocks = vm_total_size_bytes / alloc_granularity_bytes;

    // Reserve a range of virtual addresses, round up size to accommodate granularity requirements
    cuda_try(cuMemAddressReserve(&base_ptr, vm_total_size_bytes, 0ULL, 0ULL, 0ULL));

    // fprintf(stderr, "cuMemAddressReserve => %p + %ld (%ld KB)\n", (void *)base_ptr, vm_total_size_bytes,
    //                 vm_total_size_bytes / 1024);

    ::std::vector<CUmemAccessDesc> accessDesc(ndevs);
    for (int d = 0; d < ndevs; d++)
    {
      accessDesc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc[d].location.id   = d;
      accessDesc[d].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Compute mapping at allocation granularity
    ::std::vector<pos4> owner;
    owner.reserve(nblocks);
    for (size_t i = 0; i < nblocks; i++)
    {
      owner.push_back(
        block_to_grid_pos(i * block_size_bytes / elemsize, alloc_granularity_bytes / elemsize, delinearize));
    }

    // We create one allocation handle per block
    meta.reserve(nblocks);

    // Try to merge blocks with the same position
    for (size_t i = 0; i < nblocks;)
    {
      pos4 p   = owner[i];
      size_t j = 0;
      // Count consecutive blocks with the same position in the grid
      while ((i + j < nblocks) && (owner[i + j] == p))
      {
        j++;
      }

      meta.emplace_back(grid_pos_to_dev(p), j * alloc_granularity_bytes, i * block_size_bytes);

      i += j;
    }

    // fprintf(stderr, "GOT %ld effective blocks (%ld blocks)\n", nblocks_effective, nblocks);

    // Create a physical allocation per block, this is not mapped in
    // virtual memory yet.
    for (auto& item : meta)
    {
      CUmemAllocationProp create_prop = {};
      create_prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
      create_prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
      // Physically allocate this block on the appropriate device
      // fprintf(stderr, "MAP EFFECTIVE BLOCK %d on dev %d\n", i, hdl_dev[i]);
      create_prop.location.id = item.dev;

      cuda_safe_call(cuMemCreate(&item.alloc_handle, item.size, &create_prop, 0));

      // Device where this was allocated

      assert(item.offset + item.size <= vm_total_size_bytes);
      // fprintf(stderr, "cuMemMap(base_ptr %p = %p + %p, %p, 0ULL, hdl[%d], 0ULL)\n", base_ptr + offset,
      // base_ptr,
      //        offset, sz, i);
      cuda_safe_call(cuMemMap(base_ptr + item.offset, item.size, 0ULL, item.alloc_handle, 0ULL));

      for (int d = 0; d < ndevs; d++)
      {
        int set_access = 1;
        if (item.dev != d)
        {
          cuda_safe_call(cudaDeviceCanAccessPeer(&set_access, d, item.dev));

          if (!set_access)
          {
            fprintf(stderr, "Warning : Cannot enable peer access between devices %d and %d\n", d, item.dev);
          }
        }

        if (set_access == 1)
        {
          cuda_safe_call(cuMemSetAccess(base_ptr + item.offset, item.size, &accessDesc[d], 1ULL));
        }
      }
    }
    // fprintf(stderr, "localized_array (this = %p) : nblocks_effective %ld\n", this, nblocks_effective);
  }

  localized_array()                                  = delete;
  localized_array(const localized_array&)            = delete;
  localized_array(localized_array&&)                 = delete;
  localized_array& operator=(const localized_array&) = delete;
  localized_array& operator=(localized_array&&)      = delete;

  ~localized_array()
  {
    // fprintf(stderr, "~localized_array (this = %p) ... base ptr %p vm_total_size_bytes %ld - nblocks_effective
    // %ld\n", this, (void *)base_ptr, vm_total_size_bytes, nblocks_effective);
    for (auto& item : meta)
    {
      size_t offset = item.offset;
      size_t sz     = item.size;
      cuda_safe_call(cuMemUnmap(base_ptr + offset, sz));
      cuda_safe_call(cuMemRelease(item.alloc_handle));
    }

    cuda_safe_call(cuMemAddressFree(base_ptr, vm_total_size_bytes));
  }

  // Convert the device pointer in the device API back to a raw void * pointer
  void* get_base_ptr() const
  {
    return reinterpret_cast<void*>(base_ptr);
  }

  /*
   * This equality operator is for example used to find entries in an allocation cache which match a specific request
   */
  template <typename... P>
  bool operator==(::std::tuple<P&...> t) const
  {
    // tuple arguments :
    // 0 : grid, 1 : mapper, 2 : delinearize function, 3 : total size, 4 elem_size, 5 : data_dims
    bool result = grid == ::std::get<0>(t) && mapper == ::std::get<1>(t)
               && this->total_size_bytes == ::std::get<3>(t) * ::std::get<4>(t) && elemsize == ::std::get<4>(t);
    if (result)
    {
      assert(this->total_size_bytes == ::std::get<3>(t) * ::std::get<4>(t));
      assert(data_dims == ::std::get<5>(t));
    }
    return result;
  }

  void merge(const event_list& source)
  {
    prereqs.merge(source);
  }

  void merge_into(event_list& target)
  {
    target.merge(mv(prereqs));
    prereqs.clear();
  }

private:
  int grid_pos_to_dev(pos4 grid_pos)
  {
    return device_ordinal(grid.get_place(grid_pos).affine_data_place());
  }

  // linearized_index : expressed in number of entries from the base, not bytes
  // allocation_granularity expressed in number of entries
  template <typename F>
  pos4 block_to_grid_pos(size_t linearized_index, size_t allocation_granularity, F&& delinearize)
  {
#if 0
        // Our first strategy consists in mapping the block at the location of the first entry of the block
        return index_to_grid_pos(linearized_index, delinearize);
#else
    ::std::random_device rd;
    ::std::mt19937 gen(rd());
    ::std::uniform_int_distribution<> dis(0, static_cast<int>(allocation_granularity - 1));

    const size_t nsamples = 10;
    ::std::array<pos4, nsamples> sampled_pos;
    for (size_t sample = 0; sample < nsamples; sample++)
    {
      size_t index        = linearized_index + dis(gen);
      sampled_pos[sample] = index_to_grid_pos(index, delinearize);
    }

    // Count the number of occurrences of each pos
    ::std::unordered_map<pos4, size_t, hash<pos4>> sample_cnt;
    for (auto& s : sampled_pos)
    {
      ++sample_cnt[s];
    }

    size_t max_cnt = 0;
    pos4 max_pos;
    for (auto& s : sample_cnt)
    {
      if (s.second > max_cnt)
      {
        max_pos = s.first;
        max_cnt = s.second;
      }
    }

    // ::std::cout << "GOT BEST POS for offset " << linearized_index << " -> " << max_pos.string() << ::std::endl;

    return max_pos;
#endif
  }

  template <typename F>
  pos4 index_to_grid_pos(size_t linearized_index, F&& delinearize)
  {
    // Logical coordinates of this index
    pos4 coords = delinearize(linearized_index);

    pos4 eplace_coords = mapper(coords, data_dims, grid.get_dims());

    return eplace_coords;
  }

  event_list prereqs; // To allow reuse in a cache
  exec_place_grid grid;
  get_executor_func_t mapper = nullptr;
  ::std::vector<metadata> meta;

  // sizes in number of elements, not bytes !! TODO rename
  size_t block_size_bytes = 0;
  size_t total_size_bytes = 0;

  // size of the VA reservation in bytes
  size_t vm_total_size_bytes = 0;

  // Start of the VA reservation
  CUdeviceptr base_ptr = 0;

  // Parameter saved to allow reusing data
  dim4 data_dims;
  size_t elemsize = 0;
};

/**
 * @brief A very simple allocation cache for slices in composite data places
 */
class composite_slice_cache
{
public:
  composite_slice_cache()                             = default;
  composite_slice_cache(const composite_slice_cache&) = delete;
  composite_slice_cache(composite_slice_cache&)       = delete;
  composite_slice_cache(composite_slice_cache&&)      = default;

  [[nodiscard]] event_list deinit()
  {
    event_list result;
    cache.each([&](auto& obj) {
      obj.merge_into(result);
    });
    return result;
  }

  // Save one localized array in the cache
  void put(::std::unique_ptr<localized_array> a, const event_list& prereqs)
  {
    EXPECT(a.get());
    a->merge(prereqs);
    cache.put(mv(a));
  }

  // Look if there is a matching entry. Return it if found, create otherwise
  template <typename F>
  ::std::unique_ptr<localized_array>
  get(const data_place& place,
      get_executor_func_t mapper,
      F&& delinearize,
      size_t total_size,
      size_t elem_size,
      dim4 data_dims)
  {
    EXPECT(place.is_composite());
    return cache.get(place.get_grid(), mapper, ::std::forward<F>(delinearize), total_size, elem_size, data_dims);
  }

private:
  reserved::linear_pool<localized_array> cache;
};
} // end namespace cuda::experimental::stf::reserved
