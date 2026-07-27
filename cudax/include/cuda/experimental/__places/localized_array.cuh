//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of the localized_array class which dispatches a VMM
 *        allocation over multiple data places using a partitioner function.
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

#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>

#include <cuda/experimental/__places/places.cuh>

#include <array>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

namespace cuda::experimental::places
{
/**
 * @brief Check if localized allocation statistics should be printed
 */
inline bool localized_alloc_stats_enabled()
{
  static bool enabled = [] {
    const char* env = ::std::getenv("CUDASTF_LOCALIZED_ALLOC_STATS");
    return env != nullptr && ::std::string(env) != "0";
  }();
  return enabled;
}

//! Default number of elements sampled per block to decide the block's owner
inline constexpr size_t localized_placement_default_probes = 10;

/**
 * @brief Statistics describing how a localized allocation - or a dry-run
 * evaluation of one - distributes a tensor over data places.
 *
 * Produced by evaluate_localized_placement() and by localized_array (see
 * localized_array::get_stats()). This is the returnable form of the report
 * previously only printed to stderr under CUDASTF_LOCALIZED_ALLOC_STATS.
 */
struct localized_stats
{
  size_t total_bytes = 0; //!< requested payload size in bytes
  size_t vm_bytes    = 0; //!< block-rounded virtual reservation size in bytes
  size_t block_size  = 0; //!< placement granularity in bytes
  size_t nblocks     = 0; //!< number of placement blocks
  size_t nallocs     = 0; //!< physical allocations after merging same-owner runs

  size_t total_samples    = 0; //!< probes drawn by the block-owner sampler
  size_t matching_samples = 0; //!< probes agreeing with the chosen block owner

  //! Bytes backed by each place, keyed by data_place::to_string()
  ::std::unordered_map<::std::string, size_t> bytes_per_place;

  //! Bytes owned by each grid position, keyed by the position's linear index
  //! (dim4::get_index of the pos4; friendlier than strings across FFI)
  ::std::unordered_map<size_t, size_t> bytes_per_grid_index;

  //! Fraction of sampled elements whose owner matches the block-majority
  //! owner: an estimate of the fraction of bytes that end up local to their
  //! owner once ownership is quantized to blocks.
  double accuracy() const
  {
    return total_samples == 0 ? 1.0 : static_cast<double>(matching_samples) / static_cast<double>(total_samples);
  }
};

/**
 * @brief Placement granularity used when the caller does not specify one:
 * the device allocation granularity when a device is present, or the
 * customary 2 MiB VMM granularity for GPU-free (offline) evaluation. The
 * granularity query is the only driver interaction.
 */
inline size_t default_placement_block_size()
{
  int ndevs = 0;
  if (cudaGetDeviceCount(&ndevs) == cudaSuccess && ndevs > 0)
  {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = 0;
    return cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  }
  cudaGetLastError();
  return 2 * 1024 * 1024;
}

/**
 * @brief Decide the owner of each placement block by sampled majority vote.
 *
 * For each block, `probes` elements are sampled (reproducibly: one seeded
 * generator for the whole computation) and the most frequent owner wins. The
 * majority vote is what tolerates partitions whose boundaries do not align
 * with the block granularity: a block straddling two owners goes to the one
 * owning most of it.
 *
 * @param owner_of Callable mapping a linear element index to the pos4 of its
 *        owner in the grid
 * @param nblocks Number of placement blocks
 * @param block_size_bytes Size of a placement block in bytes (must be at
 *        least ``elemsize``)
 * @param elemsize Size of one element in bytes (must be at least 1)
 * @param total_elems Total number of elements (probes are clipped to it)
 * @param probes Number of samples per block
 * @param stats Accumulates total/matching sample counts
 */
template <typename OwnerFn>
::std::vector<pos4> compute_block_owners(
  OwnerFn&& owner_of,
  size_t nblocks,
  size_t block_size_bytes,
  size_t elemsize,
  size_t total_elems,
  size_t probes,
  localized_stats& stats)
{
  if (elemsize == 0 || block_size_bytes < elemsize)
  {
    throw ::std::invalid_argument("placement blocks must hold at least one element (elemsize in [1, block size])");
  }
  const size_t block_elems = block_size_bytes / elemsize;

  // Fixed seed: placement must be reproducible from one run to the next
  ::std::mt19937 gen(0x5EED);
  ::std::uniform_int_distribution<size_t> dis(0, block_elems - 1);

  probes = ::std::max<size_t>(1, ::std::min(probes, block_elems));

  ::std::vector<pos4> owners;
  owners.reserve(nblocks);

  ::std::vector<pos4> sampled_pos(probes);
  for (size_t i = 0; i < nblocks; i++)
  {
    // First element of the block (exact, so non-dividing element sizes do
    // not accumulate drift across blocks)
    const size_t block_start = i * block_size_bytes / elemsize;
    for (size_t sample = 0; sample < probes; sample++)
    {
      // Clip: the last block may extend past the payload
      const size_t index  = ::std::min(block_start + dis(gen), total_elems - 1);
      sampled_pos[sample] = owner_of(index);
    }

    ::std::unordered_map<pos4, size_t, ::cuda::experimental::stf::hash<pos4>> sample_cnt;
    for (const auto& s : sampled_pos)
    {
      ++sample_cnt[s];
    }

    size_t max_cnt = 0;
    pos4 max_pos;
    for (const auto& s : sample_cnt)
    {
      if (s.second > max_cnt)
      {
        max_pos = s.first;
        max_cnt = s.second;
      }
    }

    stats.total_samples += probes;
    stats.matching_samples += max_cnt;

    owners.push_back(max_pos);
  }

  return owners;
}

/**
 * @brief Call `fn(owner, first_block, num_blocks)` for each maximal run of
 * consecutive blocks with the same owner.
 */
template <typename F>
void for_each_owner_run(const ::std::vector<pos4>& owners, F&& fn)
{
  for (size_t i = 0; i < owners.size();)
  {
    const pos4 p = owners[i];
    size_t j     = 0;
    while ((i + j < owners.size()) && (owners[i + j] == p))
    {
      j++;
    }
    fn(p, i, j);
    i += j;
  }
}

/**
 * @brief An allocator that takes a mapping function to dispatch an allocation over multiple data places.
 *
 * This is the mechanism used to implement the data_place of a grid of execution places.
 * Uses CUDA Virtual Memory Management (VMM) to create a contiguous virtual address range
 * backed by physical allocations on different devices according to the partitioner.
 */
class localized_array
{
  struct metadata
  {
    metadata(data_place place_, size_t size_, size_t offset_)
        : alloc_handle{}
        , place(mv(place_))
        , size(size_)
        , offset(offset_)
    {}

    CUmemGenericAllocationHandle alloc_handle;
    const data_place place;
    size_t size;
    size_t offset;
  };

public:
  template <typename F>
  localized_array(
    exec_place grid,
    partition_fn_t mapper,
    F&& delinearize,
    size_t total_size,
    size_t elemsize,
    dim4 data_dims,
    size_t probes = localized_placement_default_probes)
      : grid(mv(grid))
      , mapper(mv(mapper))
      , total_size_bytes(total_size * elemsize)
      , data_dims(data_dims)
      , elemsize(elemsize)
  {
    const dim4 grid_dims = this->grid.get_dims();
    init(
      [&](size_t index) {
        const pos4 coords = delinearize(index);
        pos4 eplace_coords(0);
        this->mapper(&eplace_coords, coords, this->data_dims, grid_dims);
        return eplace_coords;
      },
      total_size,
      probes);
  }

  /**
   * @brief Construct from a generic owner function instead of a raw
   * partition_fn_t mapper (e.g. a stateful partition object). The owner
   * function maps a linear element index to the grid position owning it and
   * is only used during construction.
   */
  localized_array(exec_place grid,
                  const ::std::function<pos4(size_t)>& owner_of,
                  size_t total_size,
                  size_t elemsize,
                  dim4 data_dims,
                  size_t probes = localized_placement_default_probes)
      : grid(mv(grid))
      , total_size_bytes(total_size * elemsize)
      , data_dims(data_dims)
      , elemsize(elemsize)
  {
    init(owner_of, total_size, probes);
  }

  localized_array()                                  = delete;
  localized_array(const localized_array&)            = delete;
  localized_array(localized_array&&)                 = delete;
  localized_array& operator=(const localized_array&) = delete;
  localized_array& operator=(localized_array&&)      = delete;

  ~localized_array()
  {
    for (auto& item : meta)
    {
      size_t offset = item.offset;
      size_t sz     = item.size;
      cuda_try(cuMemUnmap(base_ptr + offset, sz));
      cuda_try(cuMemRelease(item.alloc_handle));
    }

    cuda_try(cuMemAddressFree(base_ptr, vm_total_size_bytes));
  }

  void* get_base_ptr() const
  {
    return reinterpret_cast<void*>(base_ptr);
  }

  /**
   * @brief Placement statistics of this allocation (see localized_stats)
   */
  const localized_stats& get_stats() const
  {
    return stats;
  }

  /*
   * This equality operator is used to find entries in an allocation cache which match a specific request
   */
  template <typename... P>
  bool operator==(::cuda::std::tuple<P&...> t) const
  {
    // tuple arguments :
    // 0 : grid, 1 : mapper, 2 : delinearize function, 3 : total size, 4 elem_size, 5 : data_dims
    bool result = grid == ::cuda::std::get<0>(t) && mapper == ::cuda::std::get<1>(t)
               && this->total_size_bytes == ::cuda::std::get<3>(t) * ::cuda::std::get<4>(t)
               && elemsize == ::cuda::std::get<4>(t) && data_dims == ::cuda::std::get<5>(t);
    if (result)
    {
      assert(this->total_size_bytes == ::cuda::std::get<3>(t) * ::cuda::std::get<4>(t));
      assert(data_dims == ::cuda::std::get<5>(t));
    }
    return result;
  }

private:
  void init(const ::std::function<pos4(size_t)>& owner_of, size_t total_size, size_t probes)
  {
    if (elemsize == 0)
    {
      throw ::std::invalid_argument("localized_array requires an element size of at least 1 byte");
    }

    cuda_try(cudaFree(nullptr));

    const int ndevs = cuda_try<cudaGetDeviceCount>();
    CUdevice dev    = cuda_try<cuCtxGetDevice>();

    int supportsVMM = cuda_try<cuDeviceGetAttribute>(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
    EXPECT(supportsVMM == 1, "Cannot create a localized_array object on this machine because it does not support VMM.");

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = dev;

    size_t alloc_granularity_bytes = cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    block_size_bytes = alloc_granularity_bytes;

    vm_total_size_bytes =
      ((total_size_bytes + alloc_granularity_bytes - 1) / alloc_granularity_bytes) * alloc_granularity_bytes;

    size_t nblocks = vm_total_size_bytes / alloc_granularity_bytes;

    base_ptr = cuda_try<cuMemAddressReserve>(vm_total_size_bytes, 0ULL, 0ULL, 0ULL);

    ::std::vector<CUmemAccessDesc> accessDesc(ndevs);
    for (int d = 0; d < ndevs; d++)
    {
      accessDesc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc[d].location.id   = d;
      accessDesc[d].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    stats.total_bytes = total_size_bytes;
    stats.vm_bytes    = vm_total_size_bytes;
    stats.block_size  = block_size_bytes;
    stats.nblocks     = nblocks;

    const ::std::vector<pos4> owners =
      compute_block_owners(owner_of, nblocks, block_size_bytes, elemsize, total_size, probes, stats);

    meta.reserve(nblocks);

    for_each_owner_run(owners, [&](pos4 p, size_t first_block, size_t num_blocks) {
      data_place place  = grid_pos_to_place(p);
      size_t alloc_size = num_blocks * block_size_bytes;
      stats.bytes_per_place[place.to_string()] += alloc_size;
      stats.bytes_per_grid_index[this->grid.get_dims().get_index(p)] += alloc_size;
      meta.emplace_back(mv(place), alloc_size, first_block * block_size_bytes);
    });

    stats.nallocs = meta.size();

    if (localized_alloc_stats_enabled())
    {
      print_stats(owners);
    }

    for (auto& item : meta)
    {
      int item_dev = device_ordinal(item.place);

      cuda_try(item.place.mem_create(&item.alloc_handle, item.size));

      _CCCL_ASSERT(item.offset + item.size <= vm_total_size_bytes, "Allocation offset out of bounds");
      cuda_try(cuMemMap(base_ptr + item.offset, item.size, 0ULL, item.alloc_handle, 0ULL));

      for (int d = 0; d < ndevs; d++)
      {
        int set_access = 1;
        if (item_dev != d)
        {
          set_access = cuda_try<cudaDeviceCanAccessPeer>(d, item_dev);

          if (!set_access)
          {
            fprintf(stderr, "Warning : Cannot enable peer access between devices %d and %d\n", d, item_dev);
          }
        }

        if (set_access == 1)
        {
          cuda_try(cuMemSetAccess(base_ptr + item.offset, item.size, &accessDesc[d], 1ULL));
        }
      }
    }
  }

  void print_stats(const ::std::vector<pos4>& owners)
  {
    fprintf(stderr, "\n=== Localized Array Allocation Statistics ===\n");
    fprintf(stderr, "Total size: %zu bytes (%.2f MB)\n", stats.total_bytes, stats.total_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "VM reservation: %zu bytes (%.2f MB)\n", stats.vm_bytes, stats.vm_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Block size: %zu bytes (%.2f KB)\n", stats.block_size, stats.block_size / 1024.0);
    fprintf(stderr, "Number of blocks: %zu (merged into %zu allocations)\n", stats.nblocks, stats.nallocs);
    fprintf(stderr, "Number of places: %zu\n", stats.bytes_per_place.size());

    fprintf(stderr, "\nAllocation distribution by place:\n");
    for (const auto& entry : stats.bytes_per_place)
    {
      double pct = 100.0 * entry.second / stats.vm_bytes;
      fprintf(stderr,
              "  %s: %zu bytes (%.2f MB, %.1f%%)\n",
              entry.first.c_str(),
              entry.second,
              entry.second / (1024.0 * 1024.0),
              pct);
    }

    if (stats.total_samples > 0)
    {
      fprintf(stderr,
              "\nPlacement accuracy: %.1f%% (%zu/%zu samples matched chosen position)\n",
              100.0 * stats.accuracy(),
              stats.matching_samples,
              stats.total_samples);
    }

    fprintf(stderr, "\nAllocation map (%zu allocations):\n", meta.size());
    fprintf(stderr, "  %-6s  %-12s  %-12s  %-10s  %s\n", "Index", "Offset", "Size", "Blocks", "Place");
    fprintf(stderr, "  %-6s  %-12s  %-12s  %-10s  %s\n", "-----", "------", "----", "------", "-----");
    for (size_t idx = 0; idx < meta.size(); idx++)
    {
      const auto& item   = meta[idx];
      size_t num_blocks  = item.size / stats.block_size;
      size_t start_block = item.offset / stats.block_size;
      (void) start_block;
      fprintf(stderr,
              "  %-6zu  %-12zu  %-12zu  %-10zu  %s\n",
              idx,
              item.offset,
              item.size,
              num_blocks,
              item.place.to_string().c_str());
    }

    fprintf(stderr, "\nBlock ownership map (each char = 1 block, 0-9/a-z = place index):\n  ");
    ::std::unordered_map<::std::string, char> place_to_char;
    char next_char = '0';
    for (size_t i = 0; i < owners.size(); i++)
    {
      ::std::string place_str = grid_pos_to_place(owners[i]).to_string();
      if (place_to_char.find(place_str) == place_to_char.end())
      {
        place_to_char[place_str] = next_char;
        if (next_char == '9')
        {
          next_char = 'a';
        }
        else
        {
          next_char++;
        }
      }
      fprintf(stderr, "%c", place_to_char[place_str]);
      if ((i + 1) % 80 == 0)
      {
        fprintf(stderr, "\n  ");
      }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "\n  Legend:\n");
    for (const auto& entry : place_to_char)
    {
      fprintf(stderr, "    %c = %s\n", entry.second, entry.first.c_str());
    }

    fprintf(stderr, "==============================================\n\n");
  }

  data_place grid_pos_to_place(pos4 grid_pos)
  {
    return grid.get_place(grid_pos).affine_data_place();
  }

  exec_place grid;
  partition_fn_t mapper = nullptr;
  ::std::vector<metadata> meta;

  size_t block_size_bytes = 0;
  size_t total_size_bytes = 0;

  size_t vm_total_size_bytes = 0;

  CUdeviceptr base_ptr = 0;

  dim4 data_dims;
  size_t elemsize = 0;

  localized_stats stats;
};

/**
 * @brief Evaluate - without allocating anything - how a localized allocation
 * would distribute a tensor over the places of a grid.
 *
 * Runs the exact same block-owner decision procedure as localized_array and
 * returns the resulting statistics, so callers can score a candidate mapping
 * (and tune its parameters) before committing memory.
 *
 * Extents follow the dimension-0-fastest convention of dim4::get_index().
 *
 * @param grid Grid of execution places the mapper distributes over
 * @param mapper Partition function mapping element coordinates to a place
 * @param data_dims Extents of the tensor
 * @param elemsize Size of one element in bytes
 * @param probes Number of samples per block for the majority vote
 * @param block_size Placement granularity in bytes; 0 selects the device
 *        allocation granularity when a device is present, or a 2 MiB default
 *        otherwise (this granularity query is the only driver interaction)
 */
[[nodiscard]] inline localized_stats evaluate_localized_placement(
  const exec_place& grid,
  partition_fn_t mapper,
  dim4 data_dims,
  size_t elemsize,
  size_t probes     = localized_placement_default_probes,
  size_t block_size = 0)
{
  if (block_size == 0)
  {
    block_size = default_placement_block_size();
  }

  localized_stats stats;

  const size_t total_elems = data_dims.size();
  stats.total_bytes        = total_elems * elemsize;
  stats.vm_bytes           = ((stats.total_bytes + block_size - 1) / block_size) * block_size;
  stats.block_size         = block_size;
  stats.nblocks            = stats.vm_bytes / block_size;

  const dim4 grid_dims = grid.get_dims();

  const ::std::vector<pos4> owners = compute_block_owners(
    [&](size_t index) {
      pos4 eplace_coords(0);
      mapper(&eplace_coords, data_dims.index_to_pos(index), data_dims, grid_dims);
      return eplace_coords;
    },
    stats.nblocks,
    block_size,
    elemsize,
    total_elems,
    probes,
    stats);

  for_each_owner_run(owners, [&](pos4 p, size_t /*first_block*/, size_t num_blocks) {
    const data_place place = grid.get_place(p).affine_data_place();
    stats.bytes_per_place[place.to_string()] += num_blocks * block_size;
    stats.bytes_per_grid_index[grid.get_dims().get_index(p)] += num_blocks * block_size;
    stats.nallocs++;
  });

  return stats;
}

inline ::std::unordered_map<void*, ::std::unique_ptr<localized_array>>& get_composite_alloc_registry()
{
  static ::std::unordered_map<void*, ::std::unique_ptr<localized_array>> reg;
  return reg;
}

inline void* allocate_composite_data_place(const data_place_composite& p, dim4 data_dims, size_t elemsize)
{
  const exec_place& grid       = p.get_grid();
  const partition_fn_t& mapper = p.get_partitioner();
  // Linear memory follows the dimension-0-fastest convention of
  // dim4::get_index(), like STF slices; the partitioner receives true element
  // coordinates within data_dims.
  auto delinearize = [data_dims](size_t i) {
    return data_dims.index_to_pos(i);
  };
  auto arr  = ::std::make_unique<localized_array>(grid, mapper, delinearize, data_dims.size(), elemsize, data_dims);
  void* ptr = arr->get_base_ptr();
  get_composite_alloc_registry()[ptr] = ::std::move(arr);
  return ptr;
}

inline void deallocate_composite_data_place(void* ptr)
{
  get_composite_alloc_registry().erase(ptr);
}
} // namespace cuda::experimental::places
