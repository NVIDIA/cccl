// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_AGENT_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_AGENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
// 0 selects the measured architecture/geometry scopes below.
// 1 forces int32 tile arithmetic and shared prefixes on every architecture.
// 2 also narrows shared prefixes to uint16; arithmetic remains int32.
// Global pointers, boundary rows and output offsets always retain Offset.
#  ifndef _CUDAX_SORT_MWAY_TILE_LOCAL_MODE
#    define _CUDAX_SORT_MWAY_TILE_LOCAL_MODE 0
#  endif
static_assert(_CUDAX_SORT_MWAY_TILE_LOCAL_MODE >= 0 && _CUDAX_SORT_MWAY_TILE_LOCAL_MODE <= 2);
#  include <cub/block/block_load_to_shared.cuh>
#  include <cub/block/block_merge_sort.cuh>
#  include <cub/block/block_store.cuh>
#  include <cub/thread/thread_load.cuh>
#  include <cub/util_type.cuh>

#  include <cuda/ptx>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/type_traits>

#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/merge_path.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/serial_merge.h>

#  include <nv/target>

#  include <cstring>
// Disable scoped instruction paths independently of the indexing/refill controls.
// Eligibility is checked by optimized_for for each architecture and geometry.
#  ifndef _CUDAX_SORT_MWAY_MWAY_INSTRUCTION_OPTIMIZATION
#    define _CUDAX_SORT_MWAY_MWAY_INSTRUCTION_OPTIMIZATION 1
#  endif
#  include <cuda/experimental/mgmn/__algorithm/common.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/partition.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/policy.h>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
// Mode bits:1 batches value gathers before bulk stores;2 uses read-only global loads.
// Value input slices are immutable during a merge launch. These architecture defaults
// were selected using completed whole-sort timings and independent NCU/Nsys captures.
_CCCL_HOST_DEVICE_API constexpr int mway_arch_value_gather_mode(int cc)
{
#  if defined(_CUDAX_SORT_MWAY_MWAY_VALUE_GATHER_MODE)
  (void) cc;
  static_assert(_CUDAX_SORT_MWAY_MWAY_VALUE_GATHER_MODE >= 0 && _CUDAX_SORT_MWAY_MWAY_VALUE_GATHER_MODE <= 3);
  return _CUDAX_SORT_MWAY_MWAY_VALUE_GATHER_MODE;
#  else
  switch (cc)
  {
    case 890:
      return 2;
    case 900:
      return 3;
    case 1000:
      return 3;
    case 1030:
      return 3;
    case 1070:
      return 1;
    default:
      return 0;
  }
#  endif
}
#  if defined(__CUDA_ARCH__)
inline constexpr int mway_value_gather_mode = mway_arch_value_gather_mode(__CUDA_ARCH__);
#  else
inline constexpr int mway_value_gather_mode = 0;
#  endif
inline constexpr bool mway_value_gather_batch = (mway_value_gather_mode & 1) != 0;
template <typename Value>
_CCCL_DEVICE_API _CCCL_FORCEINLINE Value mway_gather_value(const Value* ptr)
{
  if constexpr ((mway_value_gather_mode & 2) != 0)
  {
    return CUB_NS_QUALIFIER::ThreadLoad<CUB_NS_QUALIFIER::LOAD_LDG>(ptr);
  }
  else
  {
    return *ptr;
  }
}

// Buffering of the merge tree, decided per target architecture at compile time: two ping-pong
// tile buffers where the SM has 228 KB of shared memory (cc 9.0, 10.x), one buffer with
// register-staged rounds where it has 100-128 KB (cc 8.x, 12.x). The choice lives inside the agent
// body (`if constexpr`), never in a template argument, so the kernel's mangled name is the same in
// the host and device compilation passes. Override with -D_CUDAX_SORT_MWAY_MWAY_DOUBLE_BUFFER=0|1 for an
// A/B on any card. Measured 2026-09-22 (NOTES_K_IPT_L2.md §8.3-8.7): single buffer -8..-11 % keys
// / -9 % pairs on RTX 3090 / RTX 5090 (best tile vs best tile), -5 % on the L40S; double buffer
// +15..18 % faster than single on B200 / B300. A tile whose two buffers would not fit the 48 KiB
// static limit falls back to the single buffer regardless (agent_mway_merge_t::double_buffer).
// The rules are functions of the compute capability (as __CUDA_ARCH__ spells it: 890, 1000, 1200)
// so that the host, which has no __CUDA_ARCH__, can evaluate the *device's* choice at run time and
// size the kernel's dynamic shared memory to match (agent_mway_merge_t::temp_storage_bytes).
_CCCL_HOST_DEVICE_API constexpr bool mway_arch_double_buffer(int cc)
{
#  if defined(_CUDAX_SORT_MWAY_MWAY_DOUBLE_BUFFER)
  return (_CUDAX_SORT_MWAY_MWAY_DOUBLE_BUFFER) != 0 && cc >= 0;
#  else
  return cc >= 900 && cc < 1200;
#  endif
}
#  if defined(__CUDA_ARCH__)
inline constexpr int mway_compile_arch = __CUDA_ARCH__;
#  else
inline constexpr int mway_compile_arch = -1; // host pass: no layout decision is made here
#  endif
inline constexpr bool mway_double_buffer = mway_arch_double_buffer(mway_compile_arch);

// Bulk-copy (TMA) load of the K slices on sm_90+, two-buffer agent only: warp w issues one
// cp.async.bulk per slice w, w + 8, ... into a 16-byte-aligned staging layout in the second buffer
// (a misaligned head / tail of up to 15 bytes per slice is peeled by the warp's lanes), all warps
// arrive on one mbarrier and wait for the bytes, and round 0 of the merge tree reads the slices
// from staging while writing the first buffer. That replaces the per-thread block-strided load loop
// (33 of the 168 thread-instructions per key on the B200, NOTES_K_IPT_L2.md §8.8). Override with
// -D_CUDAX_SORT_MWAY_MWAY_BULK_LOAD=0|1; below sm_90 the same staging layout is filled with ordinary
// 16-byte vector loads so the path can be tested anywhere (forced with the two-buffer override).
_CCCL_HOST_DEVICE_API constexpr bool mway_arch_bulk_load(int cc)
{
#  if defined(_CUDAX_SORT_MWAY_MWAY_BULK_LOAD)
  return (_CUDAX_SORT_MWAY_MWAY_BULK_LOAD) != 0 && cc >= 0;
#  else
  return cc >= 900;
#  endif
}
inline constexpr bool mway_bulk_load = mway_arch_bulk_load(mway_compile_arch);

// Largest two-buffer layout the agent will use; above it the tile runs single-buffered. The merge
// kernels take their storage as dynamic shared memory and opt in above 48 KiB, so this is a tuning
// cap (blocks per SM), not the static limit. 96 KiB: 512 x 15 i32 tiles (63 KB, 3 blocks on a
// 228 KB part) and i64 keys at ipt 17 (70 KB) qualify. Override -D_CUDAX_SORT_MWAY_MWAY_TWO_BUFFER_MAX_KB.
#  if !defined(_CUDAX_SORT_MWAY_MWAY_TWO_BUFFER_MAX_KB)
#    define _CUDAX_SORT_MWAY_MWAY_TWO_BUFFER_MAX_KB 96
#  endif
inline constexpr ::cuda::std::size_t mway_two_buffer_max_bytes =
  ::cuda::std::size_t{_CUDAX_SORT_MWAY_MWAY_TWO_BUFFER_MAX_KB} * 1024;

// Blocks per SM the merge kernel's __launch_bounds__ asks for. On the 228 KB parts the two-buffer
// agent fits 7 blocks by shared memory (7 x 33 KB); asking ptxas for 7 caps it at 36 registers,
// which the bulk-load keys variant otherwise exceeds (40, one block fewer). 0 = no constraint
// elsewhere: it must be 0, not 1 -- an explicit 1 changes ptxas's allocation (sm_89 single-buffer
// keys 40 -> 46 registers, pairs 40 -> 48), 0 reproduces the omitted argument exactly.
#  if defined(_CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS)
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_VALUE (_CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS)
#  elif defined(__CUDA_ARCH__)
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_VALUE ((__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200) ? 7 : 0)
#  else
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_VALUE 0
#  endif
inline constexpr int mway_merge_min_blocks_per_sm = _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_VALUE;
#  if defined(_CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS)
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS_VALUE (_CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS)
#  elif defined(__CUDA_ARCH__)
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS_VALUE ((__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200) ? 6 : 0)
#  else
#    define _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS_VALUE 0
#  endif
inline constexpr int mway_merge_min_blocks_pairs_per_sm = _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_PAIRS_VALUE;
// Same, for the single-buffer agent (no bulk load); 0 = no constraint. In units of 256-thread blocks like the two above
// (8 = 32 registers, whatever the block size).
#  if defined(_CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_SINGLE)
inline constexpr int mway_merge_min_blocks_single_per_sm = _CUDAX_SORT_MWAY_MWAY_MIN_BLOCKS_SINGLE;
#  else
inline constexpr int mway_merge_min_blocks_single_per_sm = 0;
#  endif

//! Kernel-argument bundle: up to `mway_max_k_limit` input array pointers/lengths, passed by value
//! (small enough to be an ordinary kernel parameter -- 32 * (8 + 8) = 512 bytes for 64-bit
//! pointers/offsets). Keys only: `vptr` exists to satisfy the Input interface and is never called
//! when ValueT is CUB_NS_QUALIFIER::NullType.
template <typename KeyT, typename Offset>
struct mway_input
{
  const KeyT* ptrs[mway_max_k_limit];
  Offset counts[mway_max_k_limit];
  int k;

  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE const KeyT* ptr(int a) const
  {
    return ptrs[a];
  }
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE const CUB_NS_QUALIFIER::NullType* vptr(int) const
  {
    return nullptr;
  }
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE Offset count(int a) const
  {
    return counts[a];
  }
};

//! Tile-local index type carried through the merge tree for key-value pairs.
using mway_tile_index_t = ::cuda::std::uint16_t;

// ---------------------------------------------------------------------------------------------
// Key transforms: sort floating-point keys as same-width unsigned integers
// ---------------------------------------------------------------------------------------------
//
// A double compare is a dependent FP64 op; on parts with reduced FP64 throughput (GeForce: 1/64 rate;
// B300) the block sort's log2(4352) compares per key run 1.5-3x slower than i64's and the merge tree
// pays 10-20 % (bench/types/README.md, NOTES_K_IPT_L2.md §9.2). The radix-sort trick removes the
// FP compare from every kernel: map each key to an unsigned integer whose `<` agrees with the
// float's `<` (flip all bits of negatives, set the sign bit of non-negatives), sort the integers,
// map back at the final store. Applied only when the comparator is known to be plain less-than
// (mway_is_less). The mapping refines `<` into a total order: -0.0 sorts before +0.0 (they compare
// equal under `<`, so a stable sort would otherwise keep their input order), and NaNs sort to the
// two ends by sign bit instead of being unordered. Same-width integer kernels are reused, so no new
// merge-kernel instantiation is needed beyond the store-back variant.
template <typename T>
struct mway_identity_key_xf
{
  using in_t                     = T;
  using key_t                    = T;
  static constexpr bool identity = true;
  _CCCL_HOST_DEVICE_API static key_t to(in_t k)
  {
    return k;
  }
  _CCCL_HOST_DEVICE_API static in_t from(key_t k)
  {
    return k;
  }
  _CCCL_HOST_DEVICE_API static key_t from_bits(key_t k)
  {
    return k;
  }
};

//! Only for 8-byte floats: FP32 compares run at full rate on every part, so f32 gains nothing (measured
//! within 0.4 % on four cards), while FP64 is 1/64 rate on GeForce / Ada and cut on the B300.
template <typename F>
struct mway_float_key_xf
{
  static_assert(::cuda::std::is_floating_point_v<F> && sizeof(F) == 8, "double keys only");
  using in_t                     = F;
  using key_t                    = ::cuda::std::uint64_t;
  static constexpr bool identity = false;
  static constexpr key_t top     = key_t{1} << 63;
  _CCCL_HOST_DEVICE_API static key_t to(in_t f)
  {
    key_t u;
    memcpy(&u, &f, sizeof(F));
    return (u & top) ? static_cast<key_t>(~u) : static_cast<key_t>(u | top);
  }
  //! The original key's bit pattern, as key_t: the inverse map without the type change, so a tile
  //! can be mapped back in place and stored with the integer store into the (aliased) float array.
  _CCCL_HOST_DEVICE_API static key_t from_bits(key_t u)
  {
    return (u & top) ? static_cast<key_t>(u & ~top) : static_cast<key_t>(~u);
  }
  _CCCL_HOST_DEVICE_API static in_t from(key_t u)
  {
    u = from_bits(u);
    in_t f;
    memcpy(&f, &u, sizeof(F));
    return f;
  }
};

//! Whether the part's FP64 compare is slow enough for the integer image to pay: reduced-rate FP64
//! on cc 7.5, 8.6 / 8.7 / 8.9 (GeForce Turing / Ampere, Ada: 1/64 of FP32), 10.3 (B300) and 12.x
//! (consumer Blackwell); full rate on 7.0, 8.0, 9.0, 10.0 (V100, A100, H100, B200), where the
//! transform is a wash and costs its last-pass store and the cub 2-way last pass (B200 2026-09-28:
//! +0.7 % keys, +3 % pairs before the spill fix; NOTES_K_IPT_L2.md §9.11).
_CCCL_HOST_DEVICE_API constexpr bool mway_arch_float_as_int(int cc)
{
  return cc == 750 || (cc >= 860 && cc < 900) || cc == 1030 || cc >= 1200;
}

_CCCL_HOST_DEVICE_API constexpr int mway_log2_floor(int v)
{
  int l = 0;
  while ((2 << l) <= v)
  {
    ++l;
  }
  return l;
}

template <typename Policy, typename KeyT, typename Offset, typename ValueT, bool DoubleBuffer, bool BulkLoad, bool Local32>
struct mway_merge_storage
{
  static constexpr mway_policy policy    = Policy::value;
  static constexpr int threads_per_block = policy.threads_per_block;
  static constexpr int items_per_thread  = policy.items_per_thread;
  static constexpr int items_per_tile    = items_per_thread * threads_per_block;
  static constexpr bool has_values       = !::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  using block_store_keys =
    CUB_NS_QUALIFIER::BlockStore<KeyT, threads_per_block, items_per_thread, policy.store_algorithm>;
  using block_store_values =
    CUB_NS_QUALIFIER::BlockStore<ValueT, threads_per_block, items_per_thread, policy.store_algorithm>;
  using index_t = mway_tile_index_t;

  // Staging layout of the bulk-copy load: slice a starts head_pad_a bytes into a 16-byte-aligned
  // slot (head_pad = its source address mod 16, so the bulk copy is aligned on both sides) and the
  // slot is round_up(head_pad_a + bytes_a, 16). head_pad <= 16 - sizeof(KeyT) and the round-up adds
  // < 16, so a slot exceeds its slice by at most 31 - sizeof(KeyT) bytes; the trailing 16 are for
  // the one-past-the-end read of the last run's head. Independent of the actual run lengths. (Every
  // byte counts: at 256 x 15 / i32 the B200 fits 7 blocks per SM only below 32256 B -- shared
  // memory is allocated in 128 B granules.)
  static constexpr int bulk_align = 16;
  static constexpr int stage_bytes =
    items_per_tile * static_cast<int>(sizeof(KeyT))
    + (2 * bulk_align - 1 - static_cast<int>(sizeof(KeyT))) * mway_max_k_limit + bulk_align;

  // Index buffers only exist for pairs and second buffers only for the double-buffer agent; a
  // zero-length array is not legal C++, so size the absent ones 1.
  // One slack slot for the one-past-the-end refill.
  static constexpr int key_slack      = 1;
  static constexpr int index_buf_size = has_values ? items_per_tile + 1 : 1;
  static constexpr int key_buf_b_size =
    (BulkLoad && DoubleBuffer)
      ? (stage_bytes + static_cast<int>(sizeof(KeyT)) - 1) / static_cast<int>(sizeof(KeyT))
      : (DoubleBuffer ? items_per_tile + key_slack : 1);
  static constexpr int index_buf_b_size = (DoubleBuffer && has_values) ? items_per_tile + 1 : 1;

  struct buffers_t
  {
    // Single bulk load stages padded slices here; round one packs them in place.
    alignas((BulkLoad && !DoubleBuffer) ? bulk_align : alignof(KeyT))
      KeyT buf[(BulkLoad && !DoubleBuffer)
                 ? (stage_bytes + static_cast<int>(sizeof(KeyT)) - 1) / static_cast<int>(sizeof(KeyT))
                 : items_per_tile + key_slack];
    index_t idx[index_buf_size]; // pairs only: tile-local source index of the key at each position
    alignas(bulk_align) KeyT buf_b[key_buf_b_size]; // double buffer only: the other ping-pong buffer (and the bulk-load
                                                    // staging)
    index_t idx_b[index_buf_b_size]; // double buffer + pairs only
  };
  struct type
  {
    // The block-store scratch (items_per_tile x sizeof(T) for the warp-transpose store) aliases
    // the tile buffer: by the time the stores run the last round's output is in registers. The
    // union is sized by the largest member, so a value type wider than the key costs shared
    // memory only through the store scratch.
    union
    {
      buffers_t bufs;
      typename block_store_keys::TempStorage store_keys;
      typename block_store_values::TempStorage store_values;
    };
    // Per-tile run bookkeeping, ONE copy per block (broadcast shared-memory reads) rather than a
    // dynamically-indexed per-thread array (which the compiler must place in local memory: 32
    // Offsets x several arrays x 256 threads was more bytes of local-memory traffic per tile than
    // the tile itself -- the v1 kernel's dominant cost).
    ::cuda::std::conditional_t<Local32 && _CUDAX_SORT_MWAY_TILE_LOCAL_MODE == 2,
                               ::cuda::std::uint16_t,
                               ::cuda::std::conditional_t<Local32, int, Offset>>
      run_off[mway_max_k_limit + 1]; // tile-local prefix offsets of the K runs; run_off[k] = total
    const KeyT* run_src[mway_max_k_limit]; // input.ptr(a) + this tile's cut in run a: the slice's first key
    const ValueT* run_vsrc[has_values ? mway_max_k_limit : 1]; // pairs: the same for the values
    // Bulk-copy load only: where slice a starts in the staging buffer (entry k = end of the last
    // slice, so the empty second half of an odd pair has a readable head), and the mbarrier the
    // copies complete on.
    ::cuda::std::uint16_t stage_off[BulkLoad ? mway_max_k_limit + 1 : 1]; // element offsets into buf_b
    alignas(8)::cuda::std::uint64_t bulk_mbar;
  };
};

//! `Input` is any type exposing `int k`, `const KeyT* ptr(int a)`, `Offset count(int a)` and, for
//! pairs, `const ValueT* vptr(int a)` for `a < k` -- `mway_input` (explicit pointer table, the
//! standalone K-way merge) or `mway_uniform_runs` (arithmetic run descriptors, the K-way
//! mergesort's per-pass batched merge; see mway_sort.cuh). ValueT = CUB_NS_QUALIFIER::NullType selects the
//! keys-only agent.
//! `KeyXf`: key transform (mway_float_key_xf) whose `key_t` is KeyT; when `store_original` is set the
//! tile is stored as KeyXf::in_t (the last pass of a transformed sort writes the caller's type back).
template <typename Policy,
          typename KeyT,
          typename Offset,
          typename CompareOp,
          typename Input  = mway_input<KeyT, Offset>,
          typename ValueT = CUB_NS_QUALIFIER::NullType,
          typename KeyXf  = mway_identity_key_xf<KeyT>>
struct agent_mway_merge_t
{
  static_assert(::cuda::std::is_same_v<typename KeyXf::key_t, KeyT>,
                "the transform's integer type is the agent's key type");
  static constexpr mway_policy policy = Policy::value;

  static constexpr int threads_per_block = policy.threads_per_block;
  static constexpr int items_per_thread  = policy.items_per_thread;
  static constexpr int items_per_tile    = items_per_thread * threads_per_block;
  static constexpr bool has_values       = !::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;
  static_assert(items_per_tile <= 65535, "tile-local indices are 16-bit");

  template <bool DoubleBuffer, bool BulkLoad, bool Local32 = false>
  using storage_for = mway_merge_storage<Policy, KeyT, Offset, ValueT, DoubleBuffer, BulkLoad, Local32>;

  // Two buffers where the architecture wants them (mway_arch_double_buffer) AND the two-buffer
  // layout is under mway_two_buffer_max_bytes; otherwise this instantiation uses the single buffer.
  // Use the baseline layout to choose buffering, so an optimization cannot flip the algorithm.
  _CCCL_HOST_DEVICE_API static constexpr bool double_buffer_for(bool arch_double, bool arch_bulk)
  {
    return arch_double
        && (arch_bulk ? sizeof(typename storage_for<true, true>::type)
                      : sizeof(typename storage_for<true, false>::type))
             <= mway_two_buffer_max_bytes;
  }
  _CCCL_HOST_DEVICE_API static constexpr bool optimized_for(int cc, bool dbl)
  {
    return _CUDAX_SORT_MWAY_MWAY_INSTRUCTION_OPTIMIZATION != 0 && (cc == 1000 || (cc == 1200 && !dbl)) && !has_values
        && KeyXf::identity && ::cuda::std::is_same_v<KeyT, int> && threads_per_block == 512
        && policy.store_algorithm == CUB_NS_QUALIFIER::BLOCK_STORE_WARP_TRANSPOSE && mway_arch_bulk_load(cc)
        && (dbl ? items_per_thread == 17 : items_per_thread == 15);
  }
  _CCCL_HOST_DEVICE_API static constexpr bool single_bulk_for(int cc)
  {
    return optimized_for(cc, false);
  }
  _CCCL_HOST_DEVICE_API static constexpr bool local32_for(int cc, bool dbl)
  {
    // Broad narrowing regressed some architectures in the integration sweep.
    // Keep it selectable while defaulting to the previously measured scopes.
    return _CUDAX_SORT_MWAY_TILE_LOCAL_MODE != 0 || optimized_for(cc, dbl) || (cc == 1200 && !dbl);
  }
  static constexpr bool double_buffer         = double_buffer_for(mway_double_buffer, mway_bulk_load);
  static constexpr bool instruction_optimized = optimized_for(mway_compile_arch, double_buffer);
  static constexpr bool bulk_load             = mway_bulk_load && (double_buffer || single_bulk_for(mway_compile_arch));

  using storage_t      = storage_for<double_buffer, bulk_load, local32_for(mway_compile_arch, double_buffer)>;
  using temp_storage_t = typename storage_t::type;
  using TempStorage    = CUB_NS_QUALIFIER::Uninitialized<temp_storage_t>;
  static constexpr int bulk_align = storage_t::bulk_align;
  // Above the 48 KiB static limit the kernel takes its storage as dynamic shared memory (device
  // pass; the host applies the same threshold to temp_storage_bytes(cc), see mway_merge_dynamic_smem).
  static constexpr ::cuda::std::size_t static_smem_limit = 48 * 1024;
  static constexpr bool dynamic_smem                     = sizeof(temp_storage_t) > static_smem_limit;

  //! Bytes of dynamic shared memory the merge kernel needs on a device of compute capability `cc`
  //! (as __CUDA_ARCH__ spells it: 890, 1000, 1200). Evaluates the same rules the device pass used,
  //! so host and device agree on the layout. The launcher opts the kernel in above 48 KiB.
  _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::size_t temp_storage_bytes(int cc)
  {
    const bool arch_double = mway_arch_double_buffer(cc);
    const bool arch_bulk   = mway_arch_bulk_load(cc);
    const bool dbl         = double_buffer_for(arch_double, arch_bulk);
    const bool narrow      = local32_for(cc, dbl);
    const bool bulk        = arch_bulk && (dbl || single_bulk_for(cc));
    if (narrow)
    {
      return dbl ? (bulk ? sizeof(typename storage_for<true, true, true>::type)
                         : sizeof(typename storage_for<true, false, true>::type))
                 : (bulk ? sizeof(typename storage_for<false, true, true>::type)
                         : sizeof(typename storage_for<false, false, true>::type));
    }
    return dbl ? (bulk ? sizeof(typename storage_for<true, true, false>::type)
                       : sizeof(typename storage_for<true, false, false>::type))
               : sizeof(typename storage_for<false, false, false>::type);
  }

  // Blocks per SM asked of ptxas (see mway_merge_min_blocks_per_sm): 7 for keys, 6 for pairs on the
  // 228 KB parts (pairs spill 68 bytes per thread at the 7-block register cap of 36; 6 gives 40).
  // Only for the bulk-load two-buffer agent: a tile that fell back to the single buffer (i64 keys at
  // ipt 13-17) needs 53-62 registers and spilled under the cap (+10-15 % on the B200, job 4485667).
  // The 7 / 6 are for 256-thread blocks; scale to the block size so the register cap stays 32 / 40
  // (a 512-thread block at "7 blocks" would be asked for 18 registers and spill everything).
  static constexpr int min_blocks_per_sm =
    (!bulk_load || !double_buffer)
      ? mway_merge_min_blocks_single_per_sm * 256 / threads_per_block
      : (has_values ? mway_merge_min_blocks_pairs_per_sm : mway_merge_min_blocks_per_sm) * 256 / threads_per_block;

  using block_store_keys   = typename storage_t::block_store_keys;
  using block_store_values = typename storage_t::block_store_values;
  using index_t            = mway_tile_index_t;

  temp_storage_t& storage;
  Input input;
  KeyT* keys_out;
  ValueT* values_out; // pairs only (nullptr for keys)
  CompareOp compare_op;
  const Offset* boundaries; // (num_tiles + 1) * mway_max_k_limit, this tile's row at boundaries +
                            // tile_idx*mway_max_k_limit
  KeyT* samples_out; // sampled partition of the NEXT pass: every sample_stride-th output key of this group
                     // (group-relative: samples_out[(tile_base + pos) / sample_stride]); nullptr = none
  int sample_stride;
  bool store_original; // transformed sorts, last pass: store KeyXf::from(key) into (KeyXf::in_t*) keys_out

  //! Largest index `i` in [0, count) with off[i * stride] <= idx (off is non-decreasing).
  //! Binary search over <= 32 entries: 5 steps of broadcast-friendly shared-memory reads.
  template <typename LocalIndex, typename LocalOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static int find_run(LocalIndex idx, const LocalOffset* off, int count, int stride)
  {
    int lo = 0;
    int hi = count; // answer in [lo, hi)
    while (hi - lo > 1)
    {
      const int mid = lo + (hi - lo) / 2;
      if (off[mid * stride] <= idx)
      {
        lo = mid;
      }
      else
      {
        hi = mid;
      }
    }
    return lo;
  }

private:
  // Fixed-size serial merge; callers guarantee valid outputs or bound their stores.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void serial_full_window(
    const KeyT* first, const KeyT* second, int a, int b, int n1, int n2, KeyT (&output)[items_per_thread])
  {
    const int second_offset = static_cast<int>(second - first);
    int unused_indices[items_per_thread];
    CUB_NS_QUALIFIER::detail::serial_merge<true>(
      first, a, second_offset + b, n1 - a, n2 - b, output, unused_indices, compare_op, KeyT{});
  }

  // Global boundary cuts stay Offset. Two-way searches can use tile-local ints.
  template <bool TwoWay>
  using tile_offset_t =
    ::cuda::std::conditional_t<TwoWay || local32_for(mway_compile_arch, double_buffer), int, Offset>;

  static constexpr int pair_warps          = threads_per_block / 32;
  static constexpr bool power_of_two_warps = (pair_warps & (pair_warps - 1)) == 0;
  static constexpr int pair_capacity       = items_per_thread + 2;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE KeyT* staging_buffer()
  {
    return double_buffer ? storage.bufs.buf_b : storage.bufs.buf;
  }

  struct pair_window
  {
    int begin;
    int end;
  };

  _CCCL_DEVICE_API _CCCL_FORCEINLINE static pair_window split_pair(int count, int log_threads, int lane)
  {
    const int threads = 1 << log_threads;
    const int nominal = items_per_thread * threads;
    if (count <= nominal)
    {
      const int begin = (::cuda::std::min) (items_per_thread * lane, count);
      return {begin, (::cuda::std::min) (begin + items_per_thread, count)};
    }
    if (count <= nominal + threads)
    {
      // Assign surplus keys in twos, with at most one single extra key.
      const int remainder   = count - nominal;
      const int extra_pairs = remainder >> 1;
      const int odd         = remainder & 1;
      const int begin =
        items_per_thread * lane + 2 * (::cuda::std::min) (lane, extra_pairs) + (lane > extra_pairs ? odd : 0);
      return {begin, begin + items_per_thread + (lane < extra_pairs ? 2 : (lane == extra_pairs ? odd : 0))};
    }
    const int items     = count >> log_threads;
    const int remainder = count & (threads - 1);
    const int begin     = lane * items + (::cuda::std::min) (lane, remainder);
    return {begin, begin + items + (lane < remainder ? 1 : 0)};
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int galloping_cut(const KeyT* first, const KeyT* second, int n1, int n2, int diag)
  {
    const int guess =
      static_cast<int>(__fdividef(static_cast<float>(diag) * static_cast<float>(n1), static_cast<float>(n1 + n2)));
    bool far;
    return merge_path_gallop<16, 3>(first, second, n1, n2, diag, guess, compare_op, far);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void
  refill_head(bool take_second, const KeyT* first, const KeyT* second, int& i1, int& i2, KeyT& head1, KeyT& head2)
  {
    if constexpr (detail::serial_merge_ops::enabled<KeyT>)
    {
      detail::serial_merge_ops::refill(take_second, first, second, i1, i2, head1, head2);
    }
    else if (take_second)
    {
      head2 = second[++i2];
    }
    else
    {
      head1 = first[++i1];
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE TileOffset prepare_runs(Offset tile_idx, int k)
  {
    const int tid      = static_cast<int>(threadIdx.x);
    const Offset* row0 = boundaries + tile_idx * mway_max_k_limit;
    const Offset* row1 = row0 + mway_max_k_limit;
    auto* run_off      = storage.run_off;
    auto* run_src      = storage.run_src;
    auto* run_vsrc     = storage.run_vsrc;
    auto* stage_off    = storage.stage_off;

    // Warp 0: lane a reads run a's cut points and the warp prefix-sums the run lengths.
    if (tid < 32)
    {
      const bool owns       = tid < k;
      const Offset beg      = owns ? row0[tid] : Offset{0};
      const TileOffset len  = owns ? static_cast<TileOffset>(row1[tid] - beg) : TileOffset{0};
      const TileOffset excl = mway_warp_exclusive_scan_add(0xffffffffu, len);
      if (owns)
      {
        run_off[tid] = excl;
        run_src[tid] = input.ptr(tid) + beg;
        if constexpr (has_values)
        {
          run_vsrc[tid] = input.vptr(tid) + beg;
        }
        if (tid == k - 1)
        {
          run_off[k] = excl + len;
        }
      }
      if constexpr (bulk_load)
      {
        // Staging: 16-byte-aligned slots of round_up(head_pad + bytes, 16) per slice (see
        // stage_bytes), prefix-summed by the same warp; slice a is placed head_pad bytes into its
        // slot so that its first 16-byte-aligned source byte lands on an aligned destination.
        const auto src_addr = reinterpret_cast<::cuda::std::uintptr_t>(owns ? input.ptr(tid) + beg : nullptr);
        const int head_pad  = static_cast<int>(src_addr & (bulk_align - 1)); // == (16 - align_diff) % 16
        const int bytes     = static_cast<int>(len) * static_cast<int>(sizeof(KeyT));
        const int slot      = owns ? ((head_pad + bytes + bulk_align - 1) / bulk_align) * bulk_align : 0;
        const int slot_beg  = mway_warp_exclusive_scan_add(0xffffffffu, slot);
        if (owns)
        {
          // Offsets are in elements: a KeyT-aligned source address is congruent to a multiple of
          // sizeof(KeyT) mod 16, so head_pad is one too. Stored 16-bit (staging is < 64K elements)
          // and rebased on buf_b at use, so round 0 keeps 32-bit shared-memory addressing.
          stage_off[tid] = static_cast<::cuda::std::uint16_t>((slot_beg + head_pad) / static_cast<int>(sizeof(KeyT)));
          if (tid == k - 1)
          {
            stage_off[k] = static_cast<::cuda::std::uint16_t>((slot_beg + slot) / static_cast<int>(sizeof(KeyT)));
          }
        }
        if constexpr (mway_compile_arch >= 900)
        {
          if (tid == 0)
          {
            ::cuda::ptx::mbarrier_init(&storage.bulk_mbar, pair_warps);
            ::cuda::ptx::fence_mbarrier_init(::cuda::ptx::sem_release, ::cuda::ptx::scope_cluster);
          }
        }
      }
    }
    __syncthreads();
    return static_cast<TileOffset>(run_off[k]);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void prepare_pair_rounds(int k, int& pair_mask)
  {
    const int tid       = static_cast<int>(threadIdx.x);
    const auto* run_off = storage.run_off;
    // Each lane checks one round. The mask becomes visible after the load barrier.
    if (!double_buffer && tid < 32)
    {
      const int lgk = 31 - __clz(k);
      bool ok       = false;
      if (tid < lgk && k == (1 << lgk))
      {
        const int pairs    = k >> (tid + 1);
        ok                 = power_of_two_warps && pairs <= pair_warps;
        const Offset max_n = static_cast<Offset>(items_per_thread + 1) * 32 * (pair_warps / pairs);
        for (int q = 0; q < pairs; ++q)
        {
          ok = ok && (run_off[(q + 1) << (tid + 1)] - run_off[q << (tid + 1)] <= max_n);
        }
      }
      const unsigned m = __ballot_sync(0xffffffffu, ok);
      if (tid == 0)
      {
        pair_mask = static_cast<int>(m);
      }
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void load_bulk_slices(int k)
  {
    const int tid         = static_cast<int>(threadIdx.x);
    const auto* run_off   = storage.run_off;
    const auto* run_src   = storage.run_src;
    const auto* stage_off = storage.stage_off;
    KeyT* stage           = staging_buffer();

    // Each warp copies aligned slices and peels their scalar edges.
    constexpr int num_warps                         = threads_per_block / 32;
    const int warp                                  = tid >> 5;
    const int lane                                  = tid & 31;
    [[maybe_unused]] ::cuda::std::uint32_t tx_bytes = 0;
    for (int a = warp; a < k; a += num_warps)
    {
      const KeyT* src      = run_src[a];
      KeyT* dst            = stage + stage_off[a];
      const int len        = static_cast<int>(run_off[a + 1] - run_off[a]);
      const int bytes      = len * static_cast<int>(sizeof(KeyT));
      const auto src_addr  = reinterpret_cast<::cuda::std::uintptr_t>(src);
      const int align_diff = static_cast<int>((bulk_align - (src_addr & (bulk_align - 1))) & (bulk_align - 1));
      const int head_bytes = align_diff < bytes ? align_diff : bytes;
      const int bulk_bytes = (bytes - head_bytes) & ~(bulk_align - 1);
      const int head_elems = head_bytes / static_cast<int>(sizeof(KeyT));
      const int tail_beg   = (head_bytes + bulk_bytes) / static_cast<int>(sizeof(KeyT));
      if (lane < head_elems) // head: < 16 bytes, so < 16 elements
      {
        dst[lane] = src[lane];
      }
      else if (lane >= 16 && tail_beg + (lane - 16) < len) // tail: likewise < 16 elements
      {
        dst[tail_beg + (lane - 16)] = src[tail_beg + (lane - 16)];
      }
      if (bulk_bytes > 0)
      {
        char* bdst       = reinterpret_cast<char*>(dst) + head_bytes; // 16-byte aligned by construction
        const char* bsrc = reinterpret_cast<const char*>(src) + head_bytes;
        if constexpr (mway_compile_arch >= 900)
        {
          if (lane == 0)
          {
            ::cuda::ptx::cp_async_bulk(
              ::cuda::ptx::space_shared,
              ::cuda::ptx::space_global,
              bdst,
              bsrc,
              static_cast<::cuda::std::uint32_t>(bulk_bytes),
              &storage.bulk_mbar);
          }
          tx_bytes += static_cast<::cuda::std::uint32_t>(bulk_bytes);
        }
        else
        {
          for (int off = lane * bulk_align; off < bulk_bytes; off += 32 * bulk_align)
          {
            *reinterpret_cast<uint4*>(bdst + off) = *reinterpret_cast<const uint4*>(bsrc + off);
          }
        }
      }
    }
    if constexpr (mway_compile_arch >= 900)
    {
      if (lane == 0)
      {
        ::cuda::ptx::mbarrier_arrive_expect_tx(
          ::cuda::ptx::sem_release, ::cuda::ptx::scope_cta, ::cuda::ptx::space_shared, &storage.bulk_mbar, tx_bytes);
      }
      while (!::cuda::ptx::mbarrier_try_wait_parity(&storage.bulk_mbar, 0u))
      {
      }
    }
  }

  template <bool TwoWay, typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void load_blocked(TileOffset total)
  {
    const int tid       = static_cast<int>(threadIdx.x);
    const auto* run_off = storage.run_off;
    const auto* run_src = storage.run_src;
    KeyT* buf           = storage.bufs.buf;
    index_t* idx        = storage.bufs.idx;

    // Load the block-strided window into registers before writing to shared memory.
    int run = 0;
    KeyT staged[items_per_thread];
#  pragma unroll
    for (int item = 0; item < items_per_thread; ++item)
    {
      const TileOffset i = static_cast<TileOffset>(threads_per_block) * item + tid;
      if (i < total)
      {
        if constexpr (TwoWay)
        {
          run = i < run_off[1] ? 0 : 1;
        }
        else
        {
          while (i >= run_off[run + 1])
          {
            ++run;
          }
        }
        staged[item] = run_src[run][i - run_off[run]];
      }
    }
#  pragma unroll
    for (int item = 0; item < items_per_thread; ++item)
    {
      const TileOffset i = static_cast<TileOffset>(threads_per_block) * item + tid;
      if (i < total)
      {
        buf[i] = staged[item];
        if constexpr (has_values)
        {
          idx[i] = static_cast<index_t>(i);
        }
      }
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void merge_pair_warp_window(
    int stride,
    int round,
    int log_k,
    bool staged,
    const KeyT* buf_in,
    KeyT* buf_out,
    const index_t* idx_in,
    index_t* idx_out)
  {
    const auto* run_off       = storage.run_off;
    const auto* stage_off     = storage.stage_off;
    KeyT* stage               = staging_buffer();
    constexpr int log_warps   = mway_log2_floor(pair_warps);
    const int log_threads     = log_warps - (log_k - round - 1) + 5;
    const int pair            = static_cast<int>(threadIdx.x) >> log_threads;
    const int lane            = static_cast<int>(threadIdx.x) & ((1 << log_threads) - 1);
    const int a               = pair << (round + 1);
    const TileOffset beg1     = run_off[a];
    const TileOffset beg2     = run_off[a + stride];
    const TileOffset pair_end = run_off[a + 2 * stride];
    const auto window         = split_pair(static_cast<int>(pair_end - beg1), log_threads, lane);
    const KeyT* in1           = buf_in;
    const KeyT* in2           = buf_in;
    if constexpr (bulk_load)
    {
      if (staged)
      {
        in1 = stage + (static_cast<int>(stage_off[a]) - static_cast<int>(beg1));
        in2 = stage + (static_cast<int>(stage_off[a + stride]) - static_cast<int>(beg2));
      }
    }
    if (window.begin < window.end)
    {
      const int lo    = window.begin;
      const int n_out = window.end - lo;
      const int cut =
        galloping_cut(in1 + beg1, in2 + beg2, static_cast<int>(beg2 - beg1), static_cast<int>(pair_end - beg2), lo);
      int i1                = static_cast<int>(beg1) + cut;
      int i2                = static_cast<int>(beg2) + (lo - cut);
      const int end1        = static_cast<int>(beg2);
      const int end2        = static_cast<int>(pair_end);
      KeyT key1             = in1[i1];
      KeyT key2             = in2[i2];
      const TileOffset wpos = beg1 + lo;
      auto step             = [&](int item) {
        const bool take2     = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
        buf_out[wpos + item] = (take2 ? key2 : key1);
        if constexpr (has_values)
        {
          const int src        = take2 ? i2 : i1;
          idx_out[wpos + item] = staged ? static_cast<index_t>(src) : idx_in[src];
        }
        refill_head(take2, in1, in2, i1, i2, key1, key2);
      };
#  pragma unroll
      for (int item = 0; item < pair_capacity; ++item)
      {
        if (item < n_out)
        {
          step(item);
        }
      }
      for (int item = pair_capacity; item < n_out; ++item) // a pair over pair_capacity * threads keys: rolled tail,
                                                           // same merge state
      {
        step(item);
      }
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void merge_blocked_to_shared(
    int k,
    int stride,
    int num_pairs,
    TileOffset pos,
    TileOffset end,
    bool staged,
    const KeyT* buf_in,
    KeyT* buf_out,
    const index_t* idx_in,
    index_t* idx_out)
  {
    const auto* run_off   = storage.run_off;
    const auto* stage_off = storage.stage_off;
    const KeyT* stage     = staging_buffer();

    while (pos < end)
    {
      const int p               = find_run(pos, run_off, num_pairs, 2 * stride);
      const int a               = 2 * p * stride;
      const int a_end           = (::cuda::std::min) (a + stride, k);
      const int b_end           = (::cuda::std::min) (a + 2 * stride, k);
      const TileOffset beg1     = run_off[a];
      const TileOffset beg2     = run_off[a_end];
      const TileOffset pair_end = run_off[b_end];
      const int local_diag      = static_cast<int>(pos - beg1);
      const int n_out           = static_cast<int>((::cuda::std::min) (end, pair_end) - pos);
      const KeyT* in1           = buf_in;
      const KeyT* in2           = buf_in;
      if constexpr (bulk_load)
      {
        if (staged)
        {
          in1 = stage + (static_cast<int>(stage_off[a]) - static_cast<int>(beg1));
          in2 = stage + (static_cast<int>(stage_off[a_end]) - static_cast<int>(beg2));
        }
      }
      const int keys1_beg = merge_path(
        in1 + beg1,
        in2 + beg2,
        static_cast<int>(beg2 - beg1),
        static_cast<int>(pair_end - beg2),
        local_diag,
        compare_op);
      int i1         = static_cast<int>(beg1) + keys1_beg;
      int i2         = static_cast<int>(beg2) + (local_diag - keys1_beg);
      const int end1 = static_cast<int>(beg2);
      const int end2 = static_cast<int>(pair_end);
      // The tile has a slack slot for the one-past-end refill.
      KeyT key1 = in1[i1];
      KeyT key2 = in2[i2];
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        if (item < n_out)
        {
          const bool take2    = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
          buf_out[pos + item] = (take2 ? key2 : key1);
          if constexpr (has_values)
          {
            // Round 0 of the staged load has no index buffer yet: the index is the position.
            const int src       = take2 ? i2 : i1;
            idx_out[pos + item] = staged ? static_cast<index_t>(src) : idx_in[src];
          }
          refill_head(take2, in1, in2, i1, i2, key1, key2);
        }
      }
      pos += n_out;
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE bool try_store_bulk(
    Offset tile_base,
    TileOffset total,
    TileOffset base,
    TileOffset end,
    int k,
    KeyT* buf_in,
    KeyT* buf_out,
    const index_t* idx_in)
  {
    const int tid        = static_cast<int>(threadIdx.x);
    const auto* run_off  = storage.run_off;
    const auto* run_vsrc = storage.run_vsrc;
    ValueT val_local[items_per_thread];
    // The second buffer can hold gathered values if each value fits in a key slot.
    constexpr bool value_fits_buf_out = has_values && sizeof(ValueT) <= sizeof(KeyT);
    KeyT* dst                         = keys_out + tile_base;
    const auto dst_addr               = reinterpret_cast<::cuda::std::uintptr_t>(dst);
    if ((dst_addr & (bulk_align - 1)) != 0)
    {
      return false;
    }

    ValueT* vdst     = nullptr;
    bool values_bulk = false;
    if constexpr (value_fits_buf_out)
    {
      vdst        = values_out + tile_base;
      values_bulk = (reinterpret_cast<::cuda::std::uintptr_t>(vdst) & (bulk_align - 1)) == 0;
    }
    if constexpr (has_values)
    {
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const TileOffset pos = base + item;
        if (pos < end)
        {
          const TileOffset src = static_cast<TileOffset>(idx_in[pos]);
          const int run        = find_run(src, run_off, k, 1);
          const ValueT v       = ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_gather_value(
            run_vsrc[run] + (src - run_off[run]));
          if constexpr (value_fits_buf_out)
          {
            if (values_bulk && !mway_value_gather_batch)
            {
              reinterpret_cast<ValueT*>(buf_out)[pos] = v;
              continue; // staged in shared memory instead of a register this tile
            }
          }
          val_local[item] = v;
        }
      }
      if constexpr (mway_value_gather_batch && value_fits_buf_out)
      {
        if (values_bulk)
        {
#  pragma unroll
          for (int item = 0; item < items_per_thread; ++item)
          {
            const TileOffset pos = base + item;
            if (pos < end)
            {
              reinterpret_cast<ValueT*>(buf_out)[pos] = val_local[item];
            }
          }
        }
      }
    }
    if (samples_out != nullptr)
    {
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const TileOffset pos = base + item;
        if (pos < end)
        {
          const Offset g = tile_base + pos;
          if (g % static_cast<TileOffset>(sample_stride) == 0)
          {
            samples_out[g / static_cast<TileOffset>(sample_stride)] = buf_in[pos];
          }
        }
      }
    }
    if constexpr (!KeyXf::identity)
    {
      // Convert the final transformed keys before handing them to the bulk store.
      if (store_original)
      {
#  pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
          const TileOffset pos = base + item;
          if (pos < end)
          {
            buf_in[pos] = KeyXf::from_bits(buf_in[pos]);
          }
        }
      }
    }
    if constexpr (value_fits_buf_out)
    {
      // buf_out's new contents (the gathered values) need the same barrier the round loop
      // gave buf_in for free; buf_in needs one only if the transform above just wrote it.
      __syncthreads();
    }
    else if constexpr (!KeyXf::identity)
    {
      if (store_original)
      {
        __syncthreads();
      }
    }
    const int bytes      = static_cast<int>(total) * static_cast<int>(sizeof(KeyT));
    const int bulk_bytes = bytes & ~(bulk_align - 1);
    const int tail_beg   = bulk_bytes / static_cast<int>(sizeof(KeyT));
    if (tid < static_cast<int>(total) - tail_beg) // < 16 bytes of tail
    {
      dst[tail_beg + tid] = buf_in[tail_beg + tid];
    }
    int vbulk_bytes = 0, vtail_beg = 0;
    if constexpr (value_fits_buf_out)
    {
      if (values_bulk)
      {
        const int vbytes = static_cast<int>(total) * static_cast<int>(sizeof(ValueT));
        vbulk_bytes      = vbytes & ~(bulk_align - 1);
        vtail_beg        = vbulk_bytes / static_cast<int>(sizeof(ValueT));
        if (tid < static_cast<int>(total) - vtail_beg) // < 16 bytes of tail
        {
          vdst[vtail_beg + tid] = reinterpret_cast<ValueT*>(buf_out)[vtail_beg + tid];
        }
      }
    }
    if (bulk_bytes > 0 || vbulk_bytes > 0)
    {
      char* bdst        = reinterpret_cast<char*>(dst);
      const char* bsrc  = reinterpret_cast<const char*>(buf_in);
      char* vbdst       = value_fits_buf_out ? reinterpret_cast<char*>(vdst) : nullptr;
      const char* vbsrc = value_fits_buf_out ? reinterpret_cast<const char*>(buf_out) : nullptr;
      if constexpr (mway_compile_arch >= 900)
      {
        // The rounds wrote buf_in (and, for values, this block just wrote buf_out) through
        // the generic proxy: order those writes before the async-proxy read (every writer
        // fences, then the block synchronises), then one thread issues both copies and waits
        // until they have read their buffers before the block may exit or reuse them.
        ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
        __syncthreads();
        if (tid == 0)
        {
          if (bulk_bytes > 0)
          {
            ::cuda::ptx::cp_async_bulk(
              ::cuda::ptx::space_global,
              ::cuda::ptx::space_shared,
              bdst,
              bsrc,
              static_cast<::cuda::std::uint32_t>(bulk_bytes));
          }
          if (vbulk_bytes > 0)
          {
            ::cuda::ptx::cp_async_bulk(
              ::cuda::ptx::space_global,
              ::cuda::ptx::space_shared,
              vbdst,
              vbsrc,
              static_cast<::cuda::std::uint32_t>(vbulk_bytes));
          }
          ::cuda::ptx::cp_async_bulk_commit_group();
          ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
        }
      }
      else
      {
        for (int off = tid * bulk_align; off < bulk_bytes; off += threads_per_block * bulk_align)
        {
          *reinterpret_cast<uint4*>(bdst + off) = *reinterpret_cast<const uint4*>(bsrc + off);
        }
        for (int off = tid * bulk_align; off < vbulk_bytes; off += threads_per_block * bulk_align)
        {
          *reinterpret_cast<uint4*>(vbdst + off) = *reinterpret_cast<const uint4*>(vbsrc + off);
        }
      }
    }
    if constexpr (value_fits_buf_out)
    {
      if (values_bulk)
      {
        return true; // both keys and values are done
      }
    }
    if constexpr (has_values)
    {
      __syncthreads(); // the key copy has read the buffers; the value store's scratch aliases them
      block_store_values{storage.store_values}.Store(values_out + tile_base, val_local, static_cast<int>(total));
    }
    return true;
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void store_register_tile(
    Offset tile_base,
    TileOffset total,
    TileOffset base,
    TileOffset end,
    KeyT (&out_keys)[items_per_thread],
    ValueT (&val_local)[items_per_thread])
  {
    // Samples for the next pass's partition (NOTES_K_IPT_L2.md §9.13): one key per sample_stride,
    // written from the register window, so every segment's sample array is produced by the kernel
    // that produces the segment and no kernel re-reads the array to sample it.
    if (samples_out != nullptr)
    {
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const Offset pos = base + item;
        if (pos < end)
        {
          const Offset g = tile_base + pos;
          if (g % static_cast<Offset>(sample_stride) == 0)
          {
            samples_out[g / static_cast<Offset>(sample_stride)] = out_keys[item];
          }
        }
      }
    }

    // CUB_NS_QUALIFIER::BlockStore expects its `items` input in *blocked* order (thread tid owns
    // [tid*items_per_thread, (tid+1)*items_per_thread)), which is exactly the window layout.
    if constexpr (!KeyXf::identity)
    {
      // Last pass of a transformed sort: map back in place (same width) and use the one store below
      // into the aliased caller's array. A second BlockStore instantiation for the original type
      // spilled 16-56 bytes under the launch-bound register caps (B200, §9.11).
      if (store_original)
      {
#  pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
          out_keys[item] = KeyXf::from_bits(out_keys[item]);
        }
      }
    }
    // The optimized single route omits validity checks on full tiles; tails stay checked.
    if constexpr (!double_buffer && (instruction_optimized || mway_compile_arch == 1200))
    {
      if (total == items_per_tile)
      {
        block_store_keys{storage.store_keys}.Store(keys_out + tile_base, out_keys);
      }
      else
      {
        block_store_keys{storage.store_keys}.Store(keys_out + tile_base, out_keys, static_cast<int>(total));
      }
    }
    else
    {
      block_store_keys{storage.store_keys}.Store(keys_out + tile_base, out_keys, static_cast<int>(total));
    }
    if constexpr (has_values)
    {
      __syncthreads(); // store_keys and store_values alias
      if constexpr (!double_buffer && mway_compile_arch == 1200)
      {
        if (total == items_per_tile)
        {
          block_store_values{storage.store_values}.Store(values_out + tile_base, val_local);
        }
        else
        {
          block_store_values{storage.store_values}.Store(values_out + tile_base, val_local, static_cast<int>(total));
        }
      }
      else
      {
        block_store_values{storage.store_values}.Store(values_out + tile_base, val_local, static_cast<int>(total));
      }
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  merge_two_way_to_shared(TileOffset total, TileOffset base, bool from_stage, const KeyT* buf_in, KeyT* buf_out)
  {
    const auto* run_off   = storage.run_off;
    const auto* stage_off = storage.stage_off;
    const KeyT* stage     = staging_buffer();
    const int n1          = static_cast<int>(run_off[1]);
    const int n2          = static_cast<int>(total) - n1;
    const KeyT* first     = from_stage ? stage + static_cast<int>(stage_off[0]) : buf_in;
    const KeyT* second    = from_stage ? stage + static_cast<int>(stage_off[1]) : buf_in + n1;
    const int diag        = (::cuda::std::min) (static_cast<int>(base), static_cast<int>(total));
    const int a           = galloping_cut(first, second, n1, n2, diag);
    KeyT output[items_per_thread];
    serial_full_window(first, second, a, diag - a, n1, n2, output);
#  pragma unroll
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int pos = static_cast<int>(base) + item;
      if (pos < static_cast<int>(total))
      {
        buf_out[pos] = output[item];
      }
    }
    __syncthreads();
  }

  template <bool TwoWay, typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void merge_ping_pong(Offset tile_base, TileOffset total, bool loaded, int k)
  {
    const auto* run_off   = storage.run_off;
    const auto* run_vsrc  = storage.run_vsrc;
    const TileOffset base = static_cast<TileOffset>(items_per_thread) * static_cast<int>(threadIdx.x);
    const TileOffset end  = (::cuda::std::min) (base + static_cast<TileOffset>(items_per_thread), total);
    KeyT* buf             = storage.bufs.buf;
    KeyT* stage           = staging_buffer();
    index_t* idx          = storage.bufs.idx;

    // Round zero reads the padded slices when the input used the bulk loader.
    const bool from_stage = bulk_load && loaded;
    KeyT* buf_in          = from_stage ? stage : buf;
    KeyT* buf_out         = from_stage ? buf : stage;
    index_t* idx_in       = from_stage ? storage.bufs.idx_b : idx;
    index_t* idx_out      = from_stage ? idx : storage.bufs.idx_b;

    if constexpr (TwoWay && instruction_optimized && items_per_thread == 17)
    {
      merge_two_way_to_shared(total, base, from_stage, buf_in, buf_out);
      buf_in = buf_out;
      idx_in = idx_out;
    }
    else
    {
      const bool pair_path = power_of_two_warps && (k & (k - 1)) == 0 && k <= 2 * pair_warps;
      const int log_k      = 31 - __clz(k);
      for (int stride = 1, round = 0; stride < k; stride *= 2, ++round)
      {
        const int live      = (k + stride - 1) >> round;
        const int num_pairs = (live + 1) >> 1;
        const bool staged   = from_stage && stride == 1;
        if (pair_path)
        {
          merge_pair_warp_window<TileOffset>(stride, round, log_k, staged, buf_in, buf_out, idx_in, idx_out);
        }
        else
        {
          merge_blocked_to_shared(k, stride, num_pairs, base, end, staged, buf_in, buf_out, idx_in, idx_out);
        }

        __syncthreads(); // all writes complete before swapping the buffers
        KeyT* keys       = buf_in;
        buf_in           = buf_out;
        buf_out          = keys;
        index_t* indices = idx_in;
        idx_in           = idx_out;
        idx_out          = indices;
      }
    }

    if (try_store_bulk(tile_base, total, base, end, k, buf_in, buf_out, idx_in))
    {
      return;
    }

    KeyT out_keys[items_per_thread];
    ValueT val_local[items_per_thread];
// The destination is misaligned. Gather values directly into the blocked register window.
#  pragma unroll
    for (int item = 0; item < items_per_thread; ++item)
    {
      const TileOffset pos = base + item;
      if (pos < end)
      {
        out_keys[item] = buf_in[pos];
        if constexpr (has_values)
        {
          const TileOffset src = static_cast<TileOffset>(idx_in[pos]);
          const int run        = find_run(src, run_off, k, 1);
          val_local[item]      = ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_gather_value(
            run_vsrc[run] + (src - run_off[run]));
        }
      }
    }
    __syncthreads(); // the buffers alias the store scratch
    store_register_tile(tile_base, total, base, end, out_keys, val_local);
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void stage_pair_window(
    int s,
    int r,
    int log_k,
    bool loaded,
    int& pw_pos,
    int& pw_cnt,
    KeyT (&staged_keys)[pair_capacity],
    index_t (&staged_idx)[has_values ? pair_capacity : 1])
  {
    const int tid         = static_cast<int>(threadIdx.x);
    const auto* run_off   = storage.run_off;
    const auto* stage_off = storage.stage_off;
    const KeyT* stage     = staging_buffer();
    const KeyT* buf       = storage.bufs.buf;
    const index_t* idx    = storage.bufs.idx;

    constexpr int lgw         = mway_log2_floor(threads_per_block / 32);
    const int lg_nthr         = lgw - (log_k - r - 1) + 5;
    const int n_thr           = 1 << lg_nthr;
    const int p               = tid >> lg_nthr;
    const int lt              = tid & (n_thr - 1);
    const int a               = p << (r + 1);
    const TileOffset beg1     = run_off[a];
    const TileOffset beg2     = run_off[a + s];
    const TileOffset pair_end = run_off[a + 2 * s];
    const bool from_stage     = bulk_load && loaded && s == 1;
    const KeyT* in1           = from_stage ? stage + stage_off[a] - beg1 : buf;
    const KeyT* in2           = from_stage ? stage + stage_off[a + s] - beg2 : buf;
    const int n_pair          = static_cast<int>(pair_end - beg1);
    const auto window         = split_pair(n_pair, lg_nthr, lt);
    const int lo              = window.begin;
    pw_cnt                    = window.end - lo;
    pw_pos                    = static_cast<int>(beg1) + lo;
    if (pw_cnt > 0)
    {
      const int keys1_beg =
        galloping_cut(in1 + beg1, in2 + beg2, static_cast<int>(beg2 - beg1), static_cast<int>(pair_end - beg2), lo);
      int i1         = static_cast<int>(beg1) + keys1_beg;
      int i2         = static_cast<int>(beg2) + (lo - keys1_beg);
      const int end1 = static_cast<int>(beg2);
      const int end2 = static_cast<int>(pair_end);
      KeyT key1      = in1[i1];
      KeyT key2      = in2[i2];
#  pragma unroll
      for (int item = 0; item < pair_capacity; ++item)
      {
        if (item < pw_cnt)
        {
          const bool take2  = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
          staged_keys[item] = (take2 ? key2 : key1);
          if constexpr (has_values)
          {
            staged_idx[item] = idx[take2 ? i2 : i1];
          }
          refill_head(take2, in1, in2, i1, i2, key1, key2);
        }
      }
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void stage_blocked_window(
    int k,
    int s,
    int num_pairs,
    bool loaded,
    TileOffset base,
    TileOffset end,
    KeyT (&staged_keys)[pair_capacity],
    index_t (&staged_idx)[has_values ? pair_capacity : 1])
  {
    const auto* run_off   = storage.run_off;
    const auto* stage_off = storage.stage_off;
    const KeyT* stage     = staging_buffer();
    const KeyT* buf       = storage.bufs.buf;
    const index_t* idx    = storage.bufs.idx;

    TileOffset pos_next = base;
    while (pos_next < end)
    {
      const int p               = find_run(pos_next, run_off, num_pairs, 2 * s);
      const int a               = 2 * p * s; // first original run of pair p
      const int a_end           = (::cuda::std::min) (a + s, k); // one past its last original run
      const int b_end           = (::cuda::std::min) (a + 2 * s, k);
      const TileOffset beg1     = run_off[a];
      const TileOffset beg2     = run_off[a_end]; // == pair_end when the pair has no second half
      const TileOffset pair_end = run_off[b_end];
      const bool from_stage     = bulk_load && loaded && s == 1;
      const KeyT* in1           = from_stage ? stage + stage_off[a] - beg1 : buf;
      const KeyT* in2           = from_stage ? stage + stage_off[a_end] - beg2 : buf;
      const TileOffset pos_stop = (::cuda::std::min) (end, pair_end);

      const int local_diag = static_cast<int>(pos_next - beg1);
      const int keys1_beg  = merge_path(
        in1 + beg1,
        in2 + beg2,
        static_cast<int>(beg2 - beg1),
        static_cast<int>(pair_end - beg2),
        local_diag,
        compare_op);
      int i1         = static_cast<int>(beg1) + keys1_beg;
      int i2         = static_cast<int>(beg2) + (local_diag - keys1_beg);
      const int end1 = static_cast<int>(beg2);
      const int end2 = static_cast<int>(pair_end);
      // The tile has a slack slot for the one-past-end refill.
      KeyT key1 = in1[i1];
      KeyT key2 = in2[i2];

#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const TileOffset pos = base + item;
        if (pos >= pos_next && pos < pos_stop)
        {
          const bool take2  = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
          staged_keys[item] = (take2 ? key2 : key1);
          if constexpr (has_values)
          {
            staged_idx[item] = idx[take2 ? i2 : i1];
          }
          refill_head(take2, in1, in2, i1, i2, key1, key2);
        }
      }
      pos_next = pos_stop;
    }
  }

  template <typename TileOffset>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  merge_single_buffer(Offset tile_base, TileOffset total, bool loaded, int k, int pair_mask)
  {
    const TileOffset base = static_cast<TileOffset>(items_per_thread) * static_cast<int>(threadIdx.x);
    const TileOffset end  = (::cuda::std::min) (base + static_cast<TileOffset>(items_per_thread), total);
    KeyT* buf             = storage.bufs.buf;
    index_t* idx          = storage.bufs.idx;
    const auto* run_off   = storage.run_off;
    const auto* run_vsrc  = storage.run_vsrc;
    KeyT staged_keys[pair_capacity];
    index_t staged_idx[has_values ? pair_capacity : 1];
    KeyT out_keys[items_per_thread];
    index_t out_idx[has_values ? items_per_thread : 1];
    ValueT val_local[items_per_thread];

    // Stage each round in registers before overwriting the shared input.
    bool in_registers = false;
    const int log_k   = 31 - __clz(k);
    for (int s = 1, r = 0; s < k; s *= 2, ++r)
    {
      const int live      = (k + s - 1) >> r;
      const int num_pairs = (live + 1) >> 1;
      const bool last     = 2 * s >= k;
      const bool pw_done  = ((pair_mask >> r) & 1) && !last;
      int pw_pos          = 0;
      int pw_cnt          = 0;
      if (pw_done)
      {
        stage_pair_window<TileOffset>(s, r, log_k, loaded, pw_pos, pw_cnt, staged_keys, staged_idx);
      }
      else
      {
        stage_blocked_window(k, s, num_pairs, loaded, base, end, staged_keys, staged_idx);
      }

      __syncthreads(); // every thread has finished reading this round's input
      if (last)
      {
        in_registers = true;
        break; // the tile is complete in registers; the store below reads it from there
      }
      if (pw_done)
      {
#  pragma unroll
        for (int item = 0; item < pair_capacity; ++item)
        {
          if (item < pw_cnt)
          {
            buf[pw_pos + item] = staged_keys[item];
            if constexpr (has_values)
            {
              idx[pw_pos + item] = staged_idx[item];
            }
          }
        }
      }
      else
      {
#  pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
          const TileOffset pos = base + item;
          if (pos < end)
          {
            buf[pos] = staged_keys[item];
            if constexpr (has_values)
            {
              idx[pos] = staged_idx[item];
            }
          }
        }
      }
      __syncthreads(); // written back: next round may read
    }

    if (!in_registers)
    {
// k == 1: nothing to merge, the loaded tile is the result. Read it out in blocked order.
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const TileOffset pos = base + item;
        if (pos < end)
        {
          staged_keys[item] = buf[pos];
          if constexpr (has_values)
          {
            staged_idx[item] = idx[pos];
          }
        }
      }
      __syncthreads(); // the buffer aliases the store scratch
    }
#  pragma unroll
    for (int item = 0; item < items_per_thread; ++item)
    {
      out_keys[item] = staged_keys[item];
      if constexpr (has_values)
      {
        out_idx[item] = staged_idx[item];
      }
    }

    // Pairs, single buffer: gather the values from global memory through the tile-local source
    // indices held in registers: which run that position belonged to, and where in it. The tile's
    // values are read exactly once in total, in a permuted order that stays within one tile.
    if constexpr (has_values)
    {
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const Offset pos = base + item;
        if (pos < end)
        {
          const Offset src = static_cast<Offset>(out_idx[item]);
          const int run    = find_run(src, run_off, k, 1);
          val_local[item]  = ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_gather_value(
            run_vsrc[run] + (src - run_off[run]));
        }
      }
    }
    store_register_tile(tile_base, total, base, end, out_keys, val_local);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void consume_two_way_single_buffer(Offset tile_idx, Offset tile_base)
  {
    const int tid      = static_cast<int>(threadIdx.x);
    const Offset* row0 = boundaries + tile_idx * mway_max_k_limit;
    const Offset* row1 = row0 + mway_max_k_limit;
    // Dedicated K2 keys path using the same single-buffer load/merge/store pattern as
    // CUB AgentMerge. Global cuts remain Offset; all positions inside this tile are int.
    using load_to_shared        = CUB_NS_QUALIFIER::detail::BlockLoadToShared<threads_per_block>;
    constexpr int minimum_align = CUB_NS_QUALIFIER::detail::LoadToSharedBufferAlignBytes<char>();
    constexpr int buffer_bytes  = CUB_NS_QUALIFIER::detail::LoadToSharedBufferSizeBytes<KeyT>(items_per_tile + 1ULL)
                                + (alignof(KeyT) < minimum_align ? 2 * minimum_align : 0);
    struct alignas(CUB_NS_QUALIFIER::detail::LoadToSharedBufferAlignBytes<KeyT>()) k2_storage_t
    {
      union
      {
        char keys[buffer_bytes];
        typename block_store_keys::TempStorage store_keys;
      };
      typename load_to_shared::TempStorage load;
    };
    __shared__ k2_storage_t work;
    const Offset cut1 = row0[0];
    const Offset cut2 = row0[1];
    const int n1      = static_cast<int>(row1[0] - cut1);
    const int n2      = static_cast<int>(row1[1] - cut2);
    const int count   = n1 + n2;
    load_to_shared loader{work.load};
    ::cuda::std::span<char> buffers{work.keys};
    auto first_buffer  = buffers.first(CUB_NS_QUALIFIER::detail::LoadToSharedBufferSizeBytes<KeyT>(n1));
    auto second_buffer = buffers.last(CUB_NS_QUALIFIER::detail::LoadToSharedBufferSizeBytes<KeyT>(n2));
    auto first         = loader.CopyAsync(
      first_buffer, ::cuda::std::span<const KeyT>{input.ptr(0) + cut1, static_cast<::cuda::std::size_t>(n1)});
    auto second = loader.CopyAsync(
      second_buffer, ::cuda::std::span<const KeyT>{input.ptr(1) + cut2, static_cast<::cuda::std::size_t>(n2)});
    auto token              = loader.Commit();
    const int second_offset = static_cast<int>(second.data() - first.data());
    loader.Wait(::cuda::std::move(token));
    const int base = items_per_thread * tid;
    const int diag = (::cuda::std::min) (base, count);
    const int a    = CUB_NS_QUALIFIER::MergePath(first.data(), second.data(), n1, n2, diag, compare_op);
    const int b    = diag - a;
    KeyT output[items_per_thread];
    int unused_indices[items_per_thread];
    CUB_NS_QUALIFIER::detail::serial_merge<true>(
      first.data(), a, second_offset + b, n1 - a, n2 - b, output, unused_indices, compare_op, KeyT{});
    if (samples_out != nullptr)
    {
#  pragma unroll
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int pos = base + item;
        if (pos < count)
        {
          const Offset global = tile_base + static_cast<Offset>(pos);
          if (global % static_cast<Offset>(sample_stride) == 0)
          {
            samples_out[global / static_cast<Offset>(sample_stride)] = output[item];
          }
        }
      }
    }
    __syncthreads(); // readers finish before the single buffer becomes BlockStore scratch
    if (count == items_per_tile)
    {
      block_store_keys{work.store_keys}.Store(keys_out + tile_base, output);
    }
    else
    {
      block_store_keys{work.store_keys}.Store(keys_out + tile_base, output, count);
    }
  }

  template <bool TwoWay>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void consume_tile_impl(Offset tile_idx, Offset tile_base)
  {
    using TileOffset       = tile_offset_t<TwoWay>;
    const int k            = TwoWay ? 2 : input.k;
    const TileOffset total = prepare_runs<TileOffset>(tile_idx, k);
    __shared__ int pair_mask;
    if constexpr (!double_buffer)
    {
      prepare_pair_rounds(k, pair_mask);
    }

    const bool loaded = bulk_load && k > 1;
    if (loaded)
    {
      load_bulk_slices(k);
    }
    else
    {
      load_blocked<TwoWay>(total);
    }
    __syncthreads();

    if constexpr (double_buffer)
    {
      merge_ping_pong<TwoWay>(tile_base, total, loaded, k);
    }
    else
    {
      merge_single_buffer(tile_base, total, loaded, k, pair_mask);
    }
  }

public:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void consume_tile(Offset tile_idx, Offset tile_base)
  {
    consume_tile_impl<false>(tile_idx, tile_base);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void consume_two_way_tile(Offset tile_idx, Offset tile_base)
  {
    if constexpr (!double_buffer && instruction_optimized)
    {
      consume_two_way_single_buffer(tile_idx, tile_base);
    }
    else
    {
      consume_tile_impl<true>(tile_idx, tile_base);
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void operator()()
  {
    const Offset tile_idx  = blockIdx.x;
    const Offset tile_base = tile_idx * items_per_tile;
    consume_tile(tile_idx, tile_base);
  }
};
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_AGENT_H
