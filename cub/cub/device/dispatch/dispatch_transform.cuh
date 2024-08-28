// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_CCCL_CUDA_COMPILER) && _CCCL_CUDACC_VER < 1105000
_CCCL_NV_DIAG_SUPPRESS(186)
#  include <cuda_pipeline_primitives.h>
// we cannot re-enable the warning here, because it is triggered outside the translation unit
// see also: https://godbolt.org/z/1x8b4hn3G
#endif // defined(_CCCL_CUDA_COMPILER) && _CCCL_CUDACC_VER < 1105000

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/device/device_for.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

CUB_NAMESPACE_BEGIN

// the ublkcp kernel needs PTX features that are only available and understood by CTK 12 and later
#if _CCCL_CUDACC_VER_MAJOR >= 12
#  define _CUB_HAS_TRANSFORM_UBLKCP
#endif // _CCCL_CUDACC_VER_MAJOR >= 12

namespace detail
{
namespace transform
{
_CCCL_HOST_DEVICE constexpr int sum()
{
  return 0;
}

// TODO(bgruber): remove with C++17
template <typename... Ts>
_CCCL_HOST_DEVICE constexpr int sum(int head, Ts... tail)
{
  return head + sum(tail...);
}

#if _CCCL_STD_VER >= 2017
template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(value_t<Its>)} + ... + 0);
}
#else // ^^^ C++17 ^^^ / vvv C++11 vvv
template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return sum(int{sizeof(value_t<Its>)}...);
}
#endif // _CCCL_STD_VER >= 2017

enum class Algorithm
{
  fallback_for,
  prefetch,
  unrolled_staged,
  memcpy_async,
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  ublkcp,
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

// TODO(bgruber): only needed so we can instantiate the kernel generically for any policy. Remove when fallback_for is
// dropped.
template <typename, typename Offset, typename F, typename RandomAccessIteratorOut, typename... RandomAccessIteratorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::fallback_for>,
  Offset,
  int,
  F,
  RandomAccessIteratorOut,
  RandomAccessIteratorIn...)
{}

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int ITEMS_PER_THREAD_NO_INPUT = 2; // items per thread when no input streams exist (just filling)
  static constexpr int MIN_ITEMS_PER_THREAD      = 1;
  static constexpr int MAX_ITEMS_PER_THREAD      = 32;
};

// Prefetches (at least on Hopper) a 128 byte cache line. Prefetching out-of-bounds addresses has no side effects
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch(const T* addr)
{
  assert(__isGlobal(addr));
  // TODO(bgruber): prefetch to L1 may be even better
  asm volatile("prefetch.global.L2 [%0];" : : "l"(addr) : "memory");
}

// TODO(bgruber): there is also the cp.async.bulk.prefetch instruction available on Hopper. May improve perf a tiny bit
// as we need to create less instructions to prefetch the same amount of data.

// overload for any iterator that is not a pointer, do nothing
template <typename It, ::cuda::std::__enable_if_t<!::cuda::std::is_pointer<It>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch(It)
{}

// this kernel guarantees stable addresses for the parameters of the user provided function
template <typename PrefetchPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteratorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  RandomAccessIteratorIn... ins)
{
  constexpr int block_dim = PrefetchPolicy::BLOCK_THREADS;
  {
    const int tile_stride = block_dim * num_elem_per_thread;
    const Offset offset   = static_cast<Offset>(blockIdx.x) * tile_stride;

    // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
    assert(offset < num_items);
    num_items -= offset;
    int dummy[] = {(ins += offset, 0)..., 0};
    (void) &dummy;
    out += offset;
  }

  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    const int idx = j * block_dim + threadIdx.x;
    // TODO(bgruber): replace by fold over comma in C++17
    int dummy[] = {(prefetch(ins + idx), 0)..., 0}; // extra zero to handle empty packs
    (void) &dummy; // nvcc 11.1 needs extra strong unused warning suppression
    (void) &idx; // nvcc 11.1 needs extra strong unused warning suppression
  }

  // ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost
  // TODO(bgruber): A6000 disagrees
#pragma unroll 1
  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    const int idx = j * block_dim + threadIdx.x;
    if (idx < num_items)
    {
      // we have to unwrap Thrust's proxy references here for backward compatibility (try zip_iterator.cu test)
      out[idx] = f(THRUST_NS_QUALIFIER::raw_reference_cast(ins[idx])...);
    }
  }
}

template <int BlockThreads, int ItemsPerThread>
struct unrolled_policy_t
{
  static constexpr int BLOCK_THREADS    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
};

// ahendriksen: no __restrict__ should be necessary on the input pointers since we already separated the load stage from
// the store stage.
template <typename UnrolledPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteratorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::unrolled_staged>,
  Offset num_items,
  int,
  F f,
  RandomAccessIteratorOut out,
  RandomAccessIteratorIn... ins)
{
  constexpr int block_dim        = UnrolledPolicy::BLOCK_THREADS;
  constexpr int items_per_thread = UnrolledPolicy::ITEMS_PER_THREAD;

  {
    constexpr int tile_stride = block_dim * items_per_thread;
    const Offset offset       = static_cast<Offset>(blockIdx.x) * tile_stride;

    // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
    num_items -= offset;
    int dummy[] = {(ins += offset, 0)..., 0};
    (void) dummy;
    out += offset;
  }

  // TODO(bgruber): we could use load vectorization here

  [&](cuda::std::array<value_t<RandomAccessIteratorIn>, items_per_thread>&&... arrays) {
  // load items_per_thread elements
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const int idx = j * block_dim + threadIdx.x;
      if (idx < num_items)
      {
        // TODO(bgruber): replace by fold over comma in C++17
        int dummy[] = {(arrays[j] = ins[idx], 0)..., 0}; // extra zero to handle empty packs
        (void) &dummy[0]; // MSVC needs extra strong unused warning supression
      }
    }
    // process items_per_thread elements
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const int idx = j * block_dim + threadIdx.x;
      if (idx < num_items)
      {
        out[idx] = f(arrays[j]...);
      }
    }
  }(cuda::std::array<value_t<RandomAccessIteratorIn>, items_per_thread>{}...);
}

// used for both, memcpy_async and ublkcp kernels
template <int BlockThreads>
struct async_copy_policy_t
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int MIN_ITEMS_PER_THREAD = 1;
  static constexpr int MAX_ITEMS_PER_THREAD = 32;
};

// TODO(bgruber) cheap copy of ::cuda::std::apply, which requires C++17.
template <class F, class Tuple, std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE auto poor_apply_impl(F&& f, Tuple&& t, ::cuda::std::index_sequence<Is...>)
  -> decltype(::cuda::std::forward<F>(f)(::cuda::std::get<Is>(::cuda::std::forward<Tuple>(t))...))
{
  return ::cuda::std::forward<F>(f)(::cuda::std::get<Is>(::cuda::std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
_CCCL_DEVICE _CCCL_FORCEINLINE auto poor_apply(F&& f, Tuple&& t)
  -> decltype(poor_apply_impl(
    ::cuda::std::forward<F>(f),
    ::cuda::std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{}))
{
  return poor_apply_impl(
    ::cuda::std::forward<F>(f),
    ::cuda::std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{});
}

// mult must be a power of 2
template <typename Integral>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto round_up_to_po2_multiple(Integral x, Integral mult) -> Integral
{
  assert(::cuda::std::has_single_bit(static_cast<::cuda::std::__make_unsigned_t<Integral>>(mult)));
  return (x + mult - 1) & ~(mult - 1);
}

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE const T* round_down_ptr(const T* ptr, unsigned alignment)
{
  assert(::cuda::std::has_single_bit(alignment));
  return reinterpret_cast<const T*>(
    reinterpret_cast<::cuda::std::uintptr_t>(ptr) & ~::cuda::std::uintptr_t{alignment - 1});
}

// Implementation notes on memcpy_async and UBLKCP kernels regarding copy alignment and padding
//
// For performance considerations of memcpy_async:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidance-for-memcpy-async
//
// We basically have to align the base pointer to 16 bytes, and copy a multiple of 16 bytes. To achieve this, when we
// copy a tile of data from an input buffer, we round down the pointer to the start of the tile to the next lower
// address that is a multiple of 16 bytes. This introduces head padding. We also round up the total number of bytes to
// copy (including head padding) to a multiple of 16 bytes, which introduces tail padding. For the bulk copy kernel, we
// have to align to 128 bytes instead of 16.
//
// However, padding memory copies like that may access the input buffer out-of-bounds. Here are some thoughts:
// * According to the CUDA programming guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses), "any address of a variable
// residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is
// always aligned to at least 256 bytes."
// * Memory protection is usually done on memory page level, which is even larger than 256 bytes for CUDA and 4KiB on
// Intel x86 and 4KiB+ ARM. Front and tail padding thus never leaves the memory page of the input buffer.
// * This should count for device memory, but also for device accessible memory living on the host.
// * The base pointer alignment and size rounding also never leaves the size of a cache line.
//
// Copying larger data blocks with head and tail padding should thus be legal. Nevertheless, an out-of-bounds read is
// still technically undefined behavior in C++. Also, compute-sanitizer flags at least such reads after the end of a
// buffer. Therefore, we lean on the safer side and protect against out of bounds reads at the beginning and end.

// A note on size and alignment: The size of a type is at least as large as its alignment. We rely on this fact in some
// conditions.
// This is guaranteed by the C++ standard, and follows from the definition of arrays: the difference between neighboring
// array element addresses is sizeof element type and each array element needs to fulfill the alignment requirement of
// the element type.

// Pointer with metadata to describe input memory for memcpy_async and UBLKCP kernels.
// cg::memcpy_async is most efficient when the data is 16-byte aligned and the size a multiple of 16 bytes
// UBLKCP is most efficient when the data is 128-byte aligned and the size a multiple of 16 bytes
template <typename T> // Cannot add alignment to signature, because we need a uniform kernel template instantiation
struct aligned_base_ptr
{
  T* ptr; // aligned pointer before the original pointer (16-byte or 128-byte, or higher if alignof(T) is larger)
  T* end; // address of one-past-last element. used to avoid out-of-bounds accesses at the end
  int head_padding; // byte offset between ptr and the original pointer. Value inside [0;15] or [0;127].

  _CCCL_HOST_DEVICE friend bool operator==(const aligned_base_ptr& a, const aligned_base_ptr& b)
  {
    return a.ptr == b.ptr && a.end == b.end && a.head_padding == b.head_padding;
  }
};

template <typename T, typename Offset>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr(const T* ptr, Offset size, int alignment) -> aligned_base_ptr<const T>
{
  const T* base_ptr = round_down_ptr(ptr, alignment);
  return aligned_base_ptr<const T>{base_ptr, ptr + size, static_cast<int>((ptr - base_ptr) * sizeof(T))};
}

constexpr int memcpy_async_alignment     = 16;
constexpr int memcpy_async_size_multiple = 16;

// Our own version of ::cuda::aligned_size_t, since we cannot include <cuda/barrier> on CUDA_ARCH < 700
template <_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t
{
  _CUDA_VSTD::size_t value; // TODO(bgruber): can this be an int?

  _CCCL_HOST_DEVICE constexpr operator size_t() const
  {
    return value;
  }
};

template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void fallback_copy(const char* src, char* dst, int num_bytes)
{
  const int elements = num_bytes / sizeof(T);
  if (threadIdx.x < elements)
  {
    reinterpret_cast<T*>(dst)[threadIdx.x] = reinterpret_cast<const T*>(src)[threadIdx.x];
  }
}

// TODO(bgruber): inline this as lambda in C++14
template <typename Offset, typename T>
_CCCL_DEVICE const T* copy_and_return_smem_dst(
  cooperative_groups::thread_block& group,
  int tile_size,
  char* smem,
  int& smem_offset,
  Offset global_offset,
  aligned_base_ptr<const T> aligned_ptr)
{
  // because SMEM base pointer and count are always multiples of 16-byte, we only need to align the SMEM start for types
  // with larger alignment
  _CCCL_IF_CONSTEXPR (alignof(T) > memcpy_async_alignment)
  {
    smem_offset = round_up_to_po2_multiple(smem_offset, static_cast<int>(alignof(T)));
  }
  const char* src                     = reinterpret_cast<const char*>(aligned_ptr.ptr + global_offset);
  char* dst                           = smem + smem_offset;
  const char* const dst_start_of_data = dst + aligned_ptr.head_padding;
  assert(reinterpret_cast<uintptr_t>(src) % memcpy_async_alignment == 0);
  assert(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0);
  assert(reinterpret_cast<uintptr_t>(dst) % memcpy_async_alignment == 0);
  assert(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0);
  assert(reinterpret_cast<uintptr_t>(dst_start_of_data) % alignof(T) == 0);

  const int padded_num_bytes         = aligned_ptr.head_padding + static_cast<int>(sizeof(T)) * tile_size;
  const int padded_num_bytes_rounded = round_up_to_po2_multiple(padded_num_bytes, memcpy_async_size_multiple);

  smem_offset += padded_num_bytes_rounded; // leave aligned address for follow-up copy

  // TODO(bgruber): In case we need to shrink the padded region (front or back) I would really like to just call
  // cooperative_groups::details::copy_like<int4>(group, dst, src, sizeof(T) * tile_size), which already handles all the
  // peeling we do here

  int bytes_to_copy = padded_num_bytes_rounded;
  _CCCL_IF_CONSTEXPR (alignof(T) < memcpy_async_alignment)
  {
    // out-of-bounds access at the front can happen, so check for it
    if (blockIdx.x == 0 && aligned_ptr.head_padding > 0)
    {
      // peel front bytes
      const int peeled_bytes_front = memcpy_async_alignment - aligned_ptr.head_padding;
      assert(peeled_bytes_front < memcpy_async_alignment);
      fallback_copy<T>(src + aligned_ptr.head_padding, dst + aligned_ptr.head_padding, peeled_bytes_front);
      // move async copy start into the buffer
      src += memcpy_async_alignment;
      dst += memcpy_async_alignment;
      bytes_to_copy -= memcpy_async_alignment; // may result in a negative value
    }
  }

  _CCCL_IF_CONSTEXPR (alignof(T) < memcpy_async_size_multiple)
  {
    // out-of-bounds access at the back can happen, so check for it
    const bool oob = reinterpret_cast<uintptr_t>(src) + bytes_to_copy > reinterpret_cast<uintptr_t>(aligned_ptr.end);
    const auto async_copy_bytes = bytes_to_copy - oob * memcpy_async_size_multiple;
    if (async_copy_bytes > 0)
    {
      cooperative_groups::memcpy_async(
        group, dst, src, aligned_size_t<memcpy_async_size_multiple>{static_cast<::cuda::std::size_t>(async_copy_bytes)});
    }
    if (oob)
    {
      dst += async_copy_bytes;
      src += async_copy_bytes;
      fallback_copy<T>(src, dst, padded_num_bytes % memcpy_async_size_multiple);
    }
  }
  else
  {
    if (bytes_to_copy > 0)
    {
      cooperative_groups::memcpy_async(
        group, dst, src, aligned_size_t<memcpy_async_size_multiple>{static_cast<::cuda::std::size_t>(bytes_to_copy)});
    }
  }

  return reinterpret_cast<const T*>(dst_start_of_data);
}

template <typename MemcpyAsyncPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::memcpy_async>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  aligned_base_ptr<const InTs>... aligned_ptrs)
{
  extern __shared__ char smem[]; // this should be __attribute((aligned(memcpy_async_alignment))), but then it clashes
                                 // with the ublkcp kernel, which sets a higher alignment, since they are both called
                                 // from the same kernel entry point (albeit one is always discarded). However, SMEM is
                                 // 16-byte aligned by default.

  constexpr int block_dim  = MemcpyAsyncPolicy::BLOCK_THREADS;
  const Offset tile_stride = block_dim * num_elem_per_thread;
  const Offset offset      = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size      = static_cast<int>(::cuda::std::min(num_items - offset, tile_stride));

  auto group           = cooperative_groups::this_thread_block();
  int smem_offset      = 0;
  const auto smem_ptrs = ::cuda::std::tuple<const InTs*...>{
    copy_and_return_smem_dst(group, tile_size, smem, smem_offset, offset, aligned_ptrs)...};
  cooperative_groups::wait(group);
  (void) smem_ptrs; // suppress unused warning for MSVC
  (void) &smem_offset; // MSVC needs extra strong unused warning supression

  // TODO(bgruber): shouldn't this be before the loading?

  // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
  out += offset;

#pragma unroll 1
  for (int i = 0; i < num_elem_per_thread; ++i)
  {
    const int idx = i * block_dim + threadIdx.x;
    if (idx < tile_size)
    {
      out[idx] = poor_apply(
        [&](const InTs*... smem_base_ptrs) {
          return f(smem_base_ptrs[idx]...);
        },
        smem_ptrs);
    }
  }
}

constexpr int bulk_copy_alignment     = 128;
constexpr int bulk_copy_size_multiple = 16;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
// TODO(bgruber): inline this as lambda in C++14
template <typename Offset, typename T>
_CCCL_DEVICE void bulk_copy_tile(
  ::cuda::std::uint64_t& bar,
  int tile_size,
  int tile_stride,
  char* smem,
  int& smem_offset,
  ::cuda::std::uint32_t& total_bytes_bulk_copied,
  Offset global_offset,
  const aligned_base_ptr<const T>& aligned_ptr)
{
  // TODO(bgruber): I think we need to align smem + smem_offset to alignof(T) here

  const char* src = reinterpret_cast<const char*>(aligned_ptr.ptr + global_offset);
  char* dst       = smem + smem_offset;
  assert(reinterpret_cast<uintptr_t>(src) % bulk_copy_alignment == 0);
  assert(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0);
  assert(reinterpret_cast<uintptr_t>(dst) % bulk_copy_alignment == 0);
  assert(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0);

  const int padded_num_bytes         = aligned_ptr.head_padding + static_cast<int>(sizeof(T)) * tile_size;
  const int padded_num_bytes_rounded = round_up_to_po2_multiple(padded_num_bytes, bulk_copy_size_multiple);

  int bytes_to_copy = padded_num_bytes_rounded;
  _CCCL_IF_CONSTEXPR (alignof(T) < bulk_copy_alignment)
  {
    // out-of-bounds access at the front can happen, so check for it
    if (blockIdx.x == 0 && aligned_ptr.head_padding > 0)
    {
      // peel front bytes
      const int peeled_bytes_front = bulk_copy_alignment - aligned_ptr.head_padding;
      assert(peeled_bytes_front < bulk_copy_alignment);
      fallback_copy<T>(src + aligned_ptr.head_padding, dst + aligned_ptr.head_padding, peeled_bytes_front);
      // move bulk copy start into the buffer
      src += bulk_copy_alignment;
      dst += bulk_copy_alignment;
      bytes_to_copy -= bulk_copy_alignment; // may result in a negative value
    }
  }

  _CCCL_IF_CONSTEXPR (alignof(T) < bulk_copy_size_multiple)
  {
    // check for out-of-bounds read at the end
    // TODO(bgruber): oob has a compile-time known part. Let's look at codegen and see if we want to have two code paths
    const bool oob = reinterpret_cast<uintptr_t>(src) + bytes_to_copy > reinterpret_cast<uintptr_t>(aligned_ptr.end);
    const auto bulk_copy_bytes = bytes_to_copy - oob * bulk_copy_size_multiple;
    if (bulk_copy_bytes > 0)
    {
      if (threadIdx.x == 0)
      {
#  if CUB_PTX_ARCH >= 900
        ::cuda::ptx::cp_async_bulk(
          ::cuda::ptx::space_cluster, ::cuda::ptx::space_global, dst, src, bulk_copy_bytes, &bar);
#  endif // CUB_PTX_ARCH >= 900
        total_bytes_bulk_copied += bulk_copy_bytes;
      }
    }
    if (oob)
    {
      dst += bulk_copy_bytes;
      src += bulk_copy_bytes;
      fallback_copy<T>(src, dst, padded_num_bytes % bulk_copy_size_multiple);
    }
  }
  else
  {
    if (bytes_to_copy > 0)
    {
      if (threadIdx.x == 0)
      {
#  if CUB_PTX_ARCH >= 900
        ::cuda::ptx::cp_async_bulk(::cuda::ptx::space_cluster, ::cuda::ptx::space_global, dst, src, bytes_to_copy, &bar);
#  endif // CUB_PTX_ARCH >= 900
      }
    }
  }

  // TODO(bgruber): I don't think this is correct
  smem_offset += static_cast<int>(sizeof(T)) * tile_stride + ::cuda::std::max(bulk_copy_alignment, int{alignof(T)});
};

// TODO(bgruber): inline this as lambda in C++14
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE const T&
fetch_operand(int tile_stride, const char* smem, int& smem_offset, int smem_idx, const aligned_base_ptr<T>& aligned_ptr)
{
  const T* smem_operand_tile_base = reinterpret_cast<const T*>(smem + smem_offset + aligned_ptr.head_padding);
  smem_offset += int{sizeof(T)} * tile_stride + bulk_copy_alignment;
  _CCCL_IF_CONSTEXPR (alignof(T) > bulk_copy_alignment)
  {
    smem_offset = round_up_to_po2_multiple(smem_offset, static_cast<int>(alignof(T)));
  }
  return smem_operand_tile_base[smem_idx];
};

_CCCL_DEVICE _CCCL_FORCEINLINE bool select_one()
{
#  if CUB_PTX_ARCH >= 900
  const ::cuda::std::uint32_t membermask = ~0;
  ::cuda::std::uint32_t is_elected;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
    "elect.sync _|P_OUT, %1;\n\t"
    "selp.b32 %0, 1, 0, P_OUT; \n"
    "}"
    : "=r"(is_elected)
    : "r"(membermask)
    :);
  return threadIdx.x < 32 && static_cast<bool>(is_elected);
#  else // CUB_PTX_ARCH >= 900
  return false;
#  endif // CUB_PTX_ARCH >= 900
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  aligned_base_ptr<const InTs>... aligned_ptrs)
{
#  if CUB_PTX_ARCH >= 900
  __shared__ uint64_t bar;
  extern __shared__ char __attribute((aligned(bulk_copy_alignment))) smem[];

  namespace ptx = ::cuda::ptx;

  constexpr int block_dim = BulkCopyPolicy::BLOCK_THREADS;
  const int tile_stride   = block_dim * num_elem_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_stride;

  // TODO(bgruber) use: `cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&]() {` with CTK
  // >= 12.1
  if (true)
  {
    if (threadIdx.x == 0)
    {
      ptx::mbarrier_init(&bar, 1);
      ptx::fence_proxy_async(ptx::space_shared); // TODO(bgruber): is this correct?
    }

    const int tile_size                = ::cuda::std::min(num_items - offset, Offset{tile_stride});
    int smem_offset                    = 0;
    ::cuda::std::uint32_t total_copied = 0;

#    if _CCCL_STD_VER >= 2017
    // Order of evaluation is always left-to-right here. So smem_offset is updated in the right order.
    (..., bulk_copy_tile(bar, tile_size, tile_stride, smem, smem_offset, total_copied, offset, aligned_ptrs));
#    else // _CCCL_STD_VER >= 2017
    // Order of evaluation is also left-to-right
    int dummy[] = {
      (bulk_copy_tile(bar, tile_size, tile_stride, smem, smem_offset, total_copied, offset, aligned_ptrs), 0)..., 0};
    (void) dummy;
#    endif // _CCCL_STD_VER >= 2017

    if (threadIdx.x == 0)
    {
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
    }
  }
  __syncthreads();

  while (!ptx::mbarrier_try_wait_parity(&bar, 0))
  {
  }

  {
    // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
    num_items -= offset;
    out += offset;
  }

  // Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark)
#    pragma unroll 1
  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    const int idx = j * block_dim + threadIdx.x;
    if (idx < num_items)
    {
      int smem_offset = 0;
      out[idx]        = f(fetch_operand(tile_stride, smem, smem_offset, idx, aligned_ptrs)...);
    }
  }
#  endif // CUB_PTX_ARCH >= 900
}
#endif // _CUB_HAS_TRANSFORM_UBLKCP

template <typename It>
union kernel_arg
{
  char dummy; // in case It is not default-constructible
  aligned_base_ptr<const value_t<It>> aligned_ptr;
  It iterator;

  _CCCL_HOST_DEVICE kernel_arg() {} // in case It is not default-constructible
};

template <typename It>
_CCCL_HOST_DEVICE auto make_iterator_kernel_arg(It it) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.iterator = it;
  return arg;
}

template <typename It, typename Offset>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr_kernel_arg(It ptr, Offset size, int alignment) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.aligned_ptr = make_aligned_base_ptr(ptr, size, alignment);
  return arg;
}

// TODO(bgruber): make a variable template in C++14
template <Algorithm Alg>
using needs_aligned_ptr_t =
  ::cuda::std::bool_constant<Alg == Algorithm::memcpy_async
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                             || Alg == Algorithm::ublkcp
#endif // _CUB_HAS_TRANSFORM_UBLKCP
                             >;

template <Algorithm Alg, typename It, ::cuda::std::__enable_if_t<needs_aligned_ptr_t<Alg>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE auto select_kernel_arg(
  ::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg) -> aligned_base_ptr<const value_t<It>>&&
{
  return ::cuda::std::move(arg.aligned_ptr);
}

template <Algorithm Alg, typename It, ::cuda::std::__enable_if_t<!needs_aligned_ptr_t<Alg>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
select_kernel_arg(::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg) -> It&&
{
  return ::cuda::std::move(arg.iterator);
}

// There is only one kernel for all algorithms, that dispatches based on the selected policy. It must be instantiated
// with the same arguments for each algorithm. Only the device compiler will then select the implementation. This
// saves some compile-time and binary size.
template <typename MaxPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteartorsIn>
__launch_bounds__(MaxPolicy::ActivePolicy::algo_policy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void transform_kernel(
    Offset num_items,
    int num_elem_per_thread,
    F f,
    RandomAccessIteratorOut out,
    kernel_arg<RandomAccessIteartorsIn>... ins)
{
  constexpr auto alg = ::cuda::std::integral_constant<Algorithm, MaxPolicy::ActivePolicy::algorithm>{};
  transform_kernel_impl<typename MaxPolicy::ActivePolicy::algo_policy>(
    alg,
    num_items,
    num_elem_per_thread,
    ::cuda::std::move(f),
    ::cuda::std::move(out),
    select_kernel_arg(alg, ::cuda::std::move(ins))...);
}

constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  // TODO(bgruber): use if-else in C++14 for better readability
  return sm_arch >= 900 ? 48 * 1024 // 32 for H100, 48 for H200
       : sm_arch >= 800 ? 16 * 1024 // A100
                        : 12 * 1024; // V100 and below
}

constexpr int
items_per_thread_from_occupancy(int block_dim, int max_block_per_sm, int min_bif, int loaded_bytes_per_iter)
{
#if _CCCL_STD_VER >= 2014
  return ::cuda::ceil_div(min_bif, max_block_per_sm * block_dim * loaded_bytes_per_iter);
#else
  return (min_bif + (max_block_per_sm * block_dim * loaded_bytes_per_iter) - 1)
       / (max_block_per_sm * block_dim * loaded_bytes_per_iter);
#endif
}

// TODO(bgruber): need a constexpr version of this function
template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE constexpr auto memcpy_async_smem_for_tile_size(int tile_size) -> int
{
  int smem_size   = 0;
  auto count_smem = [&](int size, int alignment) {
    smem_size = round_up_to_po2_multiple(smem_size, alignment);
    // max aligned_base_ptr head_padding + max padding after == 16
    smem_size += size * tile_size + ::cuda::std::max(memcpy_async_alignment, memcpy_async_size_multiple);
  };
  // TODO(bgruber): replace by fold over comma in C++17 (left to right evaluation!)
  int dummy[] = {
    (count_smem(sizeof(value_t<RandomAccessIteratorsIn>), alignof(value_t<RandomAccessIteratorsIn>)), 0)..., 0};
  (void) &dummy; // need to take the address to suppress unused warnings more strongly for nvcc 11.1
  (void) &count_smem;
  return smem_size;
}

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE constexpr auto bulk_copy_smem_for_tile_size(int tile_size) -> int
{
  // max(128, alignof(value_type)) bytes of padding for each input tile (before + after)
  // TODO(bgruber): use a fold expression in C++17
  return tile_size * loaded_bytes_per_iteration<RandomAccessIteratorsIn...>()
       + sum(::cuda::std::max(::cuda::std::max(bulk_copy_alignment, bulk_copy_size_multiple),
                              int{alignof(value_t<RandomAccessIteratorsIn>)})...);
}

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>>
{
  static constexpr int loaded_bytes_per_iter =
    ::cuda::std::max(1, loaded_bytes_per_iteration<RandomAccessIteratorsIn...>());

  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_contiguous =
    ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_contiguous_iterator<RandomAccessIteratorsIn>...>::value;
  static constexpr bool all_values_trivially_reloc =
    ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_trivially_relocatable<value_t<RandomAccessIteratorsIn>>...>::value;

  static constexpr bool can_memcpy = all_contiguous && all_values_trivially_reloc;

  // TODO(bgruber): consider a separate kernel for just filling

  // below A100
  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(300);
    // TODO(bgruber): we don't need algo, because we can just detect the type of algo_policy
    static constexpr auto algorithm =
      RequiresStableAddress || no_input_streams ? Algorithm::prefetch : Algorithm::unrolled_staged;
    using algo_policy =
      ::cuda::std::_If<RequiresStableAddress || no_input_streams,
                       prefetch_policy_t<256>,
                       unrolled_policy_t<256, items_per_thread_from_occupancy(256, 8, min_bif, loaded_bytes_per_iter)>>;
  };

  // TODO(bgruber): should we add a tuning for 750? They should have items_per_thread_from_occupancy(256, 4, ...)

  // A100
  struct policy800 : ChainedPolicy<800, policy800, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(800);
    using async_policy           = async_copy_policy_t<256>;
    static constexpr bool exhaust_smem =
      memcpy_async_smem_for_tile_size<RandomAccessIteratorsIn...>(
        async_policy::BLOCK_THREADS * async_policy::MIN_ITEMS_PER_THREAD)
      > 48 * 1024;
    static constexpr bool use_fallback = RequiresStableAddress || !can_memcpy || no_input_streams || exhaust_smem;
    static constexpr auto algorithm    = use_fallback ? Algorithm::prefetch : Algorithm::memcpy_async;
    using algo_policy                  = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, async_policy>;
  };

  // A6000
  struct policy860 : ChainedPolicy<860, policy860, policy800>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(860);
    static constexpr auto algorithm =
      (RequiresStableAddress || !can_memcpy || no_input_streams) ? Algorithm::prefetch : Algorithm::unrolled_staged;
    // default: gives 6 items fir 2 I8 per iteration, 3 items for 2 I16, 2 items for 2 F32, 1 item for F64/I128
    using unrolled_policy =
      unrolled_policy_t<256, items_per_thread_from_occupancy(256, 6, min_bif, loaded_bytes_per_iter)>;

    // TODO(bgruber): I improved the memcpy_async kernel, need to reevaluate whether to use unrolled or memcpy_async
    using algo_policy =
      ::cuda::std::_If<RequiresStableAddress || !can_memcpy || no_input_streams, prefetch_policy_t<256>, unrolled_policy>;
    // best BabelStream unroll tunings (threads, items per thread, score)
    // * I8:   256,  4, 1.043765
    // * I16:  128,  5, 1.005624
    // * F32:  256,  7, 1.004884
    // * F64:  128,  5, 1.003517
    // * I128: 256, 10, 1.031597
    // TODO(bgruber): data suggests we need more items per thread for A6000
  };

  // H100 and H200
  struct policy900 : ChainedPolicy<900, policy900, policy860>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(900);
    using async_policy           = async_copy_policy_t<256>;
    static constexpr bool exhaust_smem =
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
      bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(
#else // _CUB_HAS_TRANSFORM_UBLKCP
      memcpy_async_smem_for_tile_size<RandomAccessIteratorsIn...>(
#endif // _CUB_HAS_TRANSFORM_UBLKCP
        async_policy::BLOCK_THREADS * async_policy::MIN_ITEMS_PER_THREAD)
      > 48 * 1024;
    static constexpr bool use_fallback = RequiresStableAddress || !can_memcpy || no_input_streams || exhaust_smem;
    static constexpr auto algorithm    = use_fallback ? Algorithm::prefetch :
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                                                   Algorithm::ublkcp;
#else // _CUB_HAS_TRANSFORM_UBLKCP
                                                   Algorithm::memcpy_async;
#endif // _CUB_HAS_TRANSFORM_UBLKCP
    using algo_policy = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, async_policy>;
  };

  using max_policy = policy900;
};

// TODO(bgruber): replace by ::cuda::std::expected in C++14
template <typename T>
struct PoorExpected
{
  alignas(T) char storage[sizeof(T)];
  cudaError_t error;

  _CCCL_HOST_DEVICE PoorExpected(T value)
      : error(cudaSuccess)
  {
    new (storage) T(::cuda::std::move(value));
  }

  _CCCL_HOST_DEVICE PoorExpected(cudaError_t error)
      : error(error)
  {}

  _CCCL_HOST_DEVICE explicit operator bool() const
  {
    return error == cudaSuccess;
  }

  _CCCL_HOST_DEVICE T& operator*()
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")
    return reinterpret_cast<T&>(storage);
    _CCCL_DIAG_POP
  }

  _CCCL_HOST_DEVICE const T& operator*() const
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")
    return reinterpret_cast<const T&>(storage);
    _CCCL_DIAG_POP
  }

  _CCCL_HOST_DEVICE T* operator->()
  {
    return &**this;
  }

  _CCCL_HOST_DEVICE const T* operator->() const
  {
    return &**this;
  }
};

_CCCL_HOST_DEVICE inline PoorExpected<int> get_max_shared_memory()
{
  //  gevtushenko promised me that I can assume that the stream passed to the CUB API entry point (where the kernels
  //  will later be launched on) belongs to the currently active device. So we can just query the active device here.
  int device = 0;
  auto error = CubDebug(cudaGetDevice(&device));
  if (error != cudaSuccess)
  {
    return error;
  }

  int max_smem = 0;
  error        = CubDebug(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
  if (error != cudaSuccess)
  {
    return error;
  }

  return max_smem;
}

template <bool RequiresStableAddress,
          typename Offset,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub = policy_hub<RequiresStableAddress, RandomAccessIteratorTupleIn>>
struct dispatch_t;

template <bool RequiresStableAddress,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct dispatch_t<RequiresStableAddress,
                  Offset,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut,
                  TransformOp,
                  PolicyHub>
{
  static_assert(::cuda::std::is_same<Offset, ::cuda::std::int32_t>::value
                  || ::cuda::std::is_same<Offset, ::cuda::std::int64_t>::value,
                "cub::DeviceTransform is only tested and tuned for 32-bit or 64-bit signed offset types");

  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  Offset num_items;
  TransformOp op;
  cudaStream_t stream;

#define KERNEL_PTR                                  \
  &transform_kernel<typename PolicyHub::max_policy, \
                    Offset,                         \
                    TransformOp,                    \
                    RandomAccessIteratorOut,        \
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>

  static constexpr int loaded_bytes_per_iter = loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();

  struct elem_counts
  {
    int elem_per_thread;
    int tile_size;
    int smem_size;
  };

  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running on?
  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto configure_memcpy_async_kernel()
    -> PoorExpected<
      ::cuda::std::tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron, decltype(KERNEL_PTR), int>>
  {
    // Benchmarking shows that even for a few iteration, this loop takes around 4-7 us, so should not be a concern.
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;
    static_assert(block_dim % memcpy_async_alignment == 0,
                  "BLOCK_THREADS needs to be a multiple of memcpy_async_alignment (16)"); // then tile_size is a
                                                                                          // multiple of 16-byte
    auto determine_element_counts = [&]() -> PoorExpected<elem_counts> {
      const auto max_smem = get_max_shared_memory();
      if (!max_smem)
      {
        return max_smem.error;
      }

      elem_counts last_counts{};
      // Increase the number of output elements per thread until we reach the required bytes in flight.
      static_assert(policy_t::MIN_ITEMS_PER_THREAD <= policy_t::MAX_ITEMS_PER_THREAD, ""); // ensures the loop below
                                                                                           // runs at least once
      for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread <= +policy_t::MAX_ITEMS_PER_THREAD;
           ++elem_per_thread)
      {
        const auto tile_size = block_dim * elem_per_thread;
        const int smem_size  = memcpy_async_smem_for_tile_size<RandomAccessIteratorsIn...>(tile_size);
        if (smem_size > *max_smem)
        {
          // assert should be prevented by smem check in policy
          assert(last_counts.elem_per_thread > 0 && "MIN_ITEMS_PER_THREAD exceeds available shared memory");
          return last_counts;
        }

        if (tile_size >= num_items)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        int max_occupancy = 0;
        const auto error  = CubDebug(MaxSmOccupancy(max_occupancy, KERNEL_PTR, block_dim, smem_size));
        if (error != cudaSuccess)
        {
          return error;
        }

        const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
        if (bytes_in_flight_SM >= ActivePolicy::min_bif)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        last_counts = elem_counts{elem_per_thread, tile_size, smem_size};
      }
      return last_counts;
    };
    // this static variable exists for each template instantiation of the surrounding function and class, on which the
    // chosen element count solely depends (assuming max SMEM is constant during a program execution)
    static auto config = determine_element_counts();
    if (!config)
    {
      return config.error;
    }
    assert(config->elem_per_thread > 0);
    assert(config->tile_size > 0);
    assert(config->tile_size % memcpy_async_alignment == 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (config->smem_size != 0)); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{config->tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, config->smem_size, stream),
      KERNEL_PTR,
      config->elem_per_thread);
  }

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running
  // on?
  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto configure_ublkcp_kernel()
    -> PoorExpected<
      ::cuda::std::tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron, decltype(KERNEL_PTR), int>>
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;
    static_assert(block_dim % bulk_copy_alignment == 0,
                  "BLOCK_THREADS needs to be a multiple of bulk_copy_alignment (128)"); // then tile_size is a multiple
                                                                                        // of 128-byte

    auto determine_element_counts = [&]() -> PoorExpected<elem_counts> {
      const auto max_smem = get_max_shared_memory();
      if (!max_smem)
      {
        return max_smem.error;
      }

      elem_counts last_counts{};
      // Increase the number of output elements per thread until we reach the required bytes in flight.
      static_assert(policy_t::MIN_ITEMS_PER_THREAD <= policy_t::MAX_ITEMS_PER_THREAD, ""); // ensures the loop below
      // runs at least once
      for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread < +policy_t::MAX_ITEMS_PER_THREAD;
           ++elem_per_thread)
      {
        const int tile_size = block_dim * elem_per_thread;
        const int smem_size = bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(tile_size);
        if (smem_size > *max_smem)
        {
          // assert should be prevented by smem check in policy
          assert(last_counts.elem_per_thread > 0 && "MIN_ITEMS_PER_THREAD exceeds available shared memory");
          return last_counts;
        }

        if (tile_size >= num_items)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        int max_occupancy = 0;
        const auto error  = CubDebug(MaxSmOccupancy(max_occupancy, KERNEL_PTR, block_dim, smem_size));
        if (error != cudaSuccess)
        {
          return error;
        }

        const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
        if (ActivePolicy::min_bif <= bytes_in_flight_SM)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        last_counts = elem_counts{elem_per_thread, tile_size, smem_size};
      }
      return last_counts;
    };
    // this static variable exists for each template instantiation of the surrounding function and class, on which the
    // chosen element count solely depends (assuming max SMEM is constant during a program execution)
    static auto config = determine_element_counts();
    if (!config)
    {
      return config.error;
    }
    assert(config->elem_per_thread > 0);
    assert(config->tile_size > 0);
    assert(config->tile_size % bulk_copy_alignment == 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (config->smem_size != 0)); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{config->tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, config->smem_size, stream),
      KERNEL_PTR,
      config->elem_per_thread);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>)
  {
    auto ret = configure_ublkcp_kernel<ActivePolicy>();
    if (!ret)
    {
      return ret.error;
    }
    // TODO(bgruber): use a structured binding in C++17
    // auto [launcher, kernel, elem_per_thread] = *ret;
    return ::cuda::std::get<0>(*ret).doit(
      ::cuda::std::get<1>(*ret),
      num_items,
      ::cuda::std::get<2>(*ret),
      op,
      out,
      make_aligned_base_ptr_kernel_arg<THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>>(
        THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), num_items, bulk_copy_alignment)...);
  }
#endif // _CUB_HAS_TRANSFORM_UBLKCP

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::memcpy_async>)
  {
    auto ret = configure_memcpy_async_kernel<ActivePolicy>();
    if (!ret)
    {
      return ret.error;
    }
    // TODO(bgruber): use a structured binding in C++17
    // auto [launcher, kernel, elem_per_thread] = *ret;
    return ::cuda::std::get<0>(*ret).doit(
      ::cuda::std::get<1>(*ret),
      num_items,
      ::cuda::std::get<2>(*ret),
      op,
      out,
      make_aligned_base_ptr_kernel_arg<THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>>(
        THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in)),
        num_items,
        memcpy_async_alignment)...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_algorithm(
    cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::unrolled_staged>)
  {
    using policy_t      = typename ActivePolicy::algo_policy;
    const auto grid_dim = static_cast<unsigned int>(
      ::cuda::ceil_div(num_items, Offset{policy_t::BLOCK_THREADS * policy_t::ITEMS_PER_THREAD}));
    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, policy_t::BLOCK_THREADS, 0, stream)
      .doit(KERNEL_PTR,
            num_items,
            /* items per thread taken from policy */ -1,
            op,
            out,
            make_iterator_kernel_arg<THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>>(
              THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>)
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;
    int max_occupancy       = 0;
    const auto error        = MaxSmOccupancy(max_occupancy, KERNEL_PTR, block_dim, 0);
    if (error != cudaSuccess)
    {
      return error;
    }

    const int items_per_thread =
      loaded_bytes_per_iter == 0
        ? +policy_t::ITEMS_PER_THREAD_NO_INPUT
        : ::cuda::ceil_div(ActivePolicy::min_bif, max_occupancy * block_dim * loaded_bytes_per_iter);
    const int items_per_thread_clamped =
      ::cuda::std::clamp(items_per_thread, +policy_t::MIN_ITEMS_PER_THREAD, +policy_t::MAX_ITEMS_PER_THREAD);
    const Offset tile_size = block_dim * items_per_thread_clamped;
    const auto grid_dim    = static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size));

    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, 0, stream)
      .doit(KERNEL_PTR,
            num_items,
            items_per_thread_clamped,
            op,
            out,
            make_iterator_kernel_arg<THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>>(
              THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...);
  }

  template <std::size_t... Is>
  struct non_contiguous_fallback_op_t
  {
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
    RandomAccessIteratorOut out;
    mutable TransformOp op; // too many users forgot to mark their operator()'s const ...

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(Offset i) const
    {
      out[i] = op(::cuda::std::get<Is>(in)[i]...);
    }
  };

  // TODO(bgruber): this algorithm is never dispatched to. It is just used by the benchmarks as a baseline for the first
  // round of testing. Remove it later.
  template <typename, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::fallback_for>)
  {
    using op_t = non_contiguous_fallback_op_t<Is...>;
    return for_each::dispatch_t<Offset, op_t>::dispatch(num_items, op_t{in, out, op}, stream);
  }

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    // // TODO(bgruber): replace the overload set by if constexpr in C++17
    return invoke_algorithm<ActivePolicy>(::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{},
                                          ::cuda::std::integral_constant<Algorithm, ActivePolicy::algorithm>{});
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
    RandomAccessIteratorOut out,
    Offset num_items,
    TransformOp op,
    cudaStream_t stream)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_t dispatch{::cuda::std::move(in), ::cuda::std::move(out), num_items, ::cuda::std::move(op), stream};
    return CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
  }

#undef KERNEL_PTR
};
} // namespace transform
} // namespace detail
CUB_NAMESPACE_END
