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

#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/expected>

// cooperative groups do not support NVHPC yet
#if !_CCCL_CUDA_COMPILER(NVHPC)
#  include <cooperative_groups.h>
#  include <cooperative_groups/memcpy_async.h>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::transform
{

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE const char* round_down_ptr(const T* ptr, unsigned alignment)
{
  _CCCL_ASSERT(::cuda::std::has_single_bit(alignment), "");
  return reinterpret_cast<const char*>(
    reinterpret_cast<::cuda::std::uintptr_t>(ptr) & ~::cuda::std::uintptr_t{alignment - 1});
}

// Prefetches (at least on Hopper) a 128 byte cache line. Prefetching out-of-bounds addresses has no side effects
// TODO(bgruber): there is also the cp.async.bulk.prefetch instruction available on Hopper. May improve perf a tiny bit
// as we need to create less instructions to prefetch the same amount of data.
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch(const T* addr)
{
  // TODO(bgruber): prefetch to L1 may be even better
  asm volatile("prefetch.global.L2 [%0];" : : "l"(__cvta_generic_to_global(addr)) : "memory");
}

template <int BlockDim, typename It>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch_tile(It begin, int tile_size)
{
  if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>)
  {
    constexpr int prefetch_byte_stride = 128; // TODO(bgruber): should correspond to cache line size. Does this need to
                                              // be architecture dependent?
    const int tile_size_bytes = tile_size * sizeof(it_value_t<It>);

    // prefetch does not stall and unrolling just generates a lot of unnecessary computations and predicate handling
    _CCCL_PRAGMA_NOUNROLL()
    for (int offset = threadIdx.x * prefetch_byte_stride; offset < tile_size_bytes;
         offset += BlockDim * prefetch_byte_stride)
    {
      prefetch(reinterpret_cast<const char*>(::cuda::std::to_address(begin)) + offset);
    }
  }
}

// This kernel guarantees that objects passed as arguments to the user-provided transformation function f reside in
// global memory. No intermediate copies are taken. If the parameter type of f is a reference, taking the address of the
// parameter yields a global memory address.
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
  constexpr int block_dim = PrefetchPolicy::block_threads;
  const int tile_stride   = block_dim * num_elem_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size     = static_cast<int>((::cuda::std::min)(num_items - offset, Offset{tile_stride}));

  // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
  {
    (..., (ins += offset));
    out += offset;
  }

  (..., prefetch_tile<block_dim>(THRUST_NS_QUALIFIER::raw_reference_cast(ins), tile_size));

  auto process_tile = [&](auto full_tile, auto... ins2 /* nvcc fails to compile when just using the captured ins */) {
    // ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost
    // bgruber: but A6000 and H100 show small gains without pragma
    // _CCCL_PRAGMA_NOUNROLL()
    for (int j = 0; j < num_elem_per_thread; ++j)
    {
      const int idx = j * block_dim + threadIdx.x;
      if (full_tile || idx < tile_size)
      {
        // we have to unwrap Thrust's proxy references here for backward compatibility (try zip_iterator.cu test)
        out[idx] = f(THRUST_NS_QUALIFIER::raw_reference_cast(ins2[idx])...);
      }
    }
  };
  if (tile_stride == tile_size)
  {
    process_tile(::cuda::std::true_type{}, ins...);
  }
  else
  {
    process_tile(::cuda::std::false_type{}, ins...);
  }
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

// Pointer with metadata to describe readonly input memory for memcpy_async and UBLKCP kernels.
// cg::memcpy_async is most efficient when the data is 16-byte aligned and the size a multiple of 16 bytes
// UBLKCP is most efficient when the data is 128-byte aligned and the size a multiple of 16 bytes
template <typename T> // Cannot add alignment to signature, because we need a uniform kernel template instantiation
struct aligned_base_ptr
{
  using value_type = T;

  const char* ptr; // aligned pointer before the original pointer (16-byte or 128-byte). May not be aligned to
                   // alignof(T). E.g.: array of int3 starting at address 4, ptr == 0
  int head_padding; // byte offset between ptr and the original pointer. Value inside [0;15] or [0;127].

  _CCCL_HOST_DEVICE const T* ptr_to_elements() const
  {
    return reinterpret_cast<const T*>(ptr + head_padding);
  }

  _CCCL_HOST_DEVICE friend bool operator==(const aligned_base_ptr& a, const aligned_base_ptr& b)
  {
    return a.ptr == b.ptr && a.head_padding == b.head_padding;
  }
};

template <typename T>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr(const T* ptr, int alignment) -> aligned_base_ptr<T>
{
  const char* base_ptr = round_down_ptr(ptr, alignment);
  return aligned_base_ptr<T>{base_ptr, static_cast<int>(reinterpret_cast<const char*>(ptr) - base_ptr)};
}

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
_CCCL_DEVICE _CCCL_FORCEINLINE static bool elect_one()
{
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
}

template <typename Offset, typename T>
_CCCL_DEVICE void bulk_copy_tile_fallback(
  int tile_size,
  int tile_stride,
  char* smem,
  int& smem_offset,
  Offset global_offset,
  const aligned_base_ptr<T>& aligned_ptr)
{
  const T* src = aligned_ptr.ptr_to_elements() + global_offset;
  T* dst       = reinterpret_cast<T*>(smem + smem_offset + aligned_ptr.head_padding);
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0, "");
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0, "");

  const int bytes_to_copy = static_cast<int>(sizeof(T)) * tile_size;
  cooperative_groups::memcpy_async(cooperative_groups::this_thread_block(), dst, src, bytes_to_copy);

  // add bulk_copy_alignment to make space for the next tile's head padding
  smem_offset += static_cast<int>(sizeof(T)) * tile_stride + bulk_copy_alignment;
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_ublkcp(
  Offset num_items, int num_elem_per_thread, F f, RandomAccessIteratorOut out, aligned_base_ptr<InTs>... aligned_ptrs)
{
  __shared__ uint64_t bar;
  extern __shared__ char __align__(bulk_copy_alignment) smem[];

  namespace ptx = ::cuda::ptx;

  constexpr int block_dim = BulkCopyPolicy::block_threads;
  const int tile_stride   = block_dim * num_elem_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size     = (::cuda::std::min)(num_items - offset, Offset{tile_stride});

  const bool inner_blocks = 0 < blockIdx.x && blockIdx.x + 2 < gridDim.x;
  if (inner_blocks)
  {
    // use one thread to setup the entire bulk copy
    if (elect_one())
    {
      ptx::mbarrier_init(&bar, 1);
      ptx::fence_proxy_async(ptx::space_shared);

      int smem_offset                    = 0;
      ::cuda::std::uint32_t total_copied = 0;

      auto bulk_copy_tile = [&](auto aligned_ptr) {
        using T = typename decltype(aligned_ptr)::value_type;
        static_assert(alignof(T) <= bulk_copy_alignment, "");

        const char* src = aligned_ptr.ptr + offset * sizeof(T);
        char* dst       = smem + smem_offset;
        _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % bulk_copy_alignment == 0, "");
        _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % bulk_copy_alignment == 0, "");

        // TODO(bgruber): we could precompute bytes_to_copy on the host
        const int bytes_to_copy = round_up_to_po2_multiple(
          aligned_ptr.head_padding + static_cast<int>(sizeof(T)) * tile_stride, bulk_copy_size_multiple);

        ::cuda::ptx::cp_async_bulk(::cuda::ptx::space_cluster, ::cuda::ptx::space_global, dst, src, bytes_to_copy, &bar);
        total_copied += bytes_to_copy;

        // add bulk_copy_alignment to make space for the next tile's head padding
        smem_offset += static_cast<int>(sizeof(T)) * tile_stride + bulk_copy_alignment;
      };

      // Order of evaluation is left-to-right
      (..., bulk_copy_tile(aligned_ptrs));

      // TODO(ahendriksen): this could only have ptx::sem_relaxed, but this is not available yet
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
    }

    // all threads wait for bulk copy
    __syncthreads();
    while (!ptx::mbarrier_try_wait_parity(&bar, 0))
      ;
  }
  else
  {
    // use all threads to schedule an async_memcpy
    int smem_offset = 0;

    auto bulk_copy_tile_fallback = [&](auto aligned_ptr) {
      using T      = typename decltype(aligned_ptr)::value_type;
      const T* src = aligned_ptr.ptr_to_elements() + offset;
      T* dst       = reinterpret_cast<T*>(smem + smem_offset + aligned_ptr.head_padding);
      _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0, "");
      _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0, "");

      const int bytes_to_copy = static_cast<int>(sizeof(T)) * tile_size;
      cooperative_groups::memcpy_async(cooperative_groups::this_thread_block(), dst, src, bytes_to_copy);

      // add bulk_copy_alignment to make space for the next tile's head padding
      smem_offset += static_cast<int>(sizeof(T)) * tile_stride + bulk_copy_alignment;
    };

    // Order of evaluation is left-to-right
    (..., bulk_copy_tile_fallback(aligned_ptrs));

    cooperative_groups::wait(cooperative_groups::this_thread_block());
  }

  // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
  out += offset;

  auto process_tile = [&](auto full_tile) {
    // Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark)
    _CCCL_PRAGMA_NOUNROLL()
    for (int j = 0; j < num_elem_per_thread; ++j)
    {
      const int idx = j * block_dim + threadIdx.x;
      if (full_tile || idx < tile_size)
      {
        int smem_offset    = 0;
        auto fetch_operand = [&](auto aligned_ptr) {
          using T                         = typename decltype(aligned_ptr)::value_type;
          const T* smem_operand_tile_base = reinterpret_cast<const T*>(smem + smem_offset + aligned_ptr.head_padding);
          smem_offset += int{sizeof(T)} * tile_stride + bulk_copy_alignment;
          return smem_operand_tile_base[idx];
        };

        // need to expand into a tuple for guaranteed order of evaluation
        out[idx] = ::cuda::std::apply(
          [&](auto... values) {
            return f(values...);
          },
          ::cuda::std::tuple<InTs...>{fetch_operand(aligned_ptrs)...});
      }
    }
  };
  // explicitly calling the lambda on literal true/false lets the compiler emit the lambda twice
  if (tile_stride == tile_size)
  {
    process_tile(::cuda::std::true_type{});
  }
  else
  {
    process_tile(::cuda::std::false_type{});
  }
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  aligned_base_ptr<InTs>... aligned_ptrs)
{
  // only call the real kernel for sm90 and later
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (transform_kernel_ublkcp<BulkCopyPolicy>(num_items, num_elem_per_thread, f, out, aligned_ptrs...);));
}
#endif // _CUB_HAS_TRANSFORM_UBLKCP

template <typename It>
union kernel_arg
{
#if _CUB_HAS_TRANSFORM_UBLKCP
  aligned_base_ptr<it_value_t<It>> aligned_ptr; // first member is trivial
  static_assert(::cuda::std::is_trivial_v<decltype(aligned_ptr)>, "");
#endif
  It iterator; // may not be trivially [default|copy]-constructible

  // Sometimes It is not trivially [default|copy]-constructible (e.g.
  // thrust::normal_iterator<thrust::device_pointer<T>>), so because of
  // https://eel.is/c++draft/class.union#general-note-3, kernel_args's special members are deleted. We work around it by
  // explicitly defining them.
  _CCCL_HOST_DEVICE kernel_arg() noexcept {}
  _CCCL_HOST_DEVICE ~kernel_arg() noexcept {}

  _CCCL_HOST_DEVICE kernel_arg(const kernel_arg& other)
  {
    // since we use kernel_arg only to pass data to the device, the contained data is semantically trivially copyable,
    // even if the type system is telling us otherwise.
    ::cuda::std::memcpy(reinterpret_cast<char*>(this), reinterpret_cast<const char*>(&other), sizeof(kernel_arg));
  }
};

template <typename It>
_CCCL_HOST_DEVICE auto make_iterator_kernel_arg(It it) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  // since we switch the active member of the union, we must use placement new or construct_at. This also uses the copy
  // constructor of It, which works in more cases than assignment (e.g. thrust::transform_iterator with
  // non-copy-assignable functor, e.g. in merge sort tests)
  ::cuda::std::__construct_at(&arg.iterator, it);
  return arg;
}

template <typename It>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr_kernel_arg(It ptr, int alignment) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.aligned_ptr = make_aligned_base_ptr(ptr, alignment);
  return arg;
}

template <Algorithm Alg>
inline constexpr bool needs_aligned_ptr_v =
  false
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  || Alg == Algorithm::ublkcp
#endif // _CUB_HAS_TRANSFORM_UBLKCP
  ;

template <Algorithm Alg, typename It>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
select_kernel_arg(::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg)
{
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  if constexpr (needs_aligned_ptr_v<Alg>)
  {
    return ::cuda::std::move(arg.aligned_ptr);
  }
  else
#endif // _CUB_HAS_TRANSFORM_UBLKCP
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
__launch_bounds__(MaxPolicy::ActivePolicy::algo_policy::block_threads)
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

} // namespace detail::transform

CUB_NAMESPACE_END
