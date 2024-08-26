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
#ifdef __cpp_fold_expressions // C++17
template <typename... Its>
constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(value_t<Its>)} + ... + 0);
}
#else
constexpr int sum()
{
  return 0;
}

template <typename... Ts>
constexpr int sum(int head, Ts... tail)
{
  return head + sum(tail...);
}

template <typename... Its>
constexpr auto loaded_bytes_per_iteration() -> int
{
  return sum(int{sizeof(value_t<Its>)}...);
}
#endif

enum class Algorithm
{
  fallback_for,
  prefetch,
  unrolled_staged,
  memcpy_async
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  ,
  ublkcp
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

// TODO(bgruber): only needed so we can instantiate the kernel generically for any policy. Remove when fallback_for is
// dropped.
template <typename, typename Offset, typename F, typename RandomAccessIteartorOut, typename... RandomAccessIteartorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::fallback_for>,
  Offset,
  int,
  F,
  RandomAccessIteartorOut,
  RandomAccessIteartorIn...)
{}

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int ITEMS_PER_THREAD_NO_INPUT = 2; // items per thread when no input streams exits (just filling)
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
template <typename, typename Offset, typename F, typename RandomAccessIteartorOut, typename... RandomAccessIteartorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteartorOut out,
  RandomAccessIteartorIn... ins)
{
  {
    const int tile_stride = blockDim.x * num_elem_per_thread;
    const Offset offset   = static_cast<Offset>(blockIdx.x) * tile_stride;

    // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
    num_items -= offset;
    int dummy[] = {(ins += offset, 0)..., 0};
    (void) dummy;
    out += offset;
  }

  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    const int idx = j * blockDim.x + threadIdx.x;
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
    const int idx = j * blockDim.x + threadIdx.x;
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
          typename RandomAccessIteartorOut,
          typename... RandomAccessIteartorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::unrolled_staged>,
  Offset num_items,
  int,
  F f,
  RandomAccessIteartorOut out,
  RandomAccessIteartorIn... ins)
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

  [&](cuda::std::array<value_t<RandomAccessIteartorIn>, items_per_thread>&&... arrays) {
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
  }(cuda::std::array<value_t<RandomAccessIteartorIn>, items_per_thread>{}...);
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
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto round_up_to_po2_multiple(Integral x, Integral mult) -> Integral
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

// Pointer with metadata to describe input memory for memcpy_async and UBLKCP kernels.
// cg::memcpy_async is most efficient when the data is 16-byte aligned and the size a multiple of 16 bytes
// UBLKCP is most efficient when the data is 128-byte aligned and the size a multiple of 16 bytes
template <typename T> // Cannot add alignment to signature, because we need a uniform kernel template instantiation
struct aligned_base_ptr
{
  T* ptr; // aligned pointer before the original pointer (16-byte or 128-byte)
  int offset; // byte offset between ptr and the original pointer. Value inside [0;15] or [0;127].

  _CCCL_HOST_DEVICE friend bool operator==(const aligned_base_ptr& a, const aligned_base_ptr& b)
  {
    return a.ptr == b.ptr && a.offset == b.offset;
  }
};

template <typename T>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr(const T* ptr, int alignment) -> aligned_base_ptr<const T>
{
  // TODO(bgruber): is it actually legal to move the pointer to a lower 128-byte aligned address and start reading
  // from there in the kernel? The CUDA programming guide says:
  //   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
  //   > Any address of a variable residing in global memory or returned by one of the memory allocation routines from
  //     the driver or runtime API is always aligned to at least 256 bytes.
  // However, gevtushenko says since Linux Kernel 6, any host memory is device accessible, even stack memory
  // ahendriksen argues that all memory pages are sufficiently aligned, even those migrated from the host
  const T* base_ptr = round_down_ptr(ptr, alignment);
  return aligned_base_ptr<const T>{base_ptr, static_cast<int>((ptr - base_ptr) * sizeof(T))};
}

// For performance considerations of memcpy_async:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidance-for-memcpy-async

constexpr int memcpy_async_alignment     = 16;
constexpr int memcpy_async_size_multiple = 16;

// Our own version of ::cuda::aligned_size_t, since we cannot include <cuda/barrier> on CUDA_ARCH < 700
template <_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t
{
  _CUDA_VSTD::size_t value;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr operator size_t() const
  {
    return value;
  }
};

// TODO(bgruber): inline this as lambda in C++14
template <typename T>
_CCCL_DEVICE const T* copy_and_return_smem_dst(
  cooperative_groups::thread_block& group,
  int tile_size,
  char* smem,
  int& smem_offset,
  int global_offset,
  aligned_base_ptr<const T> aligned_ptr)
{
  const int count =
    round_up_to_po2_multiple(static_cast<int>(sizeof(T)) * tile_size + aligned_ptr.offset, memcpy_async_size_multiple);
  // because SMEM base pointer and count are always multiples of 16-byte, we only need to align the SMEM start for types
  // with larger alignment
  _CCCL_IF_CONSTEXPR (alignof(T) > memcpy_async_alignment)
  {
    smem_offset = round_up_to_po2_multiple(smem_offset, static_cast<int>(alignof(T)));
  }
  auto smem_dst = reinterpret_cast<T*>(smem + smem_offset);
  assert(reinterpret_cast<uintptr_t>(smem_dst) % memcpy_async_size_multiple == 0); // to hit optimal memcpy_async
                                                                                   // performance
  cooperative_groups::memcpy_async(
    group,
    smem_dst,
    aligned_ptr.ptr + global_offset,
    aligned_size_t<memcpy_async_size_multiple>{static_cast<::cuda::std::size_t>(count)});
  smem_offset += count;
  return smem_dst + aligned_ptr.offset;
}

template <typename, typename Offset, typename F, typename RandomAccessIteartorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::memcpy_async>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteartorOut out,
  aligned_base_ptr<const InTs>... aligned_ptrs)
{
  extern __shared__ char smem[]; // this should be __attribute((aligned(memcpy_async_alignment))), but then it clashes
                                 // with the ublkcp kernel, which sets a higher alignment, since they are both called
                                 // from the same kernel entry point (albeit one is always discarded). However, SMEM is
                                 // 16-byte aligned by default.

  const Offset tile_stride = blockDim.x * num_elem_per_thread;
  const Offset offset      = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size      = ::cuda::std::min(num_items - offset, tile_stride);

  auto group           = cooperative_groups::this_thread_block();
  int smem_offset      = 0;
  const auto smem_ptrs = ::cuda::std::tuple<const InTs*...>{
    copy_and_return_smem_dst(group, tile_size, smem, smem_offset, offset, aligned_ptrs)...};
  cooperative_groups::wait(group);
  (void) smem_ptrs; // suppress unused warning for MSVC
  (void) &smem_offset; // MSVC needs extra strong unused warning supression

  {
    // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
    num_items -= offset;
    out += offset;
  }

#pragma unroll 1
  for (int i = 0; i < num_elem_per_thread; ++i)
  {
    const int idx = i * blockDim.x + threadIdx.x;
    if (idx < num_items)
    {
      out[idx] = poor_apply(
        [&](const InTs*... smem_base_ptrs) {
          return f(smem_base_ptrs[idx]...);
        },
        smem_ptrs);
    }
  }
}

constexpr int ublkcp_alignment     = 128;
constexpr int ublkcp_size_multiple = 16;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
// TODO(bgruber): inline this as lambda in C++14
template <typename Offset, typename T>
_CCCL_DEVICE void copy_ptr_set(
  ::cuda::std::uint64_t& bar,
  int tile_size,
  int tile_stride,
  char* smem,
  int& smem_offset,
  ::cuda::std::uint32_t& total_copied,
  Offset global_offset,
  const aligned_base_ptr<T>& aligned_ptr)
{
#  if CUB_PTX_ARCH >= 900
  namespace ptx = ::cuda::ptx;
  // Copy a bit more than tile_size, to cover for base_ptr starting earlier than ptr
  const uint32_t num_bytes = round_up_to_po2_multiple(
    aligned_ptr.offset + static_cast<uint32_t>(sizeof(T)) * tile_size, static_cast<uint32_t>(ublkcp_size_multiple));
  ptx::cp_async_bulk(
    ptx::space_cluster,
    ptx::space_global,
    smem + smem_offset,
    aligned_ptr.ptr + global_offset, // Use 128-byte aligned base_ptr here
    num_bytes,
    &bar);
  smem_offset += static_cast<int>(sizeof(T)) * tile_stride + 128;
  total_copied += num_bytes;
#  endif // CUB_PTX_ARCH >= 900
};

// TODO(bgruber): inline this as lambda in C++14
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE const T&
fetch_operand(int tile_stride, const char* smem, int& smem_offset, int smem_idx, const aligned_base_ptr<T>& aligned_ptr)
{
  const T* smem_operand_tile_base = reinterpret_cast<const T*>(smem + smem_offset + aligned_ptr.offset);
  smem_offset += int{sizeof(T)} * tile_stride + ublkcp_alignment;
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

template <typename, typename Offset, typename F, typename RandomAccessIteartorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteartorOut out,
  aligned_base_ptr<const InTs>... aligned_ptrs)
{
#  if CUB_PTX_ARCH >= 900
  __shared__ uint64_t bar;
  extern __shared__ char __attribute((aligned(ublkcp_alignment))) smem[];

  namespace ptx = ::cuda::ptx;

  const int tile_stride = blockDim.x * num_elem_per_thread;
  const Offset offset   = static_cast<Offset>(blockIdx.x) * tile_stride;

  // TODO(bgruber) use: `cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&]() {` with CTK
  // >= 12.1
  if (select_one())
  {
    // Then initialize barriers
    ptx::mbarrier_init(&bar, 1);
    ptx::fence_proxy_async(ptx::space_shared);

    const int tile_size                = ::cuda::std::min(num_items - offset, Offset{tile_stride});
    int smem_offset                    = 0;
    ::cuda::std::uint32_t total_copied = 0;

#    ifdef __cpp_fold_expressions // C++17
    // Order of evaluation is always left-to-right here. So smem_offset is updated in the right order.
    (..., copy_ptr_set(bar, tile_size, tile_stride, smem, smem_offset, total_copied, offset, aligned_ptrs));
#    else
    // Order of evaluation is also left-to-right
    int dummy[] = {
      (copy_ptr_set(bar, tile_size, tile_stride, smem, smem_offset, total_copied, offset, aligned_ptrs), 0)..., 0};
    (void) dummy;
#    endif

    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
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
    const int idx = j * blockDim.x + threadIdx.x;
    if (idx < num_items)
    {
      int smem_offset = 0;
      out[idx]        = f(fetch_operand(tile_stride, smem, smem_offset, idx, aligned_ptrs)...);
    }
  }
#  endif // CUB_PTX_ARCH >= 900
}
#endif // _CUB_HAS_TRANSFORM_UBLKCP

// Type erasing "union" for kernel arguments. We cannot use a real union, because it would not be trivially copyable in
// case It is not trivially copyable (e.g. thrust::constant_iterator<custom_numeric>)
template <typename It>
struct kernel_arg
{
  using PS                               = aligned_base_ptr<const value_t<It>>;
  static constexpr std::size_t alignment = ::cuda::std::max(alignof(It), alignof(PS)); // need extra variable for GCC<9
  alignas(alignment) char storage[::cuda::std::max(sizeof(It), sizeof(PS))];

  template <typename T>
  _CCCL_HOST_DEVICE T& aliased_storage()
  {
    static_assert(::cuda::std::is_same<T, It>::value || ::cuda::std::is_same<T, PS>::value, "");
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")
    return *reinterpret_cast<T*>(storage);
    _CCCL_DIAG_POP
  }
};

template <typename It>
_CCCL_HOST_DEVICE auto make_iterator_kernel_arg(It it) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  cub::detail::uninitialized_copy_single(&arg.template aliased_storage<It>(), it);
  return arg;
}

template <typename It>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr_kernel_arg(It ptr, int alignment) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  using T = value_t<It>;
  cub::detail::uninitialized_copy_single(
    &arg.template aliased_storage<aligned_base_ptr<const T>>(), make_aligned_base_ptr(ptr, alignment));
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
  return ::cuda::std::move(arg.template aliased_storage<aligned_base_ptr<const value_t<It>>>());
}

template <Algorithm Alg, typename It, ::cuda::std::__enable_if_t<!needs_aligned_ptr_t<Alg>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
select_kernel_arg(::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg) -> It&&
{
  return ::cuda::std::move(arg.template aliased_storage<It>());
}

// There is only one kernel for all algorithms, that dispatches based on the selected policy. It must be instantiated
// with the same arguments for each algorithm. Only the device compiler will then select the implementation. This
// saves some compile-time and binary size.
template <typename MaxPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteartorOut,
          typename... RandomAccessIteartorsIn>
__launch_bounds__(MaxPolicy::ActivePolicy::algo_policy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void transform_kernel(
    Offset num_items,
    int num_elem_per_thread,
    F f,
    RandomAccessIteartorOut out,
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

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>>
{
  // TODO(gevtushenko): take a look at radix sort dispatch

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
    // TODO(bgruber): we could use unrolled_staged if we cannot memcpy
    static constexpr auto algorithm =
      (RequiresStableAddress || !can_memcpy || no_input_streams) ? Algorithm::prefetch : Algorithm::memcpy_async;
    using algo_policy = ::cuda::std::
      _If<RequiresStableAddress || !can_memcpy || no_input_streams, prefetch_policy_t<256>, async_copy_policy_t<256>>;
  };

  // TODO(bgruber): should we add a tuning for 860? They should have items_per_thread_from_occupancy(256, 6, ...)

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
    static constexpr auto algorithm =
      (RequiresStableAddress || !can_memcpy || no_input_streams)
        ? Algorithm::prefetch
        :
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
        Algorithm::ublkcp;
#else
        Algorithm::memcpy_async;
#endif
    using algo_policy = ::cuda::std::
      _If<RequiresStableAddress || !can_memcpy || no_input_streams, prefetch_policy_t<256>, async_copy_policy_t<256>>;
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
};

_CCCL_HOST_DEVICE inline PoorExpected<int> get_max_shared_memory()
{
  // gevtushenko promised me that I can assume that stream belongs to the currently active device
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

  using kernel_ptr_t =
    decltype(&transform_kernel<typename PolicyHub::max_policy,
                               Offset,
                               TransformOp,
                               RandomAccessIteratorOut,
                               THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>);
  kernel_ptr_t kernel;

  static constexpr int loaded_bytes_per_iter = loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();

  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running on?
  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto configure_memcpy_async_kernel(cuda::std::index_sequence<Is...>)
    -> PoorExpected<::cuda::std::tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron, kernel_ptr_t, int>>
  {
    // Benchmarking shows that even for a few iteration, this loop takes around 4-7 us, so should not be a concern.
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;
    static_assert(block_dim % memcpy_async_alignment == 0,
                  "BLOCK_THREADS needs to be a multiple of memcpy_async_alignment (16)"); // then tile_size is a
                                                                                          // multiple of 16-byte
    const auto max_smem = get_max_shared_memory();
    if (!max_smem)
    {
      return max_smem.error;
    }

    // TODO(bgruber): we could consider the element size here and ensure that the tilesize is a multiple of 16 bytes.
    // Increase the number of output elements per thread until we reach the required bytes in flight.
    int chosen_elem_per_thread = 0;
    int chosen_tile_size       = 0;
    int chosen_smem_size       = 0;
    for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread < +policy_t::MAX_ITEMS_PER_THREAD;
         ++elem_per_thread)
    {
      const auto tile_size = block_dim * elem_per_thread;

      int smem_size   = 0;
      auto count_smem = [&](int size, int alignment) {
        smem_size = round_up_to_po2_multiple(smem_size, alignment);
        // max aligned_base_ptr offset + max padding after == 16
        smem_size += size * tile_size + ::cuda::std::max(memcpy_async_alignment, memcpy_async_size_multiple);
      };
      // TODO(bgruber): replace by fold over comma in C++17 (left to right evaluation!)
      int dummy[] = {
        (count_smem(sizeof(value_t<RandomAccessIteratorsIn>), alignof(value_t<RandomAccessIteratorsIn>)), 0)..., 0};
      (void) &dummy; // need to take the address to suppress unused warnings more strongly for nvcc 11.1
      (void) &count_smem;

      if (smem_size > *max_smem)
      {
        break;
      }

      if (tile_size >= num_items)
      {
        chosen_elem_per_thread = elem_per_thread;
        chosen_tile_size       = tile_size;
        chosen_smem_size       = smem_size;
        break;
      }

      int max_occupancy = 0;
      const auto error  = MaxSmOccupancy(max_occupancy, kernel, block_dim, smem_size);
      if (error != cudaSuccess)
      {
        return error;
      }

      chosen_elem_per_thread = elem_per_thread;
      chosen_tile_size       = tile_size;
      chosen_smem_size       = smem_size;

      const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
      if (bytes_in_flight_SM >= ActivePolicy::min_bif)
      {
        break;
      }
    }
    assert(chosen_elem_per_thread > 0);
    assert(chosen_tile_size > 0);
    assert(chosen_tile_size % memcpy_async_alignment == 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (chosen_smem_size != 0)); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{chosen_tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, chosen_smem_size, stream),
      kernel,
      chosen_elem_per_thread);
  }

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running
  // on?
  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto configure_ublkcp_kernel(cuda::std::index_sequence<Is...>)
    -> PoorExpected<::cuda::std::tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron, kernel_ptr_t, int>>
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;
    static_assert(block_dim % ublkcp_alignment == 0,
                  "BLOCK_THREADS needs to be a multiple of ublkcp_alignment (128)"); // then tile_size is a multiple of
                                                                                     // 128-byte
    // aligned

    const auto max_smem = get_max_shared_memory();
    if (!max_smem)
    {
      return max_smem.error;
    }

    // Increase the number of output elements per thread until we reach the required bytes in flight.
    int chosen_elem_per_thread = 0;
    int chosen_tile_size       = 0;
    int chosen_smem_size       = 0;
    for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread < +policy_t::MAX_ITEMS_PER_THREAD;
         ++elem_per_thread)
    {
      constexpr int num_inputs = sizeof...(RandomAccessIteratorsIn);
      const int tile_size      = block_dim * elem_per_thread;
      // 128 bytes of padding for each input tile (before + after)
      const int smem_size =
        tile_size * loaded_bytes_per_iter + ::cuda::std::max(ublkcp_alignment, ublkcp_size_multiple) * num_inputs;

      if (smem_size > *max_smem)
      {
        break;
      }

      if (tile_size >= num_items)
      {
        chosen_elem_per_thread = elem_per_thread;
        chosen_tile_size       = tile_size;
        chosen_smem_size       = smem_size;
        break;
      }

      int max_occupancy = 0;
      const auto error  = MaxSmOccupancy(max_occupancy, kernel, block_dim, smem_size);
      if (error != cudaSuccess)
      {
        return error;
      }

      chosen_elem_per_thread = elem_per_thread;
      chosen_tile_size       = tile_size;
      chosen_smem_size       = smem_size;

      const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
      if (ActivePolicy::min_bif <= bytes_in_flight_SM)
      {
        break;
      }
    }
    assert(chosen_elem_per_thread > 0);
    assert(chosen_tile_size > 0);
    assert(chosen_tile_size % ublkcp_alignment == 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (chosen_smem_size != 0)); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{chosen_tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, chosen_smem_size, stream),
      kernel,
      chosen_elem_per_thread);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...> is, ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>)
  {
    auto ret = configure_ublkcp_kernel<ActivePolicy>(is);
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
        THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), ublkcp_alignment)...);
  }
#endif // _CUB_HAS_TRANSFORM_UBLKCP

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_algorithm(
    cuda::std::index_sequence<Is...> is, ::cuda::std::integral_constant<Algorithm, Algorithm::memcpy_async>)
  {
    auto ret = configure_memcpy_async_kernel<ActivePolicy>(is);
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
        THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), memcpy_async_alignment)...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_algorithm(
    cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::unrolled_staged>)
  {
    using policy_t      = typename ActivePolicy::algo_policy;
    const auto grid_dim = static_cast<unsigned int>(
      ::cuda::ceil_div(num_items, Offset{policy_t::BLOCK_THREADS * policy_t::ITEMS_PER_THREAD}));
    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, policy_t::BLOCK_THREADS, 0, stream)
      .doit(kernel,
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
    const auto error        = MaxSmOccupancy(max_occupancy, kernel, block_dim, 0);
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
      .doit(kernel,
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
    mutable TransformOp op; // too many users forgot to mark there operator()'s const ...

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

    auto kernel =
      &transform_kernel<typename PolicyHub::max_policy,
                        Offset,
                        TransformOp,
                        RandomAccessIteratorOut,
                        THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>;
    dispatch_t dispatch{::cuda::std::move(in), ::cuda::std::move(out), num_items, ::cuda::std::move(op), stream, kernel};
    return CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
  }
};
} // namespace transform
} // namespace detail
CUB_NAMESPACE_END
