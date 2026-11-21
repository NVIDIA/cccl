// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_SERIAL_MERGE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_SERIAL_MERGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cuda/std/cstdint>
#  include <cuda/std/type_traits>

#  include <cuda_runtime.h>
#  ifndef _CUDAX_SORT_MWAY_MWAY_LEAN_REFILL
#    define _CUDAX_SORT_MWAY_MWAY_LEAN_REFILL -1
#  endif
static_assert(_CUDAX_SORT_MWAY_MWAY_LEAN_REFILL >= -1 && _CUDAX_SORT_MWAY_MWAY_LEAN_REFILL <= 1);
// -1 or 1: enable within the supported scopes; 0: original refill.

#  include <cuda/experimental/mgmn/__algorithm/common.h>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
namespace detail::serial_merge_ops
{
#  if defined(__CUDA_ARCH__)
inline constexpr bool supported_arch =
  __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1070
  || (__CUDA_ARCH__ == 1200
#    if defined(_CUDAX_SORT_MWAY_MWAY_DOUBLE_BUFFER)
      && _CUDAX_SORT_MWAY_MWAY_DOUBLE_BUFFER == 0
#    endif
  );
#  else
inline constexpr bool supported_arch = false;
#  endif
template <class Key>
inline constexpr bool enabled =
  _CUDAX_SORT_MWAY_MWAY_LEAN_REFILL != 0 && supported_arch && ::cuda::std::is_same_v<Key, int>;

// take_b is the ORIGINAL exhaustion-aware predicate: false also preserves A-first ties.
template <class Key>
_CCCL_DEVICE_API _CCCL_FORCEINLINE Key select(bool take_b, Key a, Key b)
{
  return take_b ? b : a;
}

template <class Key>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
refill(bool take_b, const Key* first, const Key* second, int& a, int& b, Key& head_a, Key& head_b)
{
  if constexpr (enabled<Key>)
  {
    // Select CURRENT shared addresses before either index changes. The fixed
    // +4 displacement performs the selected successor access in one load.
    const unsigned address_a = static_cast<unsigned>(__cvta_generic_to_shared(first + a));
    const unsigned address_b = static_cast<unsigned>(__cvta_generic_to_shared(second + b));
    asm("{\n"
        "  .reg .pred pick_b;\n"
        "  .reg .u32 address, next;\n"
        "  setp.ne.u32 pick_b, %6, 0;\n"
        "  selp.u32 address, %5, %4, pick_b;\n"
        "  ld.shared.b32 next, [address+4];\n"
        "  @!pick_b add.s32 %0, %0, 1;\n"
        "  @pick_b add.s32 %1, %1, 1;\n"
        "  @!pick_b mov.b32 %2, next;\n"
        "  @pick_b mov.b32 %3, next;\n"
        "}"
        : "+r"(a), "+r"(b), "+r"(head_a), "+r"(head_b)
        : "r"(address_a), "r"(address_b), "r"(static_cast<unsigned>(take_b))
        : "memory");
    // All input-only operands are consumed before any output is overwritten;
    // tied read/write head operands preserve the unconsumed side explicitly.
  }
  else
  {
    if (take_b)
    {
      head_b = second[++b];
    }
    else
    {
      head_a = first[++a];
    }
  }
}
} // namespace detail::serial_merge_ops
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_SERIAL_MERGE_H
