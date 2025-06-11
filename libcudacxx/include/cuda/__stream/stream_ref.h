//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_STREAM_REF
#define _CUDA___STREAM_STREAM_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/get_stream.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief A non-owning wrapper for a `cudaStream_t`.
class stream_ref
{
protected:
  ::cudaStream_t __stream{0};

public:
  using value_type = ::cudaStream_t;

  //! @brief Constructs a `stream_ref` of the "default" CUDA stream.
  //!
  //! For behavior of the default stream,
  //! @see //! https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
  _CCCL_HIDE_FROM_ABI stream_ref() = default;

  //! @brief Constructs a `stream_ref` from a `cudaStream_t` handle.
  //!
  //! This constructor provides implicit conversion from `cudaStream_t`.
  //!
  //! @note: It is the callers responsibility to ensure the `stream_ref` does not
  //! outlive the stream identified by the `cudaStream_t` handle.
  _LIBCUDACXX_HIDE_FROM_ABI constexpr stream_ref(value_type __stream_) noexcept
      : __stream{__stream_}
  {}

  //! Disallow construction from an `int`, e.g., `0`.
  stream_ref(int) = delete;

  //! Disallow construction from `nullptr`.
  stream_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Compares two `stream_ref`s for equality
  //!
  //! @note Allows comparison with `cudaStream_t` due to implicit conversion to
  //! `stream_ref`.
  //!
  //! @param lhs The first `stream_ref` to compare
  //! @param rhs The second `stream_ref` to compare
  //! @return true if equal, false if unequal
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const stream_ref& __lhs, const stream_ref& __rhs) noexcept
  {
    return __lhs.__stream == __rhs.__stream;
  }

  //! @brief Compares two `stream_ref`s for inequality
  //!
  //! @note Allows comparison with `cudaStream_t` due to implicit conversion to
  //! `stream_ref`.
  //!
  //! @param lhs The first `stream_ref` to compare
  //! @param rhs The second `stream_ref` to compare
  //! @return true if unequal, false if equal
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const stream_ref& __lhs, const stream_ref& __rhs) noexcept
  {
    return __lhs.__stream != __rhs.__stream;
  }

  //! Returns the wrapped `cudaStream_t` handle.
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr value_type get() const noexcept
  {
    return __stream;
  }

  //! @brief Synchronizes the wrapped stream.
  //!
  //! @throws cuda::cuda_error if synchronization fails.
  void sync() const
  {
    _CCCL_TRY_CUDA_API(::cudaStreamSynchronize, "Failed to synchronize stream.", get());
  }

  //! @brief Deprecated. Use sync() instead.
  //!
  //! @deprecated Use sync() instead.
  [[deprecated("Use sync() instead.")]]
  void wait() const
  {
    sync();
  }

  //! @brief Queries if all operations on the wrapped stream have completed.
  //!
  //! @throws cuda::cuda_error if the query fails.
  //!
  //! @return `true` if all operations have completed, or `false` if not.
  [[nodiscard]] bool ready() const
  {
    const auto __result = ::cudaStreamQuery(get());
    if (__result == ::cudaErrorNotReady)
    {
      return false;
    }
    switch (__result)
    {
      case ::cudaSuccess:
        break;
      default:
        ::cuda::__throw_cuda_error(__result, "Failed to query stream.");
    }
    return true;
  }

  //! @brief Queries the priority of the wrapped stream.
  //!
  //! @throws cuda::cuda_error if the query fails.
  //!
  //! @return value representing the priority of the wrapped stream.
  [[nodiscard]] int priority() const
  {
    int __result = 0;
    _CCCL_TRY_CUDA_API(::cudaStreamGetPriority, "Failed to get stream priority", get(), &__result);
    return __result;
  }

  //! @brief Queries the \c stream_ref for itself. This makes \c stream_ref usable in places where we expect an
  //! environment with a \c get_stream_t query
  [[nodiscard]] stream_ref query(const ::cuda::get_stream_t&) const noexcept
  {
    return *this;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___STREAM_STREAM_REF
