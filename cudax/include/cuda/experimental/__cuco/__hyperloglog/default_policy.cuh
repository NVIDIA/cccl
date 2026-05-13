//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___HYPERLOGLOG_DEFAULT_POLICY_CUH
#define _CUDAX___CUCO___HYPERLOGLOG_DEFAULT_POLICY_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/countl.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/__hyperloglog/finalizer.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Default policy for `cuda::experimental::cuco::hyperloglog`.
//!
//! Bundles the three customization points of the HLL pipeline -- hash function, bit slicing,
//! and finalizer -- into a single policy. This default reproduces the behaviour shipped by
//! `cuCollections::hyperloglog`: MSB-indexed register selection, padded leading-zero count for
//! rho, and HyperLogLog++ bias correction. Custom policies (e.g. for binary interop with
//! third-party sketch libraries) can be supplied via the `_Policy` template parameter on
//! `hyperloglog` and `hyperloglog_ref`.
//!
//! @tparam _Key The item type the sketch counts.
//! @tparam _Algo The hash algorithm. Defaults to xxhash_64.
template <class _Key, hash_algorithm _Algo = hash_algorithm::xxhash_64>
struct default_hll_policy
{
  using hasher           = hash<_Key, _Algo>;
  using hash_result_type = decltype(::cuda::std::declval<hasher>()(::cuda::std::declval<_Key>()));
  using register_type    = ::cuda::std::int32_t;

  static_assert(::cuda::std::is_unsigned_v<hash_result_type>, "HyperLogLog requires an unsigned hash value type");
  static_assert(::cuda::std::numeric_limits<hash_result_type>::digits == 32
                  || ::cuda::std::numeric_limits<hash_result_type>::digits == 64,
                "HyperLogLog requires a 32-bit or 64-bit hash value type");

  hasher hasher_{};

  //! @brief Returns the underlying hash functor.
  //!
  //! @return The hash functor.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return hasher_;
  }

  //! @brief Hashes an item.
  //!
  //! @param[in] __k The item to hash.
  //! @return The hash value of `__k`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hash_result_type hash(const _Key& __k) const noexcept
  {
    return this->hasher_(__k);
  }

  //! @brief Extracts the register index from the hash.
  //!
  //! @note Index is taken from the high `__precision` bits of the hash, matching Apache Spark's
  //! HyperLogLog++ convention.
  //!
  //! @param[in] __h The hash value.
  //! @param[in] __precision The HLL precision parameter.
  //! @return The register index in `[0, 2^__precision)`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t
  register_index(hash_result_type __h, ::cuda::std::int32_t __precision) const noexcept
  {
    constexpr auto __hash_bits = ::cuda::std::numeric_limits<hash_result_type>::digits;
    return static_cast<::cuda::std::uint32_t>(__h >> (__hash_bits - __precision));
  }

  //! @brief Computes rho (1 + leading zeros of the rho source) from the hash.
  //!
  //! @note A one-bit padding bounds the leading-zero count at `hash_bits - __precision`,
  //! preventing rho overflow when the low `hash_bits - __precision` bits of the hash are zero.
  //!
  //! @param[in] __h The hash value.
  //! @param[in] __precision The HLL precision parameter.
  //! @return rho, in `[1, hash_bits - __precision + 1]`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint8_t
  register_value(hash_result_type __h, ::cuda::std::int32_t __precision) const noexcept
  {
    const auto __w_padding = hash_result_type{1} << static_cast<hash_result_type>(__precision - 1);
    return static_cast<::cuda::std::uint8_t>(::cuda::std::countl_zero((__h << __precision) | __w_padding) + 1);
  }

  //! @brief Finalizes the GPU reduction into a cardinality estimate using the HyperLogLog++
  //! bias-corrected estimator.
  //!
  //! @param[in] __z Sum of `2^-register[i]` across all registers.
  //! @param[in] __v Count of zero registers.
  //! @param[in] __precision HLL precision parameter.
  //! @return The bias-corrected cardinality estimate.
  [[nodiscard]] static _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t
  finalize(double __z, ::cuda::std::int32_t __v, ::cuda::std::int32_t __precision) noexcept
  {
    return __hyperloglog_ns::hllpp_finalizer{__precision}(__z, __v);
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___HYPERLOGLOG_DEFAULT_POLICY_CUH
