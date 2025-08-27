// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items from
//! sequences of data

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_topk.cuh>

CUB_NAMESPACE_BEGIN

//! @rst
//! @brief DeviceTopK provides device-wide, parallel operations for
//!        finding the largest (or smallest) K items from sequences of unordered data
//!        items residing within device-accessible memory.
//!
//! @par Overview
//! TopK problem tries to find the largest (or smallest) K items in an unordered list. A related problem is called
//! [*K selection problem*](https://en.wikipedia.org/wiki/Selection_algorithm), which finds the Kth largest
//! (or smallest) values in a list.
//! DeviceTopK will return K items as results (ordered or unordered). It is
//! based on an algorithm called [*AIR TopK*](https://dl.acm.org/doi/10.1145/3581784.3607062).
//!
//! @par Note
//! We only support the case where the variable K is smaller than the variable N.
//!
//! @par Supported Types
//! DeviceTopK can process all of the built-in C++ numeric primitive types
//! (`unsigned char`, `int`, `double`, etc.) as well as CUDA's `__half`
//! and `__nv_bfloat16` 16-bit floating-point types.
//!
//! @par Stability
//! DeviceTopK provides stable and unstable version.
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceTopK}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{top-k}
//!
//! @endrst

struct DeviceTopK
{
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access input iterator type for writing output values @iterator
  //!
  //! @tparam NumItemsT
  //! Type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //! Type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Pointer to the corresponding input sequence of associated value items
  //!
  //! @param[out] d_values_out
  //!   Pointer to the corresponding output sequence of associated
  //!   value items
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TopKPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumOutItemsT k,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::TopKPairs");

    static constexpr bool select_min = false;
    using offset_t                   = detail::choose_offset_t<NumItemsT>;
    using out_offset_t =
      std::conditional_t<sizeof(offset_t) < sizeof(detail::choose_offset_t<NumOutItemsT>),
                         offset_t,
                         detail::choose_offset_t<NumOutItemsT>>;
    return detail::topk::DispatchTopK<
      KeyInputIteratorT,
      KeyOutputIteratorT,
      ValueInputIteratorT,
      ValueOutputIteratorT,
      offset_t,
      out_offset_t,
      select_min>::Dispatch(d_temp_storage,
                            temp_storage_bytes,
                            d_keys_in,
                            d_keys_out,
                            d_values_in,
                            d_values_out,
                            static_cast<offset_t>(num_items),
                            k,
                            stream);
  }

  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access input iterator type for writing output values @iterator
  //!
  //! @tparam NumItemsT
  //! Type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //! Type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data to find top K
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Pointer to the corresponding input sequence of associated value items
  //!
  //! @param[out] d_values_out
  //!   Pointer to the corresponding output sequence of associated
  //!   value items
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TopKMinPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumOutItemsT k,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::TopKMinPairs");

    static constexpr bool select_min = true;
    using offset_t                   = detail::choose_offset_t<NumItemsT>;
    using out_offset_t =
      std::conditional_t<sizeof(offset_t) < sizeof(detail::choose_offset_t<NumOutItemsT>),
                         offset_t,
                         detail::choose_offset_t<NumOutItemsT>>;
    return detail::topk::DispatchTopK<
      KeyInputIteratorT,
      KeyOutputIteratorT,
      ValueInputIteratorT,
      ValueOutputIteratorT,
      offset_t,
      out_offset_t,
      select_min>::Dispatch(d_temp_storage,
                            temp_storage_bytes,
                            d_keys_in,
                            d_keys_out,
                            d_values_in,
                            d_values_out,
                            static_cast<offset_t>(num_items),
                            k,
                            stream);
  }

  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam NumItemsT
  //! Type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //! Type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT, typename KeyOutputIteratorT, typename NumItemsT, typename NumOutItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TopKKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    NumItemsT num_items,
    NumOutItemsT k,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::TopKKeys");

    static constexpr bool select_min = false;
    using offset_t                   = detail::choose_offset_t<NumItemsT>;
    using out_offset_t =
      std::conditional_t<sizeof(offset_t) < sizeof(detail::choose_offset_t<NumOutItemsT>),
                         offset_t,
                         detail::choose_offset_t<NumOutItemsT>>;
    return detail::topk::
      DispatchTopK<KeyInputIteratorT, KeyOutputIteratorT, NullType*, NullType*, offset_t, out_offset_t, select_min>::
        Dispatch(d_temp_storage,
                 temp_storage_bytes,
                 d_keys_in,
                 d_keys_out,
                 static_cast<NullType*>(nullptr),
                 static_cast<NullType*>(nullptr),
                 static_cast<offset_t>(num_items),
                 k,
                 stream);
  }

  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam NumItemsT
  //! Type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //! Type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data to find top K
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT, typename KeyOutputIteratorT, typename NumItemsT, typename NumOutItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TopKMinKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    NumItemsT num_items,
    NumOutItemsT k,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::TopKMinKeys");

    static constexpr bool select_min = true;
    using offset_t                   = detail::choose_offset_t<NumItemsT>;
    using out_offset_t =
      std::conditional_t<sizeof(offset_t) < sizeof(detail::choose_offset_t<NumOutItemsT>),
                         offset_t,
                         detail::choose_offset_t<NumOutItemsT>>;
    return detail::topk::
      DispatchTopK<KeyInputIteratorT, KeyOutputIteratorT, NullType*, NullType*, offset_t, out_offset_t, select_min>::
        Dispatch(d_temp_storage,
                 temp_storage_bytes,
                 d_keys_in,
                 d_keys_out,
                 static_cast<NullType*>(nullptr),
                 static_cast<NullType*>(nullptr),
                 static_cast<offset_t>(num_items),
                 k,
                 stream);
  }
};

CUB_NAMESPACE_END
