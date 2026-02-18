//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_REF_IMPL_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_REF_IMPL_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/device_reference.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>

#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__cuco/__detail/equal_wrapper.cuh>
#include <cuda/experimental/__cuco/__detail/utils.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#if defined(CUCO_HAS_CUDA_BARRIER)
#  include <cuda/barrier>
#endif

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Three-way insert result enum.
enum class __insert_result : ::cuda::std::int8_t
{
  __continue  = 0,
  __success   = 1,
  __duplicate = 2
};

//!
//! @brief Helper struct to store intermediate bucket probing results.
struct __bucket_probing_results
{
  ::cuda::experimental::cuco::__detail::__equal_result __state; ///< Equal result
  ::cuda::std::int32_t __intra_bucket_index; ///< Intra-bucket index

  //!
  //! @brief Constructs __bucket_probing_results.
  //!
  //! @param __state The three way equality __result
  //! @param __index Intra-bucket index
  _CCCL_DEVICE explicit constexpr __bucket_probing_results(
    ::cuda::experimental::cuco::__detail::__equal_result __state, ::cuda::std::int32_t __index) noexcept
      : __state{__state}
      , __intra_bucket_index{__index}
  {}
};

//!
//! @brief Common device non-owning "ref" implementation class.
//!
//! @note This class should NOT be used directly.
//!
//! @throw If the size of the given __key type is larger than 8 bytes
//! @throw If the given __key type doesn't have unique object representations, i.e.,
//! `::cuda::experimental::cuco::bitwise_comparable_v<Key> == false`
//! @throw If the probing scheme type is not inherited from
//! `::cuda::experimental::cuco::__detail::__probing_scheme_base`
//!
//! @tparam Key Type used for keys. Requires `::cuda::experimental::cuco::is_bitwise_comparable_v<Key>` returning true
//! @tparam Scope The scope in which operations will be performed by individual threads.
//! @tparam KeyEqual Binary callable type used to compare two keys for equality
//! @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for options)
//! @tparam StorageRef Storage ref type
//! @tparam AllowsDuplicates Flag indicating whether duplicate keys are allowed or not
template <class _Key,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          class _StorageRef,
          bool _AllowsDuplicates>
class __open_addressing_ref_impl
{
  static_assert(sizeof(_Key) <= 8, "Container does not support __key types larger than 8 bytes.");

  static_assert(::cuda::experimental::cuco::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of ::cuda::experimental::cuco::is_bitwise_comparable_v<_Key>.");

  static_assert(
    ::cuda::std::is_base_of_v<::cuda::experimental::cuco::__detail::__probing_scheme_base<_ProbingScheme::cg_size>,
                              _ProbingScheme>,
    "ProbingScheme must inherit from ::cuda::experimental::cuco::__detail::__probing_scheme_base");

public:
  using __key_type            = _Key; ///< Key type
  using __probing_scheme_type = _ProbingScheme; ///< Type of probing scheme
  using __hasher              = typename __probing_scheme_type::hasher; ///< Hash function type
  using __storage_ref_type    = _StorageRef; ///< Type of storage ref
  using __bucket_type         = typename __storage_ref_type::__bucket_type; ///< Bucket type
  using __value_type          = typename __storage_ref_type::__value_type; ///< Storage element type
  using __extent_type         = typename __storage_ref_type::__extent_type; ///< Extent type
  using __size_type           = typename __storage_ref_type::__size_type; ///< Probing scheme size type
  using __key_equal           = _KeyEqual; ///< Type of key equality binary callable
  using __iterator            = typename __storage_ref_type::__iterator; ///< Slot iterator type
  using __const_iterator      = typename __storage_ref_type::__const_iterator; ///< Const slot iterator type

  static constexpr auto __cg_size      = __probing_scheme_type::cg_size; ///< Cooperative group size
  static constexpr auto __bucket_size  = __storage_ref_type::__bucket_size; ///< Bucket size
  static constexpr auto __thread_scope = _Scope; ///< CUDA thread scope

private:
  /// Determines if the container is a key/value or key-only store
  static constexpr auto __has_payload = not ::cuda::std::is_same_v<_Key, typename _StorageRef::__value_type>;

  /// Flag indicating whether duplicate keys are allowed or not
  static constexpr auto __allows_duplicates = _AllowsDuplicates;

  // TODO: how to re-enable this check?
  // static_assert(is_bucket_extent_v<typename StorageRef::extent_type>,
  // "Extent is not a valid ::cuda::experimental::cuco::bucket_extent");

  __value_type __empty_slot_sentinel; ///< Sentinel value indicating an empty slot
  ::cuda::experimental::cuco::__detail::__equal_wrapper<__key_type, __key_equal, __allows_duplicates>
    __predicate; ///< Key equality
  __probing_scheme_type __probing_scheme; ///< Probing scheme
  __storage_ref_type __storage_ref; ///< Slot storage ref

public:
  //! @brief Constructs `__open_addressing_ref_impl`.
  //!
  //! @param __empty_slot_sentinel Sentinel indicating an empty slot
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __storage_ref Non-owning ref of slot storage
  _CCCL_HOST_DEVICE explicit constexpr __open_addressing_ref_impl(
    __value_type __empty_slot_sentinel,
    const __key_equal& __predicate,
    const __probing_scheme_type& __probing_scheme,
    __storage_ref_type __storage_ref) noexcept
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __predicate{this->__extract_key(__empty_slot_sentinel), this->__extract_key(__empty_slot_sentinel), __predicate}
      , __probing_scheme{__probing_scheme}
      , __storage_ref{__storage_ref}
  {}

  //! @brief Constructs `__open_addressing_ref_impl`.
  //!
  //! @param __empty_slot_sentinel Sentinel indicating an empty slot
  //! @param __erased_key_sentinel Sentinel indicating an erased __key
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __storage_ref Non-owning ref of slot storage
  _CCCL_HOST_DEVICE explicit constexpr __open_addressing_ref_impl(
    __value_type __empty_slot_sentinel,
    __key_type __erased_key_sentinel,
    const __key_equal& __predicate,
    const __probing_scheme_type& __probing_scheme,
    __storage_ref_type __storage_ref) noexcept
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __predicate{this->__extract_key(__empty_slot_sentinel), __erased_key_sentinel, __predicate}
      , __probing_scheme{__probing_scheme}
      , __storage_ref{__storage_ref}
  {}

  //! @brief Gets the sentinel value used to represent an empty __key slot.
  //!
  //! @return The sentinel value used to represent an empty __key slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __key_type empty_key_sentinel() const noexcept
  {
    return this->__predicate.__empty_sentinel;
  }

  //! @brief Gets the sentinel value used to represent an empty payload slot.
  //!
  //! @return The sentinel value used to represent an empty payload slot
  template <bool _Dummy = true, class _Enable = ::cuda::std::enable_if_t<__has_payload and _Dummy>>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto empty_value_sentinel() const noexcept
  {
    return this->__extract_payload(this->empty_slot_sentinel());
  }

  //! @brief Gets the sentinel value used to represent an erased __key slot.
  //!
  //! @return The sentinel value used to represent an erased __key slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __key_type erased_key_sentinel() const noexcept
  {
    return this->__predicate.__erased_sentinel;
  }

  //! @brief Gets the sentinel used to represent an empty slot.
  //!
  //! @return The sentinel value used to represent an empty slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __value_type empty_slot_sentinel() const noexcept
  {
    return __empty_slot_sentinel;
  }

  //! @brief Returns the function that compares keys for equality.
  //!
  //! @return The key equality predicate
  [[nodiscard]] _CCCL_HOST _CCCL_DEVICE constexpr ::cuda::experimental::cuco::__detail::
    __equal_wrapper<__key_type, __key_equal, __allows_duplicates>
    predicate() const noexcept
  {
    return this->__predicate;
  }

  //! @brief Gets the key comparator.
  //!
  //! @return The comparator used to compare keys
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __key_equal key_eq() const noexcept
  {
    return this->predicate().__equal;
  }

  //! @brief Gets the probing scheme.
  //!
  //! @return The probing scheme used for the container
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __probing_scheme_type probing_scheme() const noexcept
  {
    return __probing_scheme;
  }

  //! @brief Gets the function(s) used to hash keys
  //!
  //! @return The function(s) used to hash keys
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __hasher hash_function() const noexcept
  {
    return this->probing_scheme().hash_function();
  }

  //! @brief Gets the non-owning storage ref.
  //!
  //! @return The non-owning storage ref of the container
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __storage_ref_type storage_ref() const noexcept
  {
    return __storage_ref;
  }

  //!
  //! @brief Gets the maximum number of elements the container can hold.
  //!
  //! @return The maximum number of elements the container can hold
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto capacity() const noexcept
  {
    return __storage_ref.capacity();
  }

  //!
  //! @brief Gets the bucket extent of the current storage.
  //!
  //! @return The bucket extent.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __extent_type extent() const noexcept
  {
    return __storage_ref.extent();
  }

  //!
  //! @brief Returns an iterator to one past the last slot.
  //!
  //! @return An iterator to one past the last slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __iterator end() const noexcept
  {
    return __storage_ref.end();
  }

  //!
  //! @brief Returns an iterator to one past the last slot.
  //!
  //! @return An iterator to one past the last slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __iterator end() noexcept
  {
    return __storage_ref.end();
  }

  //!
  //! @brief Makes a copy of the current device reference using non-owned memory.
  //!
  //! This function is intended to be used to create shared memory copies of small static data
  //! structures, although global memory can be used as well.
  //!
  //! @tparam CG The type of the cooperative thread group
  //!
  //! @param __group The cooperative thread group used to copy the data structure
  //! @param __memory_to_use Array large enough to support `capacity` elements. Object does not take
  //! the ownership of the memory
  template <class CG>
  _CCCL_DEVICE void make_copy(CG g, __value_type* const __memory_to_use) const noexcept
  {
    const auto __num_slots = this->capacity();
#if defined(CUCO_HAS_CUDA_BARRIER)
#  pragma nv_diagnostic push
// Disables `barrier` initialization warning.
#  pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ ::cuda::barrier<::cuda::thread_scope::thread_scope_block> barrier;
#  pragma nv_diagnostic pop
    if (g.thread_rank() == 0)
    {
      init(&barrier, g.size());
    }
    g.sync();

    ::cuda::memcpy_async(g, __memory_to_use, this->storage_ref().data(), sizeof(__value_type) * __num_slots, barrier);

    barrier.arrive_and_wait();
#else
    __value_type const* const __slots_ptr = this->storage_ref().data();
    for (__size_type i = g.thread_rank(); i < __num_slots; i += g.size())
    {
      __memory_to_use[i] = __slots_ptr[i];
    }
    g.sync();
#endif
  }

  //!
  //! @brief Initializes the container storage.
  //!
  //! @note This function synchronizes the group `__tile`.
  //!
  //! @tparam CG The type of the cooperative thread group
  //!
  //! @param __tile The cooperative thread group used to initialize the container
  template <class CG>
  _CCCL_DEVICE constexpr void initialize(CG __tile) noexcept
  {
    auto __tid          = __tile.thread_rank();
    const auto __extent = static_cast<__size_type>(this->extent());

    auto* const __slots_ptr = this->storage_ref().data();
    while (__tid < __extent)
    {
      __slots_ptr[__tid] = this->empty_slot_sentinel();
      __tid += __tile.size();
    }

    __tile.sync();
  }

  //!
  //! @brief Inserts an element.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param __value The element to insert
  //!
  //! @return True if the given element is successfully inserted
  template <class _Value>
  _CCCL_DEVICE bool insert(_Value __value) noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    const auto __val = this->__heterogeneous_value(__value);
    const auto __key = this->__extract_key(__val);

    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      for (auto& __slot_content : __bucket_slots)
      {
        const auto __eq_res =
          this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__yes>(
            __key, this->__extract_key(__slot_content));

        if constexpr (not __allows_duplicates)
        {
          // If the __key is already in the container, return false
          if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__equal)
          {
            return false;
          }
        }
        if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__available)
        {
          const auto __intra_bucket_index = ::cuda::std::distance(__bucket_slots.begin(), &__slot_content);
          switch (__attempt_insert(this->__get_slot_ptr(*__probing_iter, __intra_bucket_index), __slot_content, __val))
          {
            case __insert_result::__duplicate: {
              if constexpr (__allows_duplicates)
              {
                [[fallthrough]];
              }
              else
              {
                return false;
              }
            }
            case __insert_result::__continue:
              continue;
            case __insert_result::__success:
              return true;
          }
        }
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return false;
      }
    }
  }

  //!
  //! @brief Inserts an element.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform group insert
  //! @param __value The element to insert
  //!
  //! @return True if the given element is successfully inserted
  template <bool _SupportsErase, class _Value, class _ParentCG>
  _CCCL_DEVICE bool insert(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group,
                           _Value __value) noexcept
  {
    const auto __val = this->__heterogeneous_value(__value);
    const auto __key = this->__extract_key(__val);
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      const auto [__state, __intra_bucket_index] = [&]() {
        __bucket_probing_results __result{::cuda::experimental::cuco::__detail::__equal_result::__unequal, -1};
        ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
          if (__result.__state == ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
          {
            switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__yes>(
              __key, this->__extract_key(__bucket_slots[i()])))
            {
              case ::cuda::experimental::cuco::__detail::__equal_result::__available:
                __result =
                  __bucket_probing_results{::cuda::experimental::cuco::__detail::__equal_result::__available, i()};
                break;
              case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
                if constexpr (!__allows_duplicates)
                {
                  __result =
                    __bucket_probing_results{::cuda::experimental::cuco::__detail::__equal_result::__equal, i()};
                }
                break;
              }
              default:
                break;
            }
          }
        });
        return __result;
      }();

      if constexpr (not __allows_duplicates)
      {
        // If the __key is already in the container, return false
        if (__group.any(__state == ::cuda::experimental::cuco::__detail::__equal_result::__equal))
        {
          return false;
        }
      }

      const auto __group_contains_available =
        __group.ballot(__state == ::cuda::experimental::cuco::__detail::__equal_result::__available);
      if (__group_contains_available)
      {
        const auto __src_lane = __ffs(__group_contains_available) - 1;
        auto __status         = __insert_result::__continue;
        if (__group.thread_rank() == __src_lane)
        {
          if constexpr (_SupportsErase)
          {
            __status = __attempt_insert(
              this->__get_slot_ptr(*__probing_iter, __intra_bucket_index), __bucket_slots[__intra_bucket_index], __val);
          }
          else
          {
            __status = __attempt_insert(
              this->__get_slot_ptr(*__probing_iter, __intra_bucket_index), this->empty_slot_sentinel(), __val);
          }
        }

        switch (__group.shfl(__status, __src_lane))
        {
          case __insert_result::__success:
            return true;
          case __insert_result::__duplicate: {
            if constexpr (__allows_duplicates)
            {
              [[fallthrough]];
            }
            else
            {
              return false;
            }
          }
          default:
            continue;
        }
      }
      else
      {
        ++__probing_iter;
        if (*__probing_iter == __init_idx)
        {
          return false;
        }
      }
    }
  }

  //!
  //! @brief Inserts the given element into the container.
  //!
  //! @note This API returns a pair consisting of an iterator to the inserted element (or to the
  //! element that prevented the insertion) and a `bool` denoting whether the insertion took place or
  //! not.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param __value The element to insert
  //!
  //! @return a pair consisting of an iterator to the element and a bool indicating whether the
  //! insertion is successful or not.
  template <class _Value>
  _CCCL_DEVICE ::cuda::std::pair<__iterator, bool> insert_and_find(_Value __value) noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    const auto __val      = this->__heterogeneous_value(__value);
    const auto __key      = this->__extract_key(__val);
    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      for (auto i = 0; i < __bucket_size; ++i)
      {
        const auto __eq_res =
          this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__yes>(
            __key, this->__extract_key(__bucket_slots[i]));
        auto* __slot_ptr = this->__get_slot_ptr(*__probing_iter, i);

        // If the __key is already in the container, return false
        if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__equal)
        {
          if constexpr (__has_payload)
          {
            // wait to ensure that the write to the value part also took place
            this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
          }
          return {__iterator{__slot_ptr}, false};
        }
        if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__available)
        {
          switch (this->__attempt_insert_stable(__slot_ptr, __bucket_slots[i], __val))
          {
            case __insert_result::__success: {
              if constexpr (__has_payload)
              {
                // wait to ensure that the write to the value part also took place
                this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
              }
              return {__iterator{__slot_ptr}, true};
            }
            case __insert_result::__duplicate: {
              if constexpr (__has_payload)
              {
                // wait to ensure that the write to the value part also took place
                this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
              }
              return {__iterator{__slot_ptr}, false};
            }
            default:
              continue;
          }
        }
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return {this->end(), false};
      }
    };
  }

  //!
  //! @brief Inserts the given element into the container.
  //!
  //! @note This API returns a pair consisting of an iterator to the inserted element (or to the
  //! element that prevented the insertion) and a `bool` denoting whether the insertion took place or
  //! not.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform group insert_and_find
  //! @param __value The element to insert
  //!
  //! @return a pair consisting of an iterator to the element and a bool indicating whether the
  //! insertion is successful or not.
  template <class _Value, class _ParentCG>
  _CCCL_DEVICE ::cuda::std::pair<__iterator, bool>
  insert_and_find(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group, _Value __value) noexcept
  {
    const auto __val = this->__heterogeneous_value(__value);
    const auto __key = this->__extract_key(__val);
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      const auto [__state, __intra_bucket_index] = [&]() {
        __bucket_probing_results __result{::cuda::experimental::cuco::__detail::__equal_result::__unequal, -1};
        ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
          if (__result.__state == ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
          {
            auto __res =
              this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__yes>(
                __key, this->__extract_key(__bucket_slots[i()]));
            if (__res != ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
            {
              __result = __bucket_probing_results{__res, i()};
            }
          }
        });
        return __result;
      }();

      auto* __slot_ptr = this->__get_slot_ptr(*__probing_iter, __intra_bucket_index);

      // If the __key is already in the container, return false
      const auto __group_finds_equal =
        __group.ballot(__state == ::cuda::experimental::cuco::__detail::__equal_result::__equal);
      if (__group_finds_equal)
      {
        const auto __src_lane = __ffs(__group_finds_equal) - 1;
        const auto __res      = __group.shfl(reinterpret_cast<intptr_t>(__slot_ptr), __src_lane);
        if (__group.thread_rank() == __src_lane)
        {
          if constexpr (__has_payload)
          {
            // wait to ensure that the write to the value part also took place
            this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
          }
        }
        __group.sync();
        return {__iterator{reinterpret_cast<__value_type*>(__res)}, false};
      }

      const auto __group_contains_available =
        __group.ballot(__state == ::cuda::experimental::cuco::__detail::__equal_result::__available);
      if (__group_contains_available)
      {
        const auto __src_lane = __ffs(__group_contains_available) - 1;
        const auto __res      = __group.shfl(reinterpret_cast<intptr_t>(__slot_ptr), __src_lane);
        const auto __status   = [&, target_idx = __intra_bucket_index]() {
          if (__group.thread_rank() != __src_lane)
          {
            return __insert_result::__continue;
          }
          return this->__attempt_insert_stable(__slot_ptr, __bucket_slots[target_idx], __val);
        }();

        switch (__group.shfl(__status, __src_lane))
        {
          case __insert_result::__success: {
            if (__group.thread_rank() == __src_lane)
            {
              if constexpr (__has_payload)
              {
                // wait to ensure that the write to the value part also took place
                this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
              }
            }
            __group.sync();
            return {__iterator{reinterpret_cast<__value_type*>(__res)}, true};
          }
          case __insert_result::__duplicate: {
            if (__group.thread_rank() == __src_lane)
            {
              if constexpr (__has_payload)
              {
                // wait to ensure that the write to the value part also took place
                this->__wait_for_payload(__slot_ptr->second, this->empty_value_sentinel());
              }
            }
            __group.sync();
            return {__iterator{reinterpret_cast<__value_type*>(__res)}, false};
          }
          default:
            continue;
        }
      }
      else
      {
        ++__probing_iter;
        if (*__probing_iter == __init_idx)
        {
          return {this->end(), false};
        }
      }
    }
  }

  //!
  //! @brief Erases an element.
  //!
  //! @tparam ProbeKey Input type which is convertible to '__key_type'
  //!
  //! @param __key The element to erase
  //!
  //! @return True if the given element is successfully erased
  template <class _ProbeKey>
  _CCCL_DEVICE bool erase(_ProbeKey __key) noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      for (auto& __slot_content : __bucket_slots)
      {
        const auto __eq_res =
          this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
            __key, this->__extract_key(__slot_content));

        // Key doesn't exist, return false
        if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__empty)
        {
          return false;
        }
        // Key __exists, return true if successfully deleted
        if (__eq_res == ::cuda::experimental::cuco::__detail::__equal_result::__equal)
        {
          const auto __intra_bucket_index = ::cuda::std::distance(__bucket_slots.begin(), &__slot_content);
          switch (__attempt_insert_stable(
            this->__get_slot_ptr(*__probing_iter, __intra_bucket_index), __slot_content, this->__erased_slot_sentinel()))
          {
            case __insert_result::__success:
              return true;
            case __insert_result::__duplicate:
              return false;
            default:
              continue;
          }
        }
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return false;
      }
    }
  }

  //!
  //! @brief Erases an element.
  //!
  //! @tparam ProbeKey Input type which is convertible to '__key_type'
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform group erase
  //! @param __key The element to erase
  //!
  //! @return True if the given element is successfully erased
  template <class _ProbeKey, class _ParentCG>
  _CCCL_DEVICE bool erase(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group,
                          _ProbeKey __key) noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      const auto [__state, __intra_bucket_index] = [&]() {
        __bucket_probing_results __result{::cuda::experimental::cuco::__detail::__equal_result::__unequal, -1};
        ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
          if (__result.__state == ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
          {
            auto __res = this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
              __key, this->__extract_key(__bucket_slots[i()]));
            if (__res != ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
            {
              __result = __bucket_probing_results{__res, i()};
            }
          }
        });
        return __result;
      }();

      const auto __group_contains_equal =
        __group.ballot(__state == ::cuda::experimental::cuco::__detail::__equal_result::__equal);
      if (__group_contains_equal)
      {
        const auto __src_lane = __ffs(__group_contains_equal) - 1;
        const auto __status =
          (__group.thread_rank() == __src_lane)
            ? __attempt_insert_stable(this->__get_slot_ptr(*__probing_iter, __intra_bucket_index),
                                      __bucket_slots[__intra_bucket_index],
                                      this->__erased_slot_sentinel())
            : __insert_result::__continue;

        switch (__group.shfl(__status, __src_lane))
        {
          case __insert_result::__success:
            return true;
          case __insert_result::__duplicate:
            return false;
          default:
            continue;
        }
      }

      // Key doesn't exist, return false
      if (__group.any(__state == ::cuda::experimental::cuco::__detail::__equal_result::__empty))
      {
        return false;
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return false;
      }
    }
  }

  //!
  //! @brief Indicates whether the probe __key `__key` was inserted into the container.
  //!
  //! @note If the probe __key `__key` was inserted into the container, returns true. Otherwise, returns
  //! false.
  //!
  //! @tparam ProbeKey Probe __key type
  //!
  //! @param __key The __key to search for
  //!
  //! @return A boolean indicating whether the probe __key is present
  template <class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE bool contains(_ProbeKey __key) const noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      // TODO atomic_ref::load if insert operator is present
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      for (auto i = 0; i < __bucket_size; ++i)
      {
        switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
          __key, this->__extract_key(__bucket_slots[i])))
        {
          case ::cuda::experimental::cuco::__detail::__equal_result::__unequal:
            continue;
          case ::cuda::experimental::cuco::__detail::__equal_result::__empty:
            return false;
          case ::cuda::experimental::cuco::__detail::__equal_result::__equal:
            return true;
        }
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return false;
      }
    }
  }

  //!
  //! @brief Indicates whether the probe __key `__key` was inserted into the container.
  //!
  //! @note If the probe __key `__key` was inserted into the container, returns true. Otherwise, returns
  //! false.
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform group contains
  //! @param __key The __key to search for
  //!
  //! @return A boolean indicating whether the probe __key is present
  template <class _ProbeKey, class _ParentCG>
  [[nodiscard]] _CCCL_DEVICE bool
  contains(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      const auto __state = [&]() {
        auto __res = ::cuda::experimental::cuco::__detail::__equal_result::__unequal;
        for (auto i = 0; i < __bucket_size; ++i)
        {
          __res = this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
            __key, this->__extract_key(__bucket_slots[i]));
          if (__res != ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
          {
            return __res;
          }
        }
        return __res;
      }();

      if (__group.any(__state == ::cuda::experimental::cuco::__detail::__equal_result::__equal))
      {
        return true;
      }
      if (__group.any(__state == ::cuda::experimental::cuco::__detail::__equal_result::__empty))
      {
        return false;
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return false;
      }
    }
  }

  //!
  //! @brief Finds an element in the container with __key equi__valent to the probe __key.
  //!
  //! @note Returns a un-incrementable input iterator to the element whose __key is equi__valent to
  //! `__key`. If no such element __exists, returns `end()`.
  //!
  //! @tparam ProbeKey Probe __key type
  //!
  //! @param __key The __key to search for
  //!
  //! @return An iterator to the position at which the equi__valent __key is stored
  template <class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE __iterator find(_ProbeKey __key) const noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      // TODO atomic_ref::load if insert operator is present
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      for (auto i = 0; i < __bucket_size; ++i)
      {
        switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
          __key, this->__extract_key(__bucket_slots[i])))
        {
          case ::cuda::experimental::cuco::__detail::__equal_result::__empty: {
            return this->end();
          }
          case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
            return __iterator{this->__get_slot_ptr(*__probing_iter, i)};
          }
          default:
            continue;
        }
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return this->end();
      }
    }
  }

  //!
  //! @brief Finds an element in the container with __key equi__valent to the probe __key.
  //!
  //! @note Returns a un-incrementable input iterator to the element whose __key is equi__valent to
  //! `__key`. If no such element __exists, returns `end()`.
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform this operation
  //! @param __key The __key to search for
  //!
  //! @return An iterator to the position at which the equi__valent __key is stored
  template <class _ProbeKey, class _ParentCG>
  [[nodiscard]] _CCCL_DEVICE __iterator
  find(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      const auto __bucket_slots = __storage_ref[*__probing_iter];

      const auto [__state, __intra_bucket_index] = [&]() {
        __bucket_probing_results __result{::cuda::experimental::cuco::__detail::__equal_result::__unequal, -1};
        ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
          if (__result.__state == ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
          {
            auto __res = this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
              __key, this->__extract_key(__bucket_slots[i()]));
            if (__res != ::cuda::experimental::cuco::__detail::__equal_result::__unequal)
            {
              __result = __bucket_probing_results{__res, i()};
            }
          }
        });
        return __result;
      }();

      // Find a match for the probe __key, thus return an iterator to the entry
      const auto __group_finds_match =
        __group.ballot(__state == ::cuda::experimental::cuco::__detail::__equal_result::__equal);
      if (__group_finds_match)
      {
        const auto __src_lane = __ffs(__group_finds_match) - 1;
        const auto __res      = __group.shfl(
          reinterpret_cast<intptr_t>(this->__get_slot_ptr(*__probing_iter, __intra_bucket_index)), __src_lane);
        return __iterator{reinterpret_cast<__value_type*>(__res)};
      }

      // Find an empty slot, meaning that the probe __key isn't present in the container
      if (__group.any(__state == ::cuda::experimental::cuco::__detail::__equal_result::__empty))
      {
        return this->end();
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return this->end();
      }
    }
  }

  //!
  //! @brief Counts the occurrence of a given __key contained in the container
  //!
  //! @tparam ProbeKey Probe __key type
  //!
  //! @param __key The __key to __count for
  //!
  //! @return Number of occurrences found by the current thread
  template <class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE __size_type __count(_ProbeKey __key) const noexcept
  {
    if constexpr (not __allows_duplicates)
    {
      return static_cast<__size_type>(this->contains(__key));
    }
    else
    {
      auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
      const auto __init_idx = *__probing_iter;
      __size_type __count   = 0;

      while (true)
      {
        const auto __bucket_slots                    = __storage_ref[*__probing_iter];
        ::cuda::std::int32_t __equals[__bucket_size] = {0};
        bool empty_found                             = false;

        ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
          const auto __result =
            __predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
              __key, this->__extract_key(__bucket_slots[i()]));
          __equals[i()] = (__result == ::cuda::experimental::cuco::__detail::__equal_result::__equal);
          if (__result == ::cuda::experimental::cuco::__detail::__equal_result::__empty)
          {
            empty_found = true;
          }
        });

        __count += thrust::reduce(thrust::seq, __equals, __equals + __bucket_size);

        if (empty_found)
        {
          return __count;
        }

        ++__probing_iter;
        if (*__probing_iter == __init_idx)
        {
          return __count;
        }
      }
    }
  }

  //!
  //! @brief Counts the occurrence of a given __key contained in the container
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform group count
  //! @param __key The __key to __count for
  //!
  //! @return Number of occurrences found by the current thread
  template <class _ProbeKey, class _ParentCG>
  [[nodiscard]] _CCCL_DEVICE __size_type
  __count(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;
    __size_type __count   = 0;

    while (true)
    {
      const auto __bucket_slots                    = __storage_ref[*__probing_iter];
      ::cuda::std::int32_t __equals[__bucket_size] = {0};
      bool empty_found                             = false;

      ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
        const auto __result = __predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
          __key, this->__extract_key(__bucket_slots[i()]));
        __equals[i()] = (__result == ::cuda::experimental::cuco::__detail::__equal_result::__equal);
        if (__result == ::cuda::experimental::cuco::__detail::__equal_result::__empty)
        {
          empty_found = true;
        }
      });

      __count += thrust::reduce(thrust::seq, __equals, __equals + __bucket_size);

      if (__group.any(empty_found))
      {
        return __count;
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return __count;
      }
    }
  }

  //!
  //! @brief Retrieves all the slots corresponding to all keys in the range `[__input_probe_begin,
  //! __input_probe_end)`.
  //!
  //! If __key `k = *(first + i)` __exists in the container, copies `k` to `__output_probe` and associated
  //! slot contents to `__output_match`, respectively. The output order is unspecified.
  //!
  //! Behavior is undefined if the size of the output range exceeds the number of retrieved slots.
  //! Use `__count()` to determine the size of the output range.
  //!
  //! @tparam BlockSize Size of the thread __block this operation is executed in
  //! @tparam InputProbeIt Device accessible input iterator
  //! @tparam OutputProbeIt Device accessible input iterator whose `__value_type` is
  //! convertible to the `InputProbeIt`'s `__value_type`
  //! @tparam OutputMatchIt Device accessible input iterator whose `__value_type` is
  //! convertible to the container's `__value_type`
  //! @tparam AtomicCounter Integral atomic counter type that follows the same semantics as
  //! `::cuda::(std::)atomic(_ref)`
  //!
  //! @param __block Thread __block this operation is executed in
  //! @param __input_probe_begin Beginning of the input sequence of keys
  //! @param __input_probe_end End of the input sequence of keys
  //! @param __output_probe Beginning of the sequence of keys corresponding to matching elements in
  //! `__output_match`
  //! @param __output_match Beginning of the sequence of matching elements
  //! @param __atomic_counter Atomic object of integral type that is used to __count the
  //! number of output elements
  template <int BlockSize, class InputProbeIt, class OutputProbeIt, class OutputMatchIt, class AtomicCounter>
  _CCCL_DEVICE void retrieve(
    const ::cooperative_groups::thread_block& __block,
    InputProbeIt __input_probe_begin,
    InputProbeIt __input_probe_end,
    OutputProbeIt __output_probe,
    OutputMatchIt __output_match,
    AtomicCounter& __atomic_counter) const
  {
    constexpr auto is_outer = false;
    const auto __num =
      ::cuda::experimental::cuco::__detail::__distance(__input_probe_begin, __input_probe_end); // TODO include
    const auto always_true_stencil = thrust::constant_iterator<bool>(true);
    const auto identity_predicate  = ::cuda::std::identity{};
    this->__retrieve_impl<is_outer, BlockSize>(
      __block,
      __input_probe_begin,
      __num,
      always_true_stencil,
      identity_predicate,
      __output_probe,
      __output_match,
      __atomic_counter);
  }

  //!
  //! @brief Retrieves all the slots corresponding to all keys in the range `[__input_probe_begin,
  //! __input_probe_end)`.
  //!
  //! If __key `k = *(first + i)` __exists in the container, copies `k` to `__output_probe` and associated
  //! slot contents to `__output_match`, respectively. The output order is unspecified.
  //!
  //! Behavior is undefined if the size of the output range exceeds the number of retrieved slots.
  //! Use `__count()` to determine the size of the output range.
  //!
  //! If a __key `k` has no matches in the container, then `{__key, empty_slot_sentinel}` will be added
  //! to the output sequence.
  //!
  //! @tparam BlockSize Size of the thread __block this operation is executed in
  //! @tparam InputProbeIt Device accessible input iterator
  //! @tparam OutputProbeIt Device accessible input iterator whose `__value_type` is
  //! convertible to the `InputProbeIt`'s `__value_type`
  //! @tparam OutputMatchIt Device accessible input iterator whose `__value_type` is
  //! convertible to the container's `__value_type`
  //! @tparam AtomicCounter Integral atomic counter type that follows the same semantics as
  //! `::cuda::(std::)atomic(_ref)`
  //!
  //! @param __block Thread __block this operation is executed in
  //! @param __input_probe_begin Beginning of the input sequence of keys
  //! @param __input_probe_end End of the input sequence of keys
  //! @param __output_probe Beginning of the sequence of keys corresponding to matching elements in
  //! `__output_match`
  //! @param __output_match Beginning of the sequence of matching elements
  //! @param __atomic_counter Atomic object of integral type that is used to __count the
  //! number of output elements
  template <int BlockSize, class InputProbeIt, class OutputProbeIt, class OutputMatchIt, class AtomicCounter>
  _CCCL_DEVICE void retrieve_outer(
    const ::cooperative_groups::thread_block& __block,
    InputProbeIt __input_probe_begin,
    InputProbeIt __input_probe_end,
    OutputProbeIt __output_probe,
    OutputMatchIt __output_match,
    AtomicCounter& __atomic_counter) const
  {
    constexpr auto is_outer = true;
    const auto __num =
      ::cuda::experimental::cuco::__detail::__distance(__input_probe_begin, __input_probe_end); // TODO include
    const auto always_true_stencil = thrust::constant_iterator<bool>(true);
    const auto identity_predicate  = ::cuda::std::identity{};
    this->__retrieve_impl<is_outer, BlockSize>(
      __block,
      __input_probe_begin,
      __num,
      always_true_stencil,
      identity_predicate,
      __output_probe,
      __output_match,
      __atomic_counter);
  }

  //!
  //! @brief Retrieves all the slots corresponding to all keys in the range `[__input_probe_begin,
  //! __input_probe_end)` if `__pred` of the corresponding __stencil returns true.
  //!
  //! If __key `k = *(first + i)` __exists in the container and `__pred( *(__stencil + i) )` returns true,
  //! copies `k` to `__output_probe` and associated slot contents to `__output_match`,
  //! respectively. The output order is unspecified.
  //!
  //! Behavior is undefined if the size of the output range exceeds the number of retrieved slots.
  //! Use `__count()` to determine the size of the output range.
  //!
  //! @tparam BlockSize Size of the thread __block this operation is executed in
  //! @tparam InputProbeIt Device accessible input iterator
  //! @tparam StencilIt Device accessible random access iterator whose __value_type is
  //! convertible to Predicate's argument type
  //! @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
  //! and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
  //! @tparam OutputProbeIt Device accessible input iterator whose `__value_type` is
  //! convertible to the `InputProbeIt`'s `__value_type`
  //! @tparam OutputMatchIt Device accessible input iterator whose `__value_type` is
  //! convertible to the container's `__value_type`
  //! @tparam AtomicCounter Integral atomic counter type that follows the same semantics as
  //! `::cuda::(std::)atomic(_ref)`
  //!
  //! @param __block Thread __block this operation is executed in
  //! @param __input_probe_begin Beginning of the input sequence of keys
  //! @param __input_probe_end End of the input sequence of keys
  //! @param __stencil Beginning of the __stencil sequence
  //! @param __pred Predicate to test on every element in the range `[__stencil, __stencil + __num)`
  //! @param __output_probe Beginning of the sequence of keys corresponding to matching elements in
  //! `__output_match`
  //! @param __output_match Beginning of the sequence of matching elements
  //! @param __atomic_counter Atomic object of integral type that is used to __count the
  //! number of output elements
  template <int BlockSize,
            class InputProbeIt,
            class StencilIt,
            class Predicate,
            class OutputProbeIt,
            class OutputMatchIt,
            class AtomicCounter>
  _CCCL_DEVICE void retrieve_if(
    const ::cooperative_groups::thread_block& __block,
    InputProbeIt __input_probe_begin,
    InputProbeIt __input_probe_end,
    StencilIt __stencil,
    Predicate __pred,
    OutputProbeIt __output_probe,
    OutputMatchIt __output_match,
    AtomicCounter& __atomic_counter) const
  {
    constexpr auto is_outer = false;
    const auto __num        = ::cuda::experimental::cuco::__detail::__distance(__input_probe_begin, __input_probe_end);
    this->__retrieve_impl<is_outer, BlockSize>(
      __block, __input_probe_begin, __num, __stencil, __pred, __output_probe, __output_match, __atomic_counter);
  }

  //!
  //! @brief Retrieves all the slots corresponding to all keys in the range `[__input_probe_begin,
  //! __input_probe_end)`.
  //!
  //! If __key `k = *(first + i)` __exists in the container, copies `k` to `__output_probe` and associated
  //! slot contents to `__output_match`, respectively. The output order is unspecified.
  //!
  //! Behavior is undefined if the size of the output range exceeds the number of retrieved slots.
  //! Use `__count()` to determine the size of the output range.
  //!
  //! If `IsOuter == true` and a __key `k` has no matches in the container, then `{__key,
  //! empty_slot_sentinel}` will be added to the output sequence.
  //!
  //! @tparam IsOuter Flag indicating if an inner or outer retrieve operation should be performed
  //! @tparam BlockSize Size of the thread __block this operation is executed in
  //! @tparam InputProbeIt Device accessible input iterator
  //! @tparam StencilIt Device accessible random access iterator whose __value_type is
  //! convertible to Predicate's argument type
  //! @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
  //! and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
  //! @tparam OutputProbeIt Device accessible input iterator whose `__value_type` is
  //! convertible to the `InputProbeIt`'s `__value_type`
  //! @tparam OutputMatchIt Device accessible input iterator whose `__value_type` is
  //! convertible to the container's `__value_type`
  //! @tparam AtomicCounter Integral atomic type that follows the same semantics as
  //! `::cuda::(std::)atomic(_ref)`
  //!
  //! @param __block Thread __block this operation is executed in
  //! @param __input_probe Beginning of the input sequence of keys
  //! @param __n Number of input keys
  //! @param __stencil Beginning of the __stencil sequence
  //! @param __pred Predicate to test on every element in the range `[__stencil, __stencil + __num)`
  //! @param __output_probe Beginning of the sequence of keys corresponding to matching elements in
  //! `__output_match`
  //! @param __output_match Beginning of the sequence of matching elements
  //! @param __atomic_counter Atomic object of integral type that is used to __count the
  //! number of output elements
  template <bool IsOuter,
            int BlockSize,
            class InputProbeIt,
            class StencilIt,
            class Predicate,
            class OutputProbeIt,
            class OutputMatchIt,
            class AtomicCounter>
  _CCCL_DEVICE void __retrieve_impl(
    const ::cooperative_groups::thread_block& __block,
    InputProbeIt __input_probe,
    ::cuda::experimental::cuco::__detail::__index_type __num,
    StencilIt __stencil,
    Predicate __pred,
    OutputProbeIt __output_probe,
    OutputMatchIt __output_match,
    AtomicCounter& __atomic_counter) const
  {
    namespace cg = cooperative_groups;

    if (__num == 0)
    {
      return;
    }

    using probe_type = typename ::cuda::std::iterator_traits<InputProbeIt>::value_type;

    // tuning parameter
    constexpr auto __buffer_multiplier = 1;
    static_assert(__buffer_multiplier > 0);

    constexpr auto __probing_tile_size  = __cg_size;
    constexpr auto __flushing_tile_size = ::cuda::experimental::cuco::__detail::__warp_size();
    static_assert(__flushing_tile_size >= __probing_tile_size);

    constexpr auto __num_flushing_tiles   = BlockSize / __flushing_tile_size;
    constexpr auto __max_matches_per_step = __flushing_tile_size * __bucket_size;
    constexpr auto __buffer_size          = __buffer_multiplier * __max_matches_per_step + __flushing_tile_size;

    const auto __flushing_tile = cg::tiled_partition<__flushing_tile_size, cg::thread_block>(__block);
    const auto __probing_tile  = cg::tiled_partition<__probing_tile_size, cg::thread_block>(__block);

    const auto __flushing_tile_id = __flushing_tile.meta_group_rank();
    const auto __stride           = __probing_tile.meta_group_size();
    auto __idx                    = __probing_tile.meta_group_rank();

    __shared__ ::cuda::std::pair<probe_type, __value_type> __buffers[__num_flushing_tiles][__buffer_size];
    __shared__ ::cuda::std::int32_t __counters[__num_flushing_tiles];

    if (__flushing_tile.thread_rank() == 0)
    {
      __counters[__flushing_tile_id] = 0;
    }
    __flushing_tile.sync();

    auto __flush_buffers = [&](auto __tile) {
      __size_type __offset = 0;
      const auto __count   = __counters[__flushing_tile_id];
      const auto __rank    = __tile.thread_rank();
      if (__rank == 0)
      {
        __offset = __atomic_counter.fetch_add(__count, ::cuda::memory_order_relaxed);
      }
      __offset = __tile.shfl(__offset, 0);

      // __flush_buffers
      for (auto i = __rank; i < __count; i += __tile.size())
      {
        *(__output_probe + __offset + i) = __buffers[__flushing_tile_id][i].first;
        *(__output_match + __offset + i) = __buffers[__flushing_tile_id][i].second;
      }
    };

    while (__flushing_tile.any(__idx < __num))
    {
      bool __active_flag                = __idx < __num and __pred(*(__stencil + __idx));
      const auto __active_flushing_tile = cg::binary_partition<__flushing_tile_size>(__flushing_tile, __active_flag);

      if (__active_flag)
      {
        // perform probing
        // make sure the __flushing_tile is converged at this point to get a coalesced load
        const auto __probe_key = *(__input_probe + __idx);

        auto __probing_iter =
          __probing_scheme.template make_iterator<__bucket_size>(__probing_tile, __probe_key, __storage_ref.extent());
        const auto __init_idx = *__probing_iter;

        bool __running                      = true;
        [[maybe_unused]] bool __found_match = false;

        bool __equals[__bucket_size];
        ::cuda::std::uint32_t __exists[__bucket_size];

        while (__active_flushing_tile.any(__running))
        {
          if (__running)
          {
            // TODO atomic_ref::load if insert operator is present
            const auto __bucket_slots = this->__storage_ref[*__probing_iter];

            ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
              __equals[i()] = false;
              if (__running)
              {
                // inspect slot content
                switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
                  __probe_key, this->__extract_key(__bucket_slots[i()])))
                {
                  case ::cuda::experimental::cuco::__detail::__equal_result::__empty: {
                    __running = false;
                    break;
                  }
                  case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
                    if constexpr (!__allows_duplicates)
                    {
                      __running = false;
                    }
                    __equals[i()] = true;
                    break;
                  }
                  default: {
                    break;
                  }
                }
              }
            });

            __probing_tile.sync();
            __running = __probing_tile.all(__running);
            ::cuda::static_for<__bucket_size>([&](auto i) {
              __exists[i()] = __probing_tile.ballot(__equals[i()]);
            });

            // Fill the buffer if any matching keys are found
            const auto lane_id = __probing_tile.thread_rank();
            if (thrust::any_of(thrust::seq, __exists, __exists + __bucket_size, ::cuda::std::identity{}))
            {
              if constexpr (IsOuter)
              {
                __found_match = true;
              }

              ::cuda::std::int32_t num_matches[__bucket_size];

              ::cuda::static_for<__bucket_size>([&](auto i) {
                num_matches[i()] = __popc(__exists[i()]);
              });

              ::cuda::std::int32_t __output_idx;
              if (lane_id == 0)
              {
                const auto total_matches = thrust::reduce(thrust::seq, num_matches, num_matches + __bucket_size);
                auto ref =
                  ::cuda::atomic_ref<::cuda::std::int32_t, ::cuda::thread_scope_block>{__counters[__flushing_tile_id]};
                __output_idx = ref.fetch_add(total_matches, ::cuda::memory_order_relaxed);
              }
              __output_idx = __probing_tile.shfl(__output_idx, 0);

              ::cuda::std::int32_t matches_offset = 0;
              ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
                if (__equals[i()])
                {
                  const auto lane_offset =
                    ::cuda::experimental::cuco::__detail::__count_least_significant_bits(__exists[i()], lane_id);
                  __buffers[__flushing_tile_id][__output_idx + matches_offset + lane_offset] = {
                    __probe_key, __bucket_slots[i()]};
                }
                matches_offset += num_matches[i()];
              });
            }
            // Special handling for outer cases where no match is found
            if constexpr (IsOuter)
            {
              if (!__running)
              {
                if (!__found_match and lane_id == 0)
                {
                  auto ref = ::cuda::atomic_ref<::cuda::std::int32_t, ::cuda::thread_scope_block>{
                    __counters[__flushing_tile_id]};
                  const auto __output_idx                     = ref.fetch_add(1, ::cuda::memory_order_relaxed);
                  __buffers[__flushing_tile_id][__output_idx] = {__probe_key, this->empty_slot_sentinel()};
                }
              }
            }
          } // if __running

          __active_flushing_tile.sync();
          // if the buffer has not enough empty slots for the next iteration
          if (__counters[__flushing_tile_id] > (__buffer_size - __max_matches_per_step))
          {
            __flush_buffers(__active_flushing_tile);
            __active_flushing_tile.sync();

            // reset buffer counter
            if (__active_flushing_tile.thread_rank() == 0)
            {
              __counters[__flushing_tile_id] = 0;
            }
            __active_flushing_tile.sync();
          }

          // onto the next probing bucket
          ++__probing_iter;
          if (*__probing_iter == __init_idx)
          {
            __running = false;
          }
        } // while __running
      } // if __active_flag

      // onto the next __key
      __idx += __stride;
    }

    __flushing_tile.sync();
    // entire flusing_tile has finished; flush remaining elements
    if (__counters[__flushing_tile_id] > 0)
    {
      __flush_buffers(__flushing_tile);
    }
  }

  //!
  //! @brief For a given __key, applies the function object `callback_op` to the copy of all
  //! corresponding matches found in the container.
  //!
  //! @note The return value of `callback_op`, if any, is ignored.
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam CallbackOp Type of unary callback function object
  //!
  //! @param __key The __key to search for
  //! @param __callback_op Function to apply to every matched slot
  template <class _ProbeKey, class _CallbackOp>
  _CCCL_DEVICE void for_each(_ProbeKey __key, _CallbackOp&& __callback_op) const noexcept
  {
    static_assert(__cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto __probing_iter   = __probing_scheme.template make_iterator<__bucket_size>(__key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;

    while (true)
    {
      // TODO atomic_ref::load if insert operator is present
      const auto __bucket_slots = this->__storage_ref[*__probing_iter];

      bool should_return = false;
      ::cuda::static_for<__bucket_size>([&] _CCCL_DEVICE(auto i) {
        if (!should_return)
        {
          switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
            __key, this->__extract_key(__bucket_slots[i()])))
          {
            case ::cuda::experimental::cuco::__detail::__equal_result::__empty: {
              should_return = true;
              break;
            }
            case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
              __callback_op(__bucket_slots[i()]);
              break;
            }
            default:
              break;
          }
        }
      });
      if (should_return)
      {
        return;
      }
      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return;
      }
    }
  }

  //!
  //! @brief For a given __key, applies the function object `callback_op` to the copy of all
  //! corresponding matches found in the container.
  //!
  //! @note This function uses cooperative group semantics, meaning that any thread may call the
  //! callback if it finds a matching element. If multiple elements are found within the same group,
  //! each thread with a match will call the callback with its associated element.
  //!
  //! @note The return value of `callback_op`, if any, is ignored.
  //!
  //! @note Synchronizing `__group` within `callback_op` is undefined behavior.
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam CallbackOp Type of unary callback function object
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform this operation
  //! @param __key The __key to search for
  //! @param __callback_op Function to apply to every matched slot
  template <class _ProbeKey, class _CallbackOp, class _ParentCG>
  _CCCL_DEVICE void for_each(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group,
                             _ProbeKey __key,
                             _CallbackOp&& __callback_op) const noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;
    bool empty            = false;

    while (true)
    {
      // TODO atomic_ref::load if insert operator is present
      const auto __bucket_slots = this->__storage_ref[*__probing_iter];

      for (::cuda::std::int32_t i = 0; i < __bucket_size and !empty; ++i)
      {
        switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
          __key, this->__extract_key(__bucket_slots[i])))
        {
          case ::cuda::experimental::cuco::__detail::__equal_result::__empty: {
            empty = true;
            continue;
          }
          case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
            __callback_op(__bucket_slots[i]);
            continue;
          }
          default: {
            continue;
          }
        }
      }
      if (__group.any(empty))
      {
        return;
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return;
      }
    }
  }

  //!
  //! @brief Applies the function object `callback_op` to the copy of every slot in the container
  //! with __key equi__valent to the probe __key and can additionally perform work that requires
  //! synchronizing the Cooperative Group performing this operation.
  //!
  //! @note This function uses cooperative group semantics, meaning that any thread may call the
  //! callback if it finds a matching element. If multiple elements are found within the same group,
  //! each thread with a match will call the callback with its associated element.
  //!
  //! @note Synchronizing `__group` within `callback_op` is undefined behavior.
  //!
  //! @note The return value of `callback_op`, if any, is ignored.
  //!
  //! @note The `sync_op` function can be used to perform work that requires synchronizing threads in
  //! `__group` in-between probing steps, where the number of probing steps performed between
  //! synchronization points is capped by `__bucket_size * __cg_size`. The functor will be called right
  //! after the current probing bucket has been traversed.
  //!
  //! @tparam ProbeKey Probe __key type
  //! @tparam CallbackOp Type of unary callback function object
  //! @tparam SyncOp Type of function object which accepts the current `__group` object
  //! @tparam ParentCG Type of parent Cooperative Group
  //!
  //! @param __group The Cooperative Group used to perform this operation
  //! @param __key The __key to search for
  //! @param __callback_op Function to apply to every matched slot
  //! @param __sync_op Function that is allowed to synchronize `__group` in-between probing buckets
  template <class _ProbeKey, class _CallbackOp, class _SyncOp, class _ParentCG>
  _CCCL_DEVICE void for_each(::cooperative_groups::thread_block_tile<__cg_size, _ParentCG> __group,
                             _ProbeKey __key,
                             _CallbackOp&& __callback_op,
                             _SyncOp&& __sync_op) const noexcept
  {
    auto __probing_iter =
      __probing_scheme.template make_iterator<__bucket_size>(__group, __key, __storage_ref.extent());
    const auto __init_idx = *__probing_iter;
    bool empty            = false;

    while (true)
    {
      // TODO atomic_ref::load if insert operator is present
      const auto __bucket_slots = this->__storage_ref[*__probing_iter];

      for (::cuda::std::int32_t i = 0; i < __bucket_size and !empty; ++i)
      {
        switch (this->__predicate.template operator()<::cuda::experimental::cuco::__detail::__is_insert::__no>(
          __key, this->__extract_key(__bucket_slots[i])))
        {
          case ::cuda::experimental::cuco::__detail::__equal_result::__empty: {
            empty = true;
            continue;
          }
          case ::cuda::experimental::cuco::__detail::__equal_result::__equal: {
            __callback_op(__bucket_slots[i]);
            continue;
          }
          default: {
            continue;
          }
        }
      }
      __sync_op(__group);
      if (__group.any(empty))
      {
        return;
      }

      ++__probing_iter;
      if (*__probing_iter == __init_idx)
      {
        return;
      }
    }
  }

  //!
  //! @brief Gets a pointer to the slot at the given probing index and intra-bucket index.
  //!
  //! @param __probing_idx The current probing index
  //! @param __intra_bucket_idx The index within the bucket (0 for flat storage)
  //! @return Pointer to the slot
  _CCCL_DEVICE __value_type*
  __get_slot_ptr(__size_type __probing_idx, ::cuda::std::int32_t __intra_bucket_idx) const noexcept
  {
    return __storage_ref.data() + __probing_idx + __intra_bucket_idx;
  }

  //!
  //! @brief Extracts the __key from a given value type.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param __value The input value
  //!
  //! @return The __key
  template <class _Value>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto __extract_key(_Value __value) const noexcept
  {
    if constexpr (__has_payload)
    {
      return thrust::raw_reference_cast(__value).first;
    }
    else
    {
      return thrust::raw_reference_cast(__value);
    }
  }

  //!
  //! @brief Extracts the payload from a given value type.
  //!
  //! @note This function is only available if `this->__has_payload == true`
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param __value The input value
  //!
  //! @return The payload
  template <class _Value, class _Enable = ::cuda::std::enable_if_t<__has_payload and sizeof(_Value)>>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto __extract_payload(_Value __value) const noexcept
  {
    return thrust::raw_reference_cast(__value).second;
  }

  //!
  //! @brief Converts the given type to the container's native `__value_type`.
  //!
  //! @tparam T Input type which is convertible to '__value_type'
  //!
  //! @param __value The input value
  //!
  //! @return The converted object
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE constexpr __value_type __native_value(_Value __value) const noexcept
  {
    if constexpr (__has_payload)
    {
      return {static_cast<__key_type>(this->__extract_key(__value)), this->__extract_payload(__value)};
    }
    else
    {
      return static_cast<__value_type>(__value);
    }
  }

  //!
  //! @brief Converts the given type to the container's native `__value_type` while maintaining the
  //! heterogeneous __key type.
  //!
  //! @tparam T Input type which is convertible to '__value_type'
  //!
  //! @param __value The input value
  //!
  //! @return The converted object
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE constexpr auto __heterogeneous_value(_Value __value) const noexcept
  {
    if constexpr (__has_payload and not ::cuda::std::is_same_v<_Value, __value_type>)
    {
      using mapped_type = decltype(this->empty_value_sentinel());
      if constexpr (::cuda::experimental::cuco::__detail::__is_cuda_std_pair_like<_Value>::value)
      {
        return ::cuda::std::pair{::cuda::std::get<0>(__value), static_cast<mapped_type>(::cuda::std::get<1>(__value))};
      }
      else
      {
        // hail mary (convert using .first/.second members)
        return ::cuda::std::pair{thrust::raw_reference_cast(__value.first), static_cast<mapped_type>(__value.second)};
      }
    }
    else
    {
      return thrust::raw_reference_cast(__value);
    }
  }

  //!
  //! @brief Gets the sentinel used to represent an erased slot.
  //!
  //! @return The sentinel value used to represent an erased slot
  [[nodiscard]] _CCCL_DEVICE constexpr __value_type __erased_slot_sentinel() const noexcept
  {
    if constexpr (__has_payload)
    {
      return ::cuda::std::pair{this->erased_key_sentinel(), this->empty_value_sentinel()};
    }
    else
    {
      return this->erased_key_sentinel();
    }
  }

  //!
  //! @brief Inserts the specified element with one single CAS operation.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param ____address Pointer to the slot in memory
  //! @param ____expected Element to compare against
  //! @param ____desired Element to insert
  //!
  //! @return Result of this operation, i.e., success/continue/duplicate
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE constexpr __insert_result
  packed_cas(__value_type* __address, __value_type __expected, _Value __desired) noexcept
  {
    using packed_type =
      ::cuda::std::conditional_t<sizeof(__value_type) == 4, ::cuda::std::uint32_t, ::cuda::std::uint64_t>;

    auto* __slot_ptr     = reinterpret_cast<packed_type*>(__address);
    auto* __expected_ptr = reinterpret_cast<packed_type*>(&__expected);
    auto* __desired_ptr  = reinterpret_cast<packed_type*>(&__desired);

    auto __slot_ref = ::cuda::atomic_ref<packed_type, _Scope>{*__slot_ptr};

    const auto success =
      __slot_ref.compare_exchange_strong(*__expected_ptr, *__desired_ptr, ::cuda::memory_order_relaxed);

    if (success)
    {
      return __insert_result::__success;
    }
    else
    {
      return this->__predicate.__equal_to(this->__extract_key(__desired), this->__extract_key(__expected))
              == ::cuda::experimental::cuco::__detail::__equal_result::__equal
             ? __insert_result::__duplicate
             : __insert_result::__continue;
    }
  }

  //!
  //! @brief Inserts the specified element with two back-to-back CAS operations.
  //!
  //! @note This CAS can be used exclusively for `::cuda::experimental::cuco::op::insert` operations.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param ____address Pointer to the slot in memory
  //! @param ____expected Element to compare against
  //! @param ____desired Element to insert
  //!
  //! @return Result of this operation, i.e., success/continue/duplicate
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE constexpr __insert_result
  back_to_back_cas(__value_type* __address, __value_type __expected, _Value __desired) noexcept
  {
    using mapped_type = ::cuda::std::decay_t<decltype(this->empty_value_sentinel())>;

    auto __expected_key     = __expected.first;
    auto __expected_payload = this->empty_value_sentinel();

    ::cuda::atomic_ref<__key_type, _Scope> key_ref(__address->first);
    ::cuda::atomic_ref<mapped_type, _Scope> payload_ref(__address->second);

    const auto key_cas_success = key_ref.compare_exchange_strong(
      __expected_key, static_cast<__key_type>(__desired.first), ::cuda::memory_order_relaxed);
    auto payload_cas_success =
      payload_ref.compare_exchange_strong(__expected_payload, __desired.second, ::cuda::memory_order_relaxed);

    // if __key success
    if (key_cas_success)
    {
      while (not payload_cas_success)
      {
        payload_cas_success = payload_ref.compare_exchange_strong(
          __expected_payload = this->empty_value_sentinel(), __desired.second, ::cuda::memory_order_relaxed);
      }
      return __insert_result::__success;
    }
    else if (payload_cas_success)
    {
      // This is insert-specific, cannot for `erase` operations
      payload_ref.store(this->empty_value_sentinel(), ::cuda::memory_order_relaxed);
    }

    // Our __key was already present in the slot, so our __key is a duplicate
    // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
    if (this->__predicate.__equal_to(__desired.first, __expected_key)
        == ::cuda::experimental::cuco::__detail::__equal_result::__equal)
    {
      return __insert_result::__duplicate;
    }

    return __insert_result::__continue;
  }

  //!
  //! @brief Inserts the specified element with CAS-dependent write operations.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param ____address Pointer to the slot in memory
  //! @param ____expected Element to compare against
  //! @param ____desired Element to insert
  //!
  //! @return Result of this operation, i.e., success/continue/duplicate
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE constexpr __insert_result
  cas_dependent_write(__value_type* __address, __value_type __expected, _Value __desired) noexcept
  {
    using mapped_type = ::cuda::std::decay_t<decltype(this->empty_value_sentinel())>;

    ::cuda::atomic_ref<__key_type, _Scope> key_ref(__address->first);
    auto __expected_key = __expected.first;
    const auto success  = key_ref.compare_exchange_strong(
      __expected_key, static_cast<__key_type>(__desired.first), ::cuda::memory_order_relaxed);

    // if __key success
    if (success)
    {
      ::cuda::atomic_ref<mapped_type, _Scope> payload_ref(__address->second);
      payload_ref.store(__desired.second, ::cuda::memory_order_relaxed);
      return __insert_result::__success;
    }

    // Our __key was already present in the slot, so our __key is a duplicate
    // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
    if (this->__predicate.__equal_to(__desired.first, __expected_key)
        == ::cuda::experimental::cuco::__detail::__equal_result::__equal)
    {
      return __insert_result::__duplicate;
    }

    return __insert_result::__continue;
  }

  //!
  //! @brief Attempts to insert an element into a slot.
  //!
  //! @note Dispatches the correct implementation depending on the container
  //! type and presence of other operator mixins.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param ____address Pointer to the slot in memory
  //! @param ____expected Element to compare against
  //! @param ____desired Element to insert
  //!
  //! @return Result of this operation, i.e., success/continue/duplicate
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE __insert_result
  __attempt_insert(__value_type* __address, __value_type __expected, _Value __desired) noexcept
  {
    if constexpr (sizeof(__value_type) <= 8)
    {
      return packed_cas(__address, __expected, __desired);
    }
    else
    {
#if (__CUDA_ARCH__ < 700)
      return cas_dependent_write(__address, __expected, __desired);
#else
      return back_to_back_cas(__address, __expected, __desired);
#endif
    }
  }

  //!
  //! @brief Attempts to insert an element into a slot.
  //!
  //! @note Dispatches the correct implementation depending on the container
  //! type and presence of other operator mixins.
  //!
  //! @note `stable` indicates that the payload will only be updated once from the sentinel value to the desired value,
  //! meaning there can be no ABA situations.
  //!
  //! @tparam Value Input type which is convertible to '__value_type'
  //!
  //! @param ____address Pointer to the slot in memory
  //! @param ____expected Element to compare against
  //! @param ____desired Element to insert
  //!
  //! @return Result of this operation, i.e., success/continue/duplicate
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE __insert_result
  __attempt_insert_stable(__value_type* __address, __value_type __expected, _Value __desired) noexcept
  {
    if constexpr (sizeof(__value_type) <= 8)
    {
      return packed_cas(__address, __expected, __desired);
    }
    else
    {
      return cas_dependent_write(__address, __expected, __desired);
    }
  }

  //!
  //! @brief Waits until the slot payload has been updated
  //!
  //! @note The function will return once the slot payload is no longer equal to the sentinel
  //! Value.
  //!
  //! @tparam T Map slot type
  //!
  //! @param __slot The target slot to check payload with
  //! @param __sentinel The slot sentinel value
  template <class _Value>
  _CCCL_DEVICE void __wait_for_payload(_Value& __slot, _Value __sentinel) const noexcept
  {
    auto __ref = ::cuda::atomic_ref<_Value, _Scope>{__slot};
    _Value __current;
    // TODO exponential backoff strategy
    do
    {
      __current = __ref.load(::cuda::std::memory_order_relaxed);
    } while (::cuda::experimental::cuco::__detail::__bitwise_compare(__current, __sentinel));
  }

  // TODO: Clean up the sentinel handling since it's duplicated in ref and equal wrapper
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_REF_IMPL_CUH
