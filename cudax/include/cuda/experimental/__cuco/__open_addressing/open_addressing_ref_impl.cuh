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
#include <thrust/logical.h>
#include <thrust/reduce.h>

#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__utility/static_for.h>
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
//! @tparam ProbingScheme Probing scheme (see `cuda/experimental/__cuco/probing_scheme.cuh` for options)
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
          const auto __intra_bucket_index = &__slot_content - __bucket_slots.data();
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
      if constexpr (::cuda::experimental::cuco::__detail::__is_pair_like<_Value>::value)
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
  //! @note This CAS is used exclusively to implement the insert operation.
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
