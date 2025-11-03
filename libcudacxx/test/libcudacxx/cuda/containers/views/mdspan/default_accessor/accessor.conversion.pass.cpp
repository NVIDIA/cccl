//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// USER 1 CODE

namespace user1
{
template <class ElementType>
class AccessorB;

// AccessorA is more type-erased than AccessorB.
template <class ElementType>
class AccessorA
{
public:
  using offset_policy    = AccessorA<ElementType>;
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = ElementType*;

  constexpr AccessorA() noexcept = default;

  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr AccessorA(const AccessorA<OtherElementType>&) noexcept
  {}

  // Conversion from AccessorB to AccessorA type-erases; it has no preconditions, and can therefore be safely implicit.
  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr AccessorA(const AccessorB<OtherElementType>&) noexcept
  {}

  __host__ __device__ constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
    return p[i];
  }

  __host__ __device__ constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept
  {
    return p + i;
  }
};

template <class ElementType>
class AccessorB
{
public:
  using offset_policy    = AccessorA<ElementType>;
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = ElementType*;

  constexpr AccessorB() noexcept = default;

  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr AccessorB(const AccessorB<OtherElementType>&) noexcept
  {}

  // Conversion from AccessorA to AccessorB asserts a precondition;
  // it un-type-erases from less specific AccessorA to more specific AccessorB. Thus, it is explicit.
  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr explicit AccessorB(const AccessorA<OtherElementType>&) noexcept
  {}

  __host__ __device__ constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
    return p[i];
  }

  __host__ __device__ constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept
  {
    return p + i;
  }
};
} // namespace user1

//----------------------------------------------------------------------------------------------------------------------
// USER 2 CODE

namespace user2
{
// Imagine that a different user1 writes AccessorC.
// They can't change the public interface of AccessorA, but they want to permit a conversion from AccessorC to
// AccessorA. AccessorA happens to be more type-erased than AccessorC, so this is a conversion without preconditions;
// therefore, it's not explicit.
template <class ElementType>
class AccessorC
{
public:
  using offset_policy    = AccessorC<ElementType>;
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = ElementType*;

  constexpr AccessorC() noexcept = default;

  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr AccessorC(const AccessorC<OtherElementType>&) noexcept
  {}

  __host__ __device__ constexpr operator user1::AccessorA<element_type>() const noexcept
  {
    return {};
  }

  __host__ __device__ constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
    return p[i];
  }

  __host__ __device__ constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept
  {
    return p + i;
  }
};

// Imagine that a different user1 writes AccessorD.
// They can't change the public interface of AccessorB, but they want to permit a conversion from AccessorD to
// AccessorB. This is a conversion with preconditions, so it's explicit.
template <class ElementType>
class AccessorD
{
public:
  using offset_policy    = AccessorD<ElementType>;
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = ElementType*;

  constexpr AccessorD() noexcept = default;

  template <class OtherElementType,
            cuda::std::enable_if_t<cuda::std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>, int> = 0>
  __host__ __device__ constexpr AccessorD(const AccessorD<OtherElementType>&) noexcept
  {}

  __host__ __device__ constexpr explicit operator user1::AccessorB<element_type>() const noexcept
  {
    return {};
  }

  __host__ __device__ constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
    return p[i];
  }

  __host__ __device__ constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept
  {
    return p + i;
  }
};
} // namespace user2

//----------------------------------------------------------------------------------------------------------------------
// TEST CODE

__host__ __device__ void test_basic_accessor_conversions()
{
  using user1::AccessorA;
  using user1::AccessorB;
  {
    AccessorA<float> A_acc1;
    AccessorA<float> A_acc2{A_acc1};
    AccessorA<const float> A_acc_const1{A_acc1};
    auto f = [](const AccessorA<float>& acc) -> AccessorA<const float> {
      return acc;
    };
    unused(f(AccessorA<float>{}));
    unused(A_acc2);
    unused(A_acc_const1);
  }
  {
    AccessorB<float> B_acc1;
    AccessorB<const float> B_acc_const1{B_acc1};
    AccessorB<float> B_acc2{B_acc1};
    auto f = [](const AccessorB<float>& B_acc1) -> AccessorB<const float> {
      return B_acc1;
    };
    unused(f(AccessorB<float>{}));
    unused(B_acc2);
    unused(B_acc_const1);
  }
  // Implicit conversion from AccessorB to AccessorA.
  {
    AccessorA<float> A_acc1{AccessorB<float>{}};
    AccessorA<const float> A_acc_const1{AccessorB<float>{}};
    AccessorA<const float> A_acc_const2{AccessorB<const float>{}};
    // AccessorA<float> A_acc2{AccessorB<const float>{}}; // doesn't compile, rightfully so
    auto f = [](const AccessorB<float>& B_acc1) -> AccessorA<float> {
      return B_acc1; // implicit conversion
    };
    unused(f(AccessorB<float>{}));
    unused(A_acc1);
    unused(A_acc_const1);
    unused(A_acc_const2);
  }
  // Only explicit conversion from AccessorA to AccessorB.
  {
    AccessorB<float> B_acc1{AccessorA<float>{}};
    AccessorB<const float> B_acc_const1{AccessorA<float>{}};
    AccessorB<const float> b_acc_const2{AccessorA<const float>{}};
    // AccessorB<float> B_acc2{AccessorA<const float>{}}; // doesn't compile, rightfully so
    auto f = [](const AccessorA<float>& A_acc1) -> AccessorB<float> {
      // return A_acc1; // implicit conversion doesn't compile, rightfully so
      return AccessorB<float>{A_acc1};
    };
    unused(f(AccessorA<float>{}));
    unused(B_acc1);
    unused(B_acc_const1);
    unused(b_acc_const2);
  }
}

template <template <class> class Wrapper>
__host__ __device__ void test_host_device_accessor_conversions()
{
  using user1::AccessorA;
  using user1::AccessorB;
  using WrapperA      = Wrapper<AccessorA<float>>;
  using WrapperAconst = Wrapper<AccessorA<const float>>;
  using WrapperB      = Wrapper<AccessorB<float>>;
  {
    // Test CTAD with wrapping constructor
    WrapperA wrapper_acc1(AccessorA<float>{});
    static_assert(cuda::std::is_same_v<decltype(wrapper_acc1), WrapperA>);
    WrapperAconst wrapper_acc_const1(AccessorA<const float>{});
    static_assert(cuda::std::is_same_v<decltype(wrapper_acc_const1), WrapperAconst>);

    // Test CTAD with copy constructor
    WrapperA wrapper_acc2{wrapper_acc1};
    static_assert(cuda::std::is_same_v<decltype(wrapper_acc2), decltype(wrapper_acc1)>);
    WrapperAconst wrapper_acc_const2{wrapper_acc_const1};
    static_assert(cuda::std::is_same_v<decltype(wrapper_acc_const2), decltype(wrapper_acc_const1)>);
    unused(wrapper_acc2);
    unused(wrapper_acc_const2);

    // Test converting constructor: Wrapper<AccessorA<const T>>(AccessorA<T>)
    WrapperAconst wrapper_acc_const3{wrapper_acc1};
    unused(wrapper_acc_const3);
    // Test implicit conversion: Wrapper<AccessorA<T>> -> Wrapper<AccessorA<const T>>
    auto f = [](const WrapperA& wrapper_acc1) -> WrapperAconst {
      return wrapper_acc1;
    };
    unused(f(WrapperA{}));
  }
  {
    // Test (explicit) converting constructor: Wrapper<AccessorB<T>>(AccessorA<T>)
    WrapperB wrapper_acc3{WrapperA{}};
    // Test implicit conversion from AccessorB<T> to AccessorA<T> (type erasure)
    auto f1 = [](const WrapperB& wrapper_acc1) -> WrapperA {
      return wrapper_acc1;
    };
    unused(f1(wrapper_acc3));

    // Test implicit conversion from AccessorB<T> to AccessorA<const T> (type erasure)
    auto f2 = [](const WrapperB& wrapper_acc1) -> WrapperAconst {
      return wrapper_acc1;
    };
    unused(f2(wrapper_acc3));

    // Test that implicit conversion from AccessorA<T> to AccessorB<T> is forbidden
    auto f3 = [](const WrapperA& wrapper_acc1) -> WrapperB {
      return WrapperB{wrapper_acc1};
      // return wrapper_acc1; // rightfully does not compile
    };
    unused(f3(WrapperA{}));
  }
}

template <template <class> class Wrapper>
__host__ __device__ void test_conversion()
{
  using user1::AccessorA;
  using user1::AccessorB;
  using user2::AccessorC;
  using user2::AccessorD;
  using WrapperA = Wrapper<AccessorA<float>>;
  using WrapperB = Wrapper<AccessorB<float>>;
  using WrapperC = Wrapper<AccessorC<float>>;
  using WrapperD = Wrapper<AccessorD<float>>;
  {
    // Test explicit and implicit conversion from Wrapper<AccessorC<T>> to Wrapper<AccessorA<T>>.
    // This works because Wrapper<AccessorC<T>> publicly inherits from AccessorC<T>,
    // so it inherits AccessorC<T>'s conversion operators.
    [[maybe_unused]] WrapperA wrapper_acc1{WrapperC{}};
    auto f1 = [](const WrapperA& wrapper_acc1) -> WrapperA {
      return wrapper_acc1;
    };
    unused(f1(WrapperC{}));
  }
  {
    // Test explicit conversion from Wrapper<AccessorD<T>> to Wrapper<AccessorB<T>>.
    // This works because Wrapper<AccessorD<T>> publicly inherits from AccessorD<T>,
    // so it inherits AccessorD<T>'s conversion operators.
    [[maybe_unused]] WrapperB wrapper_acc1{WrapperD{}};
    auto f1 = [](const WrapperD& w) -> WrapperB {
      return WrapperB{};
      // return w; // rightfully does not compile
    };
    unused(f1(WrapperD{}));
  }
}

// Application: conversion of Wrapper<aligned_accessor<T, N>>
// to Wrapper<default_accessor<T>>.
template <template <class> class Wrapper>
__host__ __device__ void test_aligned_to_default()
{
  using WrapperDefault = Wrapper<cuda::std::default_accessor<float>>;
  using WrapperAligned = Wrapper<cuda::std::aligned_accessor<float, 16>>;
  WrapperAligned wrapper_align_acc{cuda::std::aligned_accessor<float, 16>{}};
  auto f = [](const WrapperAligned& w) -> WrapperDefault {
    return w;
  };
  unused(f(wrapper_align_acc));
}

// Application: Explicit conversion of Wrapper<default_accessor<T>>
// to Wrapper<aligned_accessor<T, N>>.
template <template <class> class Wrapper>
__host__ __device__ void test_default_to_aligned()
{
  using WrapperDefault = Wrapper<cuda::std::default_accessor<float>>;
  using WrapperAligned = Wrapper<cuda::std::aligned_accessor<float, 16>>;
  WrapperDefault wrapper_default_acc{cuda::std::default_accessor<float>{}};
  WrapperAligned wrapper_aligned_acc{wrapper_default_acc};
  auto f = [](const WrapperDefault& w) -> WrapperAligned {
    return WrapperAligned{w};
    // return w; // rightfully does not compile
  };
  unused(wrapper_aligned_acc);
  unused(f(wrapper_default_acc));
}

template <template <class> class Wrapper>
__host__ __device__ void test_managed_conversions()
{
  using HostOrDeviceAccessor      = Wrapper<cuda::std::default_accessor<float>>;
  using ManagedAccessor           = cuda::managed_accessor<cuda::std::default_accessor<float>>;
  using HostOrDeviceAccessorConst = cuda::host_accessor<cuda::std::default_accessor<const float>>;
  using ManagedAccessorConst      = cuda::managed_accessor<cuda::std::default_accessor<const float>>;
  static_assert(cuda::is_host_device_managed_accessor_v<HostOrDeviceAccessor>);
  static_assert(cuda::is_host_device_managed_accessor_v<HostOrDeviceAccessorConst>);
  HostOrDeviceAccessor host_acc1{ManagedAccessor{}};
  HostOrDeviceAccessorConst host_acc2{ManagedAccessor{}};
  HostOrDeviceAccessorConst host_acc3{ManagedAccessorConst{}};
  unused(host_acc1);
  unused(host_acc2);
  unused(host_acc3);
  {
    auto f = [](const ManagedAccessor& w) -> HostOrDeviceAccessor {
      return w;
    };
    unused(f(ManagedAccessor{}));
  }
  {
    auto f = [](const ManagedAccessorConst& w) -> HostOrDeviceAccessorConst {
      return w;
    };
    unused(f(ManagedAccessorConst{}));
  }
  {
    auto f = [](const ManagedAccessor& w) -> HostOrDeviceAccessorConst {
      return w;
    };
    unused(f(ManagedAccessor{}));
  }
}

int main(int, char**)
{
  test_basic_accessor_conversions();
  test_host_device_accessor_conversions<cuda::host_accessor>();
  test_host_device_accessor_conversions<cuda::device_accessor>();
  test_host_device_accessor_conversions<cuda::managed_accessor>();
  test_conversion<cuda::host_accessor>();
  test_conversion<cuda::device_accessor>();
  test_conversion<cuda::managed_accessor>();
  test_aligned_to_default<cuda::host_accessor>();
  test_aligned_to_default<cuda::device_accessor>();
  test_aligned_to_default<cuda::managed_accessor>();
  test_default_to_aligned<cuda::host_accessor>();
  test_default_to_aligned<cuda::device_accessor>();
  test_default_to_aligned<cuda::managed_accessor>();
  test_managed_conversions<cuda::host_accessor>();
  test_managed_conversions<cuda::device_accessor>();
  return 0;
}
