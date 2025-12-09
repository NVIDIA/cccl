//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// WANTS_CUDADEVRT.

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

__host__ __device__ void test()
{
  // 1. Test default constructor.
  static_assert(!cuda::std::is_default_constructible_v<cuda::device_ref>);

  // 2. Test constructor from int.
  static_assert(cuda::std::is_nothrow_constructible_v<cuda::device_ref, int>);
  static_assert(cuda::std::is_convertible_v<int, cuda::device_ref>);
  {
    cuda::device_ref device{0};
    assert(device.get() == 0);
  }

  // 3. Test copy constructor.
  static_assert(cuda::std::is_trivially_copy_constructible_v<cuda::device_ref>);

  // 3. Test move constructor.
  static_assert(cuda::std::is_trivially_move_constructible_v<cuda::device_ref>);

  // 4. Test copy assignment.
  static_assert(cuda::std::is_trivially_copy_assignable_v<cuda::device_ref>);

  // 5. Test copy move assignment.
  static_assert(cuda::std::is_trivially_move_assignable_v<cuda::device_ref>);

  // 6. Test get().
  static_assert(cuda::std::is_same_v<int, decltype(cuda::std::declval<const cuda::device_ref>().get())>);
  static_assert(noexcept(cuda::std::declval<const cuda::device_ref>().get()));
  {
    const cuda::device_ref device{0};
    assert(device.get() == 0);
  }

  // 7. Test init(). Just test the signature, advanced tests are in device_rt_tests.c2h.cu.
  NV_IF_TARGET(NV_IS_HOST, ({
                 static_assert(
                   cuda::std::is_same_v<void, decltype(cuda::std::declval<const cuda::device_ref>().init())>);
                 static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().init()));

                 cuda::device_ref device{0};
                 device.init();
               }))

  namespace attrs = cuda::device_attributes;

  // 7. Test attribute(Attr). We test only 1 attribute, rest of them are tested in attributes.pass.cpp.
  static_assert(
    cuda::std::is_same_v<cuda::compute_capability,
                         decltype(cuda::std::declval<const cuda::device_ref>().attribute(attrs::compute_capability))>);
  static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().attribute(attrs::compute_capability)));
  {
    const cuda::device_ref device{0};
    assert(device.attribute(attrs::compute_capability) == attrs::compute_capability(device));
  }

  // 7. Test attribute<cudaDeviceAttr>(). We test only 1 attribute, rest of them are tested in attributes.pass.cpp.
  static_assert(cuda::std::is_same_v<
                int,
                decltype(cuda::std::declval<const cuda::device_ref>().attribute<cudaDevAttrComputeCapabilityMajor>())>);
  static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().attribute<cudaDevAttrComputeCapabilityMajor>()));
  {
    const cuda::device_ref device{0};
    assert(device.attribute<cudaDevAttrComputeCapabilityMajor>() == attrs::compute_capability_major(device));
  }

  // 8. Test name().
  NV_IF_TARGET(
    NV_IS_HOST, ({
      static_assert(
        cuda::std::is_same_v<cuda::std::string_view, decltype(cuda::std::declval<const cuda::device_ref>().name())>);
      static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().name()));

      const cuda::device_ref device{0};
      const auto name = device.name();
      assert(!name.empty());

      // Test that the name is cached.
      const auto name2 = device.name();
      assert(name.data() == name2.data());
      assert(name.size() == name2.size());
    }))

  // 9. Test has_peer_access_to(device_ref). Just test the signature, advanced tests are in device_rt_tests.c2h.cu.
  NV_IF_TARGET(NV_IS_HOST, ({
                 static_assert(
                   cuda::std::is_same_v<bool,
                                        decltype(cuda::std::declval<const cuda::device_ref>().has_peer_access_to(
                                          cuda::std::declval<cuda::device_ref>()))>);
                 static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().has_peer_access_to(
                   cuda::std::declval<cuda::device_ref>())));

                 const cuda::device_ref device{0};
                 const auto has_self_peer_access = device.has_peer_access_to(device);
                 assert(!has_self_peer_access);
               }))

  // 10. Test peers(). Just test the signature, advanced tests are in device_rt_tests.c2h.cu.
  NV_IF_TARGET(NV_IS_HOST, ({
                 static_assert(cuda::std::is_same_v<cuda::std::span<const cuda::device_ref>,
                                                    decltype(cuda::std::declval<const cuda::device_ref>().peers())>);
                 static_assert(!noexcept(cuda::std::declval<const cuda::device_ref>().peers()));

                 const cuda::device_ref device{0};
                 const auto peers = device.peers();

                 // Check that peers are cached.
                 const auto peers2 = device.peers();
                 assert(peers.data() == peers2.data());
                 assert(peers.size() == peers2.size());
               }))

  // 11. Test operator cuda::memory_location().
  NV_IF_TARGET(
    NV_IS_HOST, ({
      static_assert(
        cuda::std::is_same_v<cuda::memory_location,
                             decltype(cuda::std::declval<const cuda::device_ref>().operator cuda::memory_location())>);
      static_assert(cuda::std::is_convertible_v<cuda::device_ref, cuda::memory_location>);
      static_assert(noexcept(cuda::std::declval<const cuda::device_ref>().operator cuda::memory_location()));

      const cuda::device_ref device{0};
      const cuda::memory_location location = device;
      assert(location.type == cudaMemLocationTypeDevice);
      assert(location.id == 0);

      if (cuda::devices.size() > 1)
      {
        const cuda::memory_location location2 = cuda::device_ref{1};
        assert(location2.type == cudaMemLocationTypeDevice);
        assert(location2.id == 1);
      }
    }))

  // 12. Test operator== and operator!=.
  static_assert(cuda::std::is_same_v<bool,
                                     decltype(cuda::std::declval<const cuda::device_ref>()
                                              == cuda::std::declval<const cuda::device_ref>())>);
  static_assert(cuda::std::is_same_v<bool,
                                     decltype(cuda::std::declval<const cuda::device_ref>()
                                              != cuda::std::declval<const cuda::device_ref>())>);
  static_assert(noexcept(cuda::std::declval<const cuda::device_ref>() == cuda::std::declval<const cuda::device_ref>()));
  static_assert(noexcept(cuda::std::declval<const cuda::device_ref>() != cuda::std::declval<const cuda::device_ref>()));
  {
    // We need to make these constexpr to prevent the _CCCL_ASSERT to be triggered.
    constexpr cuda::device_ref device{0};
    constexpr cuda::device_ref device2{1};

    assert(device == device);
    assert(!(device == device2));
    assert(!(device != device));
    assert(device != device2);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
#if !defined(TEST_NO_CUDADEVRT)
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
#endif // !TEST_NO_CUDADEVRT
  return 0;
}
