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
#include <cuda/std/type_traits>
#if !_CCCL_COMPILER(NVRTC)
#  include <stdexcept>
#endif // !_CCCL_COMPILER(NVRTC)

__host__ __device__ void test()
{
  // 1. Test cuda::devices type.
  static_assert(cuda::std::is_same_v<const cuda::__all_devices, decltype(cuda::devices)>);

  // 2. Test cuda::__all_devices types.
  static_assert(cuda::std::is_same_v<cuda::device_ref, cuda::__all_devices::value_type>);
  static_assert(cuda::std::is_same_v<cuda::std::size_t, cuda::__all_devices::size_type>);
  using iterator = cuda::__all_devices::iterator;

  // 2. Test size() method.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::devices.size())>);
  static_assert(!noexcept(cuda::devices.size()));
  const auto device_count = cuda::devices.size();
  assert(device_count != 0);

  // 3. Test operator[] method.
  static_assert(cuda::std::is_same_v<cuda::device_ref, decltype(cuda::devices[cuda::std::size_t{}])>);
  static_assert(!noexcept(cuda::devices[cuda::std::size_t{}]));
  {
    const auto device = cuda::devices[0];
    assert(device.get() == 0);
  }
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, ({
                 try
                 {
                   [[maybe_unused]] const auto device = cuda::devices[19283901];
                   assert(false);
                 }
                 catch (const std::out_of_range&)
                 {
                   assert(true);
                 }
                 catch (...)
                 {
                   assert(false);
                 }
               }))
#endif // _CCCL_HAS_EXCEPTIONS()

  // 4. Test begin() and end() methods.
  NV_IF_TARGET(NV_IS_HOST, ({
                 static_assert(cuda::std::is_same_v<iterator, decltype(cuda::devices.begin())>);
                 static_assert(cuda::std::is_same_v<iterator, decltype(cuda::devices.end())>);

                 static_assert(!noexcept(cuda::devices.begin()));
                 static_assert(!noexcept(cuda::devices.end()));

                 const auto begin = cuda::devices.begin();
                 const auto end   = cuda::devices.end();

                 assert(begin != end);
                 assert(begin->get() == 0);
                 assert(end - begin == static_cast<cuda::std::ptrdiff_t>(device_count));
               }))
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
#if !defined(TEST_NO_CUDADEVRT)
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
#endif // !TEST_NO_CUDADEVRT
  return 0;
}
