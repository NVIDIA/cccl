//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__utility/ensure_current_device.cuh>
#include <cuda/experimental/device.cuh>
#include <cuda/experimental/kernel.cuh>

#include <testing.cuh>

// extern "C" __constant__ int const_data;
//
// extern "C" __global__ void kernel_ptx1(int* array, int n)
// {
//   __shared__ int shared[32];
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   if (tid < n)
//   {
//     shared[threadIdx.x] = array[tid];
//     __syncthreads();
//     array[tid] = shared[threadIdx.x + 1 % 32] + const_data;
//   }
// }
//
// extern "C" __global__ void kernel_ptx2(float* array, int n)
// {
//   __shared__ float shared[32];
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   if (tid < n)
//   {
//     shared[threadIdx.x] = array[tid];
//     __syncthreads();
//     array[tid] = shared[threadIdx.x + 1 % 32] + static_cast<float>(const_data);
//   }
// }

constexpr char kernel_ptx_src[] = R"(
.version 6.0
.target sm_52
.address_size 64

	// .globl	kernel_ptx1
.const .align 4 .u32 const_data;
// _ZZ11kernel_ptx1E6shared has been demoted
// _ZZ11kernel_ptx2E6shared has been demoted

.visible .entry kernel_ptx1(
	.param .u64 kernel_ptx1_param_0,
	.param .u32 kernel_ptx1_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;
	// demoted variable
	.shared .align 4 .b8 _ZZ11kernel_ptx1E6shared[128];

	ld.param.u64 	%rd1, [kernel_ptx1_param_0];
	ld.param.u32 	%r3, [kernel_ptx1_param_1];
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %ctaid.x;
	mov.u32 	%r1, %tid.x;
	mad.lo.s32 	%r2, %r4, %r5, %r1;
	setp.ge.s32 	%p1, %r2, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r2, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.u32 	%r6, [%rd4];
	shl.b32 	%r7, %r1, 2;
	mov.u32 	%r8, _ZZ11kernel_ptx1E6shared;
	add.s32 	%r9, %r8, %r7;
	st.shared.u32 	[%r9], %r6;
	bar.sync 	0;
	ld.const.u32 	%r10, [const_data];
	ld.shared.u32 	%r11, [%r9+4];
	add.s32 	%r12, %r10, %r11;
	st.global.u32 	[%rd4], %r12;

$L__BB0_2:
	ret;

}
	// .globl	kernel_ptx2
.visible .entry kernel_ptx2(
	.param .u64 kernel_ptx2_param_0,
	.param .u32 kernel_ptx2_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<5>;
	// demoted variable
	.shared .align 4 .b8 _ZZ11kernel_ptx2E6shared[128];

	ld.param.u64 	%rd1, [kernel_ptx2_param_0];
	ld.param.u32 	%r3, [kernel_ptx2_param_1];
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %ctaid.x;
	mov.u32 	%r1, %tid.x;
	mad.lo.s32 	%r2, %r4, %r5, %r1;
	setp.ge.s32 	%p1, %r2, %r3;
	@%p1 bra 	$L__BB1_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r2, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	shl.b32 	%r6, %r1, 2;
	mov.u32 	%r7, _ZZ11kernel_ptx2E6shared;
	add.s32 	%r8, %r7, %r6;
	st.shared.f32 	[%r8], %f1;
	bar.sync 	0;
	ld.const.u32 	%r9, [const_data];
	cvt.rn.f32.s32 	%f2, %r9;
	ld.shared.f32 	%f3, [%r8+4];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB1_2:
	ret;

}
)";

#if _CCCL_CTK_AT_LEAST(12, 1)
extern "C" __global__ void kernel_rt(int* data, int size)
{
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx] += 1;
  }
}
#endif // _CCCL_CTK_AT_LEAST(12, 1)

C2H_CCCLRT_TEST("Kernel reference", "[kernel_ref]")
{
  CUlibrary lib{};
  CUDAX_REQUIRE(cuLibraryLoadData(&lib, kernel_ptx_src, nullptr, nullptr, 0, nullptr, nullptr, 0) == CUDA_SUCCESS);

  CUkernel kernel_ptx1_handle{};
  CUDAX_REQUIRE(cuLibraryGetKernel(&kernel_ptx1_handle, lib, "kernel_ptx1") == CUDA_SUCCESS);

  CUkernel kernel_ptx2_handle{};
  CUDAX_REQUIRE(cuLibraryGetKernel(&kernel_ptx2_handle, lib, "kernel_ptx2") == CUDA_SUCCESS);

  cudax::device_ref device{0};
  cudax::__ensure_current_device device_guard{device};

  SECTION("Types")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<typename cudax::kernel_ref<void()>::value_type, CUkernel>);
  }

  SECTION("Default constructor")
  {
    STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cudax::kernel_ref<void()>>);
  }

  SECTION("Constructor from kernel handle")
  {
    STATIC_REQUIRE(!cuda::std::is_convertible_v<CUkernel, cudax::kernel_ref<void()>>);
    STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::kernel_ref<void()>, CUkernel>);

    // We currently have no way to check if the kernel parameters match
    {
      cudax::kernel_ref<void()> kernel_ref{kernel_ptx1_handle};
      CUDAX_REQUIRE(kernel_ptx1_handle == kernel_ref.get());

      cudax::kernel_ref<void()> kernel_ref2{kernel_ptx2_handle};
      CUDAX_REQUIRE(kernel_ptx2_handle == kernel_ref2.get());

      CUDAX_REQUIRE(kernel_ptx1_handle != kernel_ptx2_handle);
      CUDAX_REQUIRE(kernel_ref.get() != kernel_ref2.get());
    }
    {
      cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
      CUDAX_REQUIRE(kernel_ptx1_handle == kernel_ref.get());

      cudax::kernel_ref<void(int*, int)> kernel_ref2{kernel_ptx2_handle};
      CUDAX_REQUIRE(kernel_ptx2_handle == kernel_ref2.get());

      CUDAX_REQUIRE(kernel_ptx1_handle != kernel_ptx2_handle);
      CUDAX_REQUIRE(kernel_ref.get() != kernel_ref2.get());
    }
  }

#if _CCCL_CTK_AT_LEAST(12, 1)
  SECTION("Constructor from kernel function")
  {
    STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::kernel_ref<void(int*, int)>, decltype(kernel_rt)>);
    STATIC_REQUIRE(cuda::std::is_convertible_v<decltype(kernel_rt), cudax::kernel_ref<void(int*, int)>>);
    STATIC_REQUIRE(!cuda::std::is_constructible_v<cudax::kernel_ref<void()>, decltype(kernel_rt)>);

    CUkernel kernel_rt_handle{};
    CUDAX_REQUIRE(cudaGetKernel(&kernel_rt_handle, kernel_rt) == cudaSuccess);

    cudax::kernel_ref<void(int*, int)> kernel_ref1{kernel_rt};
    CUDAX_REQUIRE(kernel_rt_handle == kernel_ref1.get());
  }
#endif // _CCCL_CTK_AT_LEAST(12, 1)

  SECTION("Copy constructor")
  {
    STATIC_REQUIRE(cuda::std::is_trivially_copy_constructible_v<cudax::kernel_ref<void()>>);

    cudax::kernel_ref<void(int*, int)> kernel_ref1{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ptx1_handle == kernel_ref1.get());

    cudax::kernel_ref<void(int*, int)> kernel_ref2{kernel_ref1};
    CUDAX_REQUIRE(kernel_ptx1_handle == kernel_ref2.get());
    CUDAX_REQUIRE(kernel_ref1.get() == kernel_ref2.get());
  }

#if _CCCL_CTK_AT_LEAST(12, 3)
  SECTION("Name")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().name()), cuda::std::string_view>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.name() == "kernel_ptx1");
  }
#endif // _CCCL_CTK_AT_LEAST(12, 3)

  SECTION("Max threads per block")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().max_threads_per_block(device)),
                           unsigned>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.max_threads_per_block(device) > 0);
  }

  SECTION("Static shared memory size")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().static_shared_memory_size(device)),
                           cuda::std::size_t>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.static_shared_memory_size(device) >= sizeof(int) * 32);
  }

  SECTION("Constant memory size")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().const_memory_size(device)),
                           cuda::std::size_t>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.const_memory_size(device) >= sizeof(int));
  }

  SECTION("Local memory size")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().local_memory_size(device)),
                           cuda::std::size_t>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    [[maybe_unused]] const auto local_size = kernel_ref.local_memory_size(device);
  }

  SECTION("Number of registers")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().num_regs(device)),
                                        cuda::std::size_t>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.num_regs(device) > 0);
  }

  SECTION("PTX version")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().ptx_version(device)), int>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.ptx_version(device) == 520);
  }

  SECTION("Binary version")
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().binary_version(device)), int>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ref.binary_version(device) == device.arch_traits().compute_capability);
  }

  SECTION("Get handle")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::kernel_ref<void()>>().get()), CUkernel>);

    cudax::kernel_ref<void(int*, int)> kernel_ref{kernel_ptx1_handle};
    CUDAX_REQUIRE(kernel_ptx1_handle == kernel_ref.get());
  }

  SECTION("Equality/Inequality comparison")
  {
    cudax::kernel_ref<void(int*, int)> kernel_ref1{kernel_ptx1_handle};
    cudax::kernel_ref<void(int*, int)> kernel_ref2{kernel_ptx2_handle};

    CUDAX_REQUIRE(kernel_ref1 == kernel_ref1);
    CUDAX_REQUIRE(kernel_ref1 != kernel_ref2);
  }

#if _CCCL_CTK_AT_LEAST(12, 1)
  SECTION("Deduction guidelines")
  {
    cudax::kernel_ref kernel_ref1{kernel_rt};
    CUDAX_REQUIRE((cuda::std::is_same_v<decltype(kernel_ref1), cudax::kernel_ref<void(int*, int)>>) );
  }
#endif // _CCCL_CTK_AT_LEAST(12, 1)

  CUDAX_REQUIRE(cuLibraryUnload(lib) == CUDA_SUCCESS);
}
