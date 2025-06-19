//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief This test uses jit compilation to compile a simple AXPY kernel
 */

#include <cuda/experimental/stf.cuh>

#include <nvrtc.h>

using namespace cuda::experimental::stf;

const char* axpy_kernel = R"(
//#include <cuda/experimental/stf.cuh>
#include <cuda/mdspan>

// Alias identical to the one you use on the host
template <typename T, size_t dimensions = 1>
using slice =
    ::cuda::std::mdspan<T,
                        ::cuda::std::dextents<size_t, dimensions>,
                        ::cuda::std::layout_stride>;

extern "C" __global__
void axpy(int n, float a, slice<const float> x, slice<float> y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
)";

::std::string run_command(const char* cmd)
{
  std::array<char, 1024 * 64> buffer;
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe)
  {
    return result;
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()))
  {
    result += buffer.data();
  }

  if (result.back() == '\n')
  {
    result.pop_back(); // Remove trailing newline
  }

  return result;
}

int main()
{
  // Initialize CUDA
  cuda_safe_call(cuInit(0));

  // Put the include paths in the NVRTC flags
  ::std::vector<::std::string> nvrtc_flags{"-I../../libcudacxx/include"};
  ::std::string s =
    run_command(R"(echo "" | nvcc -v -x cu - -c 2>&1 | grep '#$ INCLUDES="' | grep -oP '(?<=INCLUDES=").*(?=" *$)')");
  // Split by whitespace
  std::istringstream iss(s);
  nvrtc_flags.insert(nvrtc_flags.end(), std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{});

  // Compute the exact machine
  const int device          = cuda_try<cudaGetDevice>();
  const cudaDeviceProp prop = cuda_try<cudaGetDeviceProperties>(device);
  nvrtc_flags.push_back("--gpu-architecture=compute_" + ::std::to_string(prop.major) + ::std::to_string(prop.minor));

  const CUdevice cuDevice = cuda_try<cuDeviceGet>(0);
  const CUcontext context = cuda_try<cuCtxCreate>(0, cuDevice);

  // Compile kernel
  ::std::vector<const char*> opts;
  opts.reserve(nvrtc_flags.size());
  for (const auto& s : nvrtc_flags)
  {
    opts.push_back(s.c_str());
  }
  nvrtcProgram prog = cuda_try<nvrtcCreateProgram>(axpy_kernel, "axpy.cu", 0, nullptr, nullptr);
  nvrtcResult res   = nvrtcCompileProgram(prog, opts.size(), opts.data());

  if (res != NVRTC_SUCCESS)
  {
    size_t logSize;
    cuda_safe_call(nvrtcGetProgramLogSize(prog, &logSize));

    std::string log(logSize, '\0');
    cuda_safe_call(nvrtcGetProgramLog(prog, log.data()));

    std::cerr << "NVRTC compilation failed:\n" << log << std::endl;
    std::cerr << "Error code: " << res << " (" << nvrtcGetErrorString(res) << ")\n";

    exit(1); // or handle as appropriate
  }

  size_t ptxSize = 0;
  cuda_safe_call(nvrtcGetPTXSize(prog, &ptxSize));

  std::string ptx(ptxSize, '\0');
  cuda_safe_call(nvrtcGetPTX(prog, ptx.data()));
  cuda_safe_call(nvrtcDestroyProgram(&prog));

  // Load PTX
  const CUmodule module   = cuda_try<cuModuleLoadData>(ptx.data());
  const CUfunction kernel = cuda_try<cuModuleGetFunction>(module, "axpy");

  // Prepare data
  int N   = 1024;
  float a = 2.0f;
  std::vector<float> h_x(N, 1.0f);
  std::vector<float> h_y(N, 2.0f);

  CUdeviceptr d_x, d_y;
  cuda_safe_call(cuMemAlloc(&d_x, N * sizeof(float)));
  cuda_safe_call(cuMemAlloc(&d_y, N * sizeof(float)));

  cuda_safe_call(cuMemcpyHtoD(d_x, h_x.data(), N * sizeof(float)));
  cuda_safe_call(cuMemcpyHtoD(d_y, h_y.data(), N * sizeof(float)));

  // Create mdspan slices
  auto x_slice = make_slice(reinterpret_cast<float*>(d_x), N);
  auto y_slice = make_slice(reinterpret_cast<float*>(d_y), N);

  // Launch kernel
  void* args[] = {&N, &a, &x_slice, &y_slice};
  int threads  = 256;
  int blocks   = (N + threads - 1) / threads;

  cuda_safe_call(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr));

  cuda_safe_call(cuCtxSynchronize());

  // Copy result back
  cuda_safe_call(cuMemcpyDtoH(h_y.data(), d_y, N * sizeof(float)));

  // Verify result
  for (int i = 0; i < N; ++i)
  {
    assert(h_y[i] == 4.0f);
  }

  // Cleanup
  cuda_safe_call(cuMemFree(d_x));
  cuda_safe_call(cuMemFree(d_y));
  cuda_safe_call(cuModuleUnload(module));
  cuda_safe_call(cuCtxDestroy(context));
}
