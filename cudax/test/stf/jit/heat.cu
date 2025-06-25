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
 *
 * @brief An example solving heat equation with finite differences using a
 * CUDA kernel
 */

#include <cuda/experimental/stf.cuh>
#include <cuda/experimental/__stf/nvrtc/jit_utils.cuh>

using namespace cuda::experimental::stf;

void dump_iter(slice<const double, 2> sUn, int iter)
{
  /* Create a binary file in the PPM format */
  char name[64];
  snprintf(name, 64, "heat_%06d.ppm", iter);
  FILE* f = fopen(name, "wb");
  fprintf(f, "P6\n%zu %zu\n255\n", sUn.extent(0), sUn.extent(1));
  for (size_t j = 0; j < sUn.extent(1); j++)
  {
    for (size_t i = 0; i < sUn.extent(0); i++)
    {
      int v = (int) (255.0 * sUn(i, j) / 100.0);
      // we assume values between 0.0 and 100.0 : max value is in red,
      // min is in blue
      unsigned char color[3];
      color[0] = static_cast<char>(v); /* red */
      color[1] = static_cast<char>(0); /* green */
      color[2] = static_cast<char>(255 - v); /* blue */
      fwrite(color, 1, 3, f);
    }
  }
  fclose(f);
}

const char *header_template = R"(
#include <cuda/experimental/__stf/nvrtc/slice.cuh>
using namespace cuda::experimental::stf;
)";


const char* heat_kernel_template = R"(
const double c = %a;
const double dx2 = %a;
const double dy2 = %a;

extern "C"
__global__ void %KERNEL_NAME%(slice<const double, 2> dyn_U, slice<double, 2> dyn_U1)
{
  %s U{dyn_U};
  %s U1{dyn_U1};

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = blockDim.x * gridDim.x;
  int dimy = blockDim.y * gridDim.y;

  for (size_t i = tidx + 1; i < U.extent(0)-1; i+= dimx)
    for (size_t j = tidy + 1; j < U.extent(1)-1; j += dimy)
    {
      U1(i, j) = U(i, j) + c * ((U(i - 1, j) - 2 * U(i, j) + U(i + 1, j)) / dx2 + (U(i, j - 1) - 2 * U(i, j) + U(i, j + 1)) / dy2);
    }
}

)";

int main()
{
  context ctx;

  // Initialize CUDA
  cuda_safe_call(cuInit(0));

  // Put the include paths in the NVRTC flags
  ::std::vector<::std::string> nvrtc_flags{"-I../../libcudacxx/include", "-I../../cudax/include/","-default-device"};
  ::std::string s =
    run_command(R"(echo "" | nvcc -v -x cu - -c 2>&1 | grep '#$ INCLUDES="' | grep -oP '(?<=INCLUDES=").*(?=" *$)')");

  // fprintf(stderr, "COMMAND %s\n", s.c_str());
  //  Split by whitespace
  ::std::istringstream iss(s);
  nvrtc_flags.insert(
    nvrtc_flags.end(), ::std::istream_iterator<::std::string>{iss}, ::std::istream_iterator<::std::string>{});

  // Compute the exact machine
  const int device          = cuda_try<cudaGetDevice>();
  const cudaDeviceProp prop = cuda_try<cudaGetDeviceProperties>(device);
  nvrtc_flags.push_back("--gpu-architecture=compute_" + ::std::to_string(prop.major) + ::std::to_string(prop.minor));

  const size_t N = 800;

  auto lU  = ctx.logical_data(shape_of<slice<double, 2>>(N, N));
  auto lU1 = ctx.logical_data(lU.shape());

  // Initialize the Un field with boundary conditions, and a disk at a lower
  // temperature in the middle.
  ctx.cuda_kernel(lU.write())->*[&](auto U) {
    const char *body =
     R"((size_t i, size_t j, auto U) {
            double rad = U.extent(0) / 8.0;
            double dx  = (double) i - U.extent(0) / 2;
            double dy  = (double) j - U.extent(1) / 2;

            U(i, j) = (dx * dx + dy * dy < rad * rad) ? 100.0 : 0.0;

            /* Set up boundary conditions */
            if (j == 0.0)
            {
              U(i, j) = 100.0;
            }
            if (j == U.extent(1) - 1)
            {
              U(i, j) = 0.0;
            }
            if (i == 0.0)
            {
              U(i, j) = 0.0;
            }
            if (i == U.extent(0) - 1)
            {
              U(i, j) = 0.0;
            }
       }
    )";


    auto gen_template = parallel_for_template_generator(lU.shape(), body, U);
    ::std::cout << "BEGIN GEN TEMPLATE\n";
    ::std::cout << gen_template;
    ::std::cout << "END GEN TEMPLATE\n";

    CUfunction kernel = lazy_jit(gen_template.c_str(), nvrtc_flags, header_template);
    return cuda_kernel_desc{kernel, 128, 32, 0, shape(U), U};
  };

  // diffusion constant
  double a = 0.5;

  double dx  = 0.1;
  double dy  = 0.1;
  double dx2 = dx * dx;
  double dy2 = dy * dy;

  // time step
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

  double c = a * dt;

  int nsteps     = 1000;
  int image_freq = 100;

  for (int iter = 0; iter < nsteps; iter++)
  {
    if (image_freq > 0 && iter % image_freq == 0)
    {
      // Dump Un in a PPM file
      ctx.host_launch(lU.read())->*[=](auto U) {
        dump_iter(U, iter);
      };
    }

    // Update Un using Un1 value with a finite difference scheme
#if 0
    ctx.task(lU.read(), lU1.write())->*[c, dx2, dy2, &nvrtc_flags](cudaStream_t stream, auto U, auto U1) {
      CUfunction kernel =
        lazy_jit(heat_kernel_template, nvrtc_flags, c, dx2, dy2, stringize_mdspan(U), stringize_mdspan(U1));

      void* args[] = {&U, &U1};
      // heat_kernel<<<128, 32, 0, stream>>>(U, U1, c, dx2, dy2);
      cuda_safe_call(cuLaunchKernel(kernel, 128, 1, 1, 32, 1, 1, 0, stream, args, nullptr));
    };
#else
//    ctx.cuda_kernel(lU.read(), lU1.write())->*[&](auto U, auto U1) {
//      CUfunction kernel =
//        lazy_jit(heat_kernel_template, nvrtc_flags, header_template, c, dx2, dy2, stringize_mdspan(U), stringize_mdspan(U1));
//
//      return cuda_kernel_desc{kernel, 128, 32, 0, U, U1};
//    };

  ctx.cuda_kernel(lU.read(), lU1.write())->*[&](auto U, auto U1) {
    const char *body = R"(
     (size_t i, size_t j, auto U, auto U1) {
        const double c = %a;
        const double dx2 = %a;
        const double dy2 = %a;
        // until we have box support
        if (i > 0 && j > 0 && i < U.extent(0) - 1 && j < U.extent(1) - 1) {
            U1(i, j) = U(i, j) + c * ((U(i - 1, j) - 2 * U(i, j) + U(i + 1, j)) / dx2 + (U(i, j - 1) - 2 * U(i, j) + U(i, j + 1)) / dy2);
        }
      }
      )";

    auto gen_template = parallel_for_template_generator(/*inner<1>(shape(U))*/shape(U), body, U, U1);
    CUfunction kernel = lazy_jit(gen_template.c_str(), nvrtc_flags, header_template, c, dx2, dy2);
    return cuda_kernel_desc{kernel, 128, 32, 0, /*inner<1>(shape(U))*/shape(U), U, U1};
  };


#endif

    ::std::swap(lU, lU1);
  }

  ctx.finalize();
  return 0;
}

int main2()
{
  float X[100 * 200];
  auto s = make_slice(&X[0], std::tuple{100, 200}, 200);
  ::std::cout << stringize_mdspan(s) << '\n';
  return 0;
}
