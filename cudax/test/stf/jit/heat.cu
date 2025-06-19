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

using namespace cuda::experimental::stf;

// Lazy cache by string content (can be replaced with hash or stronger keying)
template <typename... Args>
inline CUfunction lazy_jit(const char* template_str, ::std::vector<::std::string> opts, Args&&... args)
{
  static ::std::mutex cache_mutex;
  static ::std::map<::std::pair<::std::vector<::std::string>, ::std::string>, CUfunction> cache;

  auto make_printfable = [](auto&& arg) {
    using T = ::std::decay_t<decltype(arg)>;
    if constexpr (::std::is_same_v<T, ::std::string>)
    {
      return arg.c_str();
    }
    else
    {
      static_assert(::std::is_arithmetic_v<T>, "Unsupported type for JIT kernel argument");
      return arg;
    }
  };

  // Format code
  const int size = ::std::snprintf(nullptr, 0, template_str, make_printfable(args)...);
  // This will be our cache lookup key: a pair of options and the source code string
  auto key = ::std::pair(mv(opts), ::std::string(size, '\0'));
  ::std::snprintf(
    key.second.data(), key.second.size(), template_str, make_printfable(::std::forward<decltype(args)>(args))...);

  {
    ::std::lock_guard lock(cache_mutex);
    if (auto it = cache.find(key); it != cache.end())
    {
      return it->second;
    }
  }

  // Compile kernel
  nvrtcProgram prog = cuda_try<nvrtcCreateProgram>(key.second.c_str(), "jit_kernel.cu", 0, nullptr, nullptr);
  ::std::vector<const char*> raw_opts;
  raw_opts.reserve(key.first.size());
  for (const auto& s : key.first)
  {
    raw_opts.push_back(s.c_str());
  }
  nvrtcResult res = nvrtcCompileProgram(prog, raw_opts.size(), raw_opts.data());
  if (res != NVRTC_SUCCESS)
  {
    size_t log_size = 0;
    cuda_safe_call(nvrtcGetProgramLogSize(prog, &log_size));
    ::std::string log(log_size, '\0');
    cuda_safe_call(nvrtcGetProgramLog(prog, log.data()));
    ::std::cerr << "NVRTC compile error:\n" << log << ::std::endl;
    ::std::exit(1);
  }

  size_t ptx_size = 0;
  cuda_safe_call(nvrtcGetPTXSize(prog, &ptx_size));
  ::std::string ptx(ptx_size, '\0');
  cuda_safe_call(nvrtcGetPTX(prog, ptx.data()));
  cuda_safe_call(nvrtcDestroyProgram(&prog));

  CUmodule module   = cuda_try<cuModuleLoadData>(ptx.data());
  CUfunction kernel = cuda_try<cuModuleGetFunction>(module, "heat_kernel");

  {
    ::std::lock_guard lock(cache_mutex);
    cache[mv(key)] = kernel;
  }

  return kernel;
}

::std::string run_command(const char* cmd)
{
  ::std::array<char, 1024 * 64> buffer;
  ::std::string result;

  ::std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
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

const char* heat_kernel_template = R"(
#include <cuda/mdspan>

// Alias identical to the one you use on the host
template <typename T, size_t dimensions = 1>
using slice =
    ::cuda::std::mdspan<T,
                        ::cuda::std::dextents<size_t, dimensions>,
                        ::cuda::std::layout_stride>;

//extern "C"
//__global__ void heat_kernel(slice<const double, 2> U, slice<double, 2> U1, double c, double dx2, double dy2)
//{
//  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
//  int dimx = blockDim.x * gridDim.x;
//  int dimy = blockDim.y * gridDim.y;
//
//  for (size_t i = tidx + 1; i < U.extent(0)-1; i+= dimx)
//    for (size_t j = tidy + 1; j < U.extent(1)-1; j += dimy)
//    {
//      U1(i, j) = U(i, j) + c * ((U(i - 1, j) - 2 * U(i, j) + U(i + 1, j)) / dx2 + (U(i, j - 1) - 2 * U(i, j) + U(i, j + 1)) / dy2);
//    }
//}

extern "C"
__global__ void heat_kernel(%s U, %s U1, double c, double dx2, double dy2)
{
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

template <typename Mdspan, std::size_t... Is>
std::string stringize_mdspan(const Mdspan& md, std::index_sequence<Is...> = std::index_sequence<>{})
{
  constexpr std::size_t R = Mdspan::rank();
  if constexpr (R != sizeof...(Is))
  {
    return stringize_mdspan(md, std::make_index_sequence<R>{});
  }
  else
  {
    using ET       = typename Mdspan::element_type;
    using Layout   = typename Mdspan::layout_type;
    using Accessor = typename Mdspan::accessor_type;
    using XT       = typename Mdspan::extents_type;

    std::ostringstream oss;

    // mdspan<element_type,
    oss << "cuda::std::mdspan<" << type_name<ET>;

    // extents<size_t, e0, e1, ...>,
    oss << ", cuda::std::extents<size_t";
    if constexpr (R > 0)
    {
      oss << ", ";
    }

    ((oss << (Is ? ", " : "")
          << (XT::static_extent(Is) != cuda::std::dynamic_extent ? std::to_string(XT::static_extent(Is))
                                                                 : std::to_string(md.extent(Is)))),
     ...);

    oss << ">";

    // layout   (omit default)
    if constexpr (!std::is_same_v<Layout, cuda::std::layout_right>)
    {
      oss << ", " << type_name<Layout>;
    }

    // accessor (omit default)
    if constexpr (!std::is_same_v<Accessor, cuda::std::default_accessor<ET>>)
    {
      oss << ", " << type_name<Accessor>;
    }

    std::string out = oss.str();
    out += '>';

    return out;
  }
}

int main()
{
  context ctx;

  // Initialize CUDA
  cuda_safe_call(cuInit(0));

  // Put the include paths in the NVRTC flags
  ::std::vector<::std::string> nvrtc_flags{"-I../../libcudacxx/include"};
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

  const CUdevice cuDevice = cuda_try<cuDeviceGet>(0);
  const CUcontext context = cuda_try<cuCtxCreate>(0, cuDevice);

  const size_t N = 800;

  auto lU  = ctx.logical_data(shape_of<slice<double, 2>>(N, N));
  auto lU1 = ctx.logical_data(lU.shape());

  // Initialize the Un field with boundary conditions, and a disk at a lower
  // temperature in the middle.
  ctx.parallel_for(lU.shape(), lU.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto U) {
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
    ctx.task(lU.read(), lU1.write())->*[c, dx2, dy2, &nvrtc_flags](cudaStream_t stream, auto U, auto U1) {
      CUfunction kernel = lazy_jit(heat_kernel_template, nvrtc_flags, stringize_mdspan(U), stringize_mdspan(U1));

      void* args[] = {&U, &U1, const_cast<double*>(&c), const_cast<double*>(&dx2), const_cast<double*>(&dy2)};
      // heat_kernel<<<128, 32, 0, stream>>>(U, U1, c, dx2, dy2);
      cuda_safe_call(cuLaunchKernel(kernel, 128, 1, 1, 32, 1, 1, 0, stream, args, nullptr));
    };

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
