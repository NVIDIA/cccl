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

// Replaces all occurrences of 'from' with 'to' in the input string 'str'.
// This is a simple utility for named placeholder substitution (e.g., replacing %KERNEL_NAME%).
::std::string replace_all(::std::string str, ::std::string_view from, ::std::string_view to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length(); // Prevent infinite loop
    }
    return str;
}

inline void check_printf(const char* format)
{
  for (; *format; ++format)
  {
    if (*format == '%' && *++format != '%')
      throw ::std::runtime_error("Orphan format specifier: " + ::std::string(format - 1, 2));
  }
}

template <typename Head, typename... Tail>
void check_printf(const char* format, Head, Tail... rest)
{
  auto bailout = [&]() {
    throw ::std::runtime_error("Format specifier mismatch: " + ::std::string(format - 1, 2)
      + " for type " + ::std::string(type_name<Head>) + ".");
  };

  for (; *format; ++format)
  {
    if (*format != '%')
      continue;
    switch(*++format) // Move to the next character after '%'
    {
      case '%':
        continue; // Skip escaped '%'
      case 'a': case 'A': case 'f': case 'F': case 'e': case 'E': case 'g': case 'G':
        if (!::std::is_floating_point_v<Head>)
        {
          bailout();
        }
        break;
      case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': case 'c':
        if (!::std::is_integral_v<Head>)
        {
          bailout();
        }
        break;
      case 's':
        if (!::std::is_same_v<Head, const char*>)
        {
          bailout();
        }
        break;
      case 'p':
        if (!::std::is_pointer_v<Head>)
        {
          bailout();
        }
        break;
      case 'n':
        if (!::std::is_same_v<Head, int*>)
        {
          bailout();
        }
        break;
      default:
        throw std::runtime_error("Invalid format specifier: " + ::std::string(format - 1, 2));
    }
    return check_printf(format + 1, rest...);
  }
  // No more format specifiers, but still have arguments left
  throw std::runtime_error("No format specifier for the argument of type "
                           + ::std::string(type_name<Head>) + ".");
}

// Lazy cache by string content (can be replaced with hash or stronger keying)
template <typename... Args>
inline CUfunction lazy_jit(const char* template_str, ::std::vector<::std::string> opts, const char* header_template,  const Args&... args)
{
  static ::std::mutex cache_mutex;
  static ::std::map<::std::pair<::std::vector<::std::string>, ::std::string>, CUfunction> cache;

  [[maybe_unused]]
  auto make_printfable = [](const auto& arg) {
    using T = ::std::decay_t<decltype(arg)>;
    if constexpr (::std::is_same_v<const T, const ::std::string>)
    {
      return arg.c_str();
    }
    else
    {
      static_assert(::std::is_arithmetic_v<T>, "Unsupported type for JIT kernel argument");
      return arg;
    }
  };

  // XXX FIXME : we currently detect %KERNEL_NAME% in the template to replace
  // it, but if this is part of the key used to index kernels the string will
  // differ everytime. So we defer the substitution of kernel names later,
  // which introduces a problem with check_printf that will not work with
  // %KERNEL_NAME%.

  // Check if the format string is valid
  //  check_printf(template_with_name.c_str(), make_printfable(args)...);

  // Format code
  const int header_size = ::std::strlen(header_template);
  const int size = ::std::snprintf(nullptr, 0, template_str, make_printfable(args)...);
  // This will be our cache lookup key: a pair of options and the source code string
  auto key = ::std::pair(mv(opts), ::std::string(size + header_size + 1, '\0'));

  // Write header
  ::std::strcpy(key.second.data(), header_template);
  key.second.data()[header_size] = '\n'; // replace '\0'

  ::std::snprintf(key.second.data() + header_size, key.second.size() + 1, template_str, make_printfable(args)...);

  {
    ::std::lock_guard lock(cache_mutex);
    if (auto it = cache.find(key); it != cache.end())
    {
      return it->second;
    }
  }

  // Select generated kernel name: this cannot be hardcoded because we may instantiate the same template with different values
  static ::std::atomic<int> jit_kernel_cnt = 0;
  ::std::string kernel_name = "jit_kernel" + ::std::to_string(jit_kernel_cnt++);
  fprintf(stderr, "kernel_name: %s\n", kernel_name.c_str());
  ::std::string template_with_name = replace_all(key.second.c_str(), "%KERNEL_NAME%", kernel_name.c_str());

  ::std::cout << template_with_name << ::std::endl;

  // Compile kernel
  nvrtcProgram prog = cuda_try<nvrtcCreateProgram>(template_with_name.c_str(), "jit_kernel.cu", 0, nullptr, nullptr);
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

  fprintf(stderr, "loading function %s\n", kernel_name.c_str());

  CUmodule cuda_module   = cuda_try<cuModuleLoadData>(ptx.data());
  CUfunction kernel = cuda_try<cuModuleGetFunction>(cuda_module, kernel_name.c_str());

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

const char *header_template = R"(
#include <cuda/experimental/__stf/nvrtc/slice.cuh>

template <typename T, size_t dimensions = 1>
using slice = cuda::std::mdspan<T, ::cuda::std::dextents<size_t, dimensions>, ::cuda::std::layout_stride>;

template <typename T, size_t M, size_t N>
class static_slice
{
public:
    using extents_t = cuda::std::extents<size_t, M, N>;
    using layout_t  = cuda::std::layout_stride;
    using mdspan_t  = cuda::std::mdspan<T, extents_t, layout_t>;

    __host__ __device__
    static_slice(T* data,
                 typename layout_t::template mapping<extents_t> mapping)      // explicit ctor
        : view_{data, mapping}
    {}

    // convert from a dynamic-extents mdspan
    template <typename OtherMapping>
    __host__ __device__
    static_slice(const cuda::std::mdspan<T,
                                         cuda::std::dextents<size_t, 2>,
                                         OtherMapping>& dyn)
        : view_{dyn.data_handle(),
                typename layout_t::template mapping<extents_t>(dyn.mapping())}
    {
        assert(dyn.extent(0) == M && dyn.extent(1) == N);
    }

    __host__ __device__       T& operator()(size_t i, size_t j)       { return view_(i, j); }
    __host__ __device__ const T& operator()(size_t i, size_t j) const { return view_(i, j); }

    __host__ __device__ T* data()      const { return view_.data_handle(); }
    __host__ __device__ auto mapping() const { return view_.mapping();     }
    __host__ __device__ auto extents() const { return view_.extents();     }
    __host__ __device__ size_t extent(size_t i) const { return view_.extent(i); }

    __host__ __device__ constexpr size_t size() const noexcept
    {
        return M * N;
    }

private:
    mdspan_t view_;
};
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

const char *init_kernel_template = R"(
extern "C"
__global__ void %KERNEL_NAME%(slice<double, 2> dyn_U)
{
/// BEGIN TYPE CONVERSION SECTION
  %s U{dyn_U};
  const auto targs = ::cuda::std::make_tuple(U);
/// END TYPE CONVERSION SECTION


/// BEGIN BODY_SECTION (exclude "auto f =")
  auto f = [](size_t i, size_t j, auto U) {
    // printf("U(%%ld,%%ld) %%p = 100.0\n", (long)i, (long)j, &U(i, j));
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
/// END BODY_SECTION

/// BEGIN SHAPE SECTION
  // XXX HARDCODED ...
  // TODO size and index_to_coords should be "generated" from strings in a trait jit_shape<...>
  const size_t n = U.size();

  // This transforms a tuple of (shape, 1D index) into a coordinate
  auto shape_index_to_coords = [&U](size_t index)
  {
  //  printf("%%ld -> %%ld,%%ld\n", (long)index, index % U.extent(0),  index / U.extent(0));
    return ::cuda::std::make_tuple(index % U.extent(0), index / U.extent(0));
  };
/// END SHAPE SECTION

  // This will explode the targs tuple into a pack of data

  size_t _i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t _step = blockDim.x * gridDim.x;

  // Help the compiler which may not detect that a device lambda is calling a device lambda
  auto explode_args = [&](auto&&... data) {
    auto const explode_coords = [&](auto&&... coords) {
      f(coords..., data...);
    };
    // For every linearized index in the shape
    for (; _i < n; _i += _step)
    {
      ::cuda::std::apply(explode_coords, shape_index_to_coords(_i));
    }
  };
  ::cuda::std::apply(explode_args, ::std::move(targs));
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
    oss << "static_slice<" << type_name<ET>;

    // extents<size_t, e0, e1, ...>,
    if constexpr (R > 0)
    {
      oss << ", ";
    }

    ((oss << (Is ? ", " : "")
          << (XT::static_extent(Is) != cuda::std::dynamic_extent ? std::to_string(XT::static_extent(Is))
                                                                 : std::to_string(md.extent(Is)))),
     ...);

    oss << ">";

    std::string out = oss.str();

    return out;
  }
}

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
#if 0
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
#else
  ctx.cuda_kernel(lU.write())->*[&](auto U) {
    CUfunction kernel =
      lazy_jit(init_kernel_template, nvrtc_flags, header_template, stringize_mdspan(U));
    return cuda_kernel_desc{kernel, 128, 32, 0, U};
  };
#endif


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
    ctx.cuda_kernel(lU.read(), lU1.write())->*[&](auto U, auto U1) {
      CUfunction kernel =
        lazy_jit(heat_kernel_template, nvrtc_flags, header_template, c, dx2, dy2, stringize_mdspan(U), stringize_mdspan(U1));

      return cuda_kernel_desc{kernel, 128, 32, 0, U, U1};
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
