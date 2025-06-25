//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Helpers to use NVRTC in tasks (host side)
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/slice_core.cuh>
#include <string>
#include <stdexcept>

namespace cuda::experimental::stf
{

// Replaces all occurrences of 'from' with 'to' in the input string 'str'.
// This is a simple utility for named placeholder substitution (e.g., replacing %KERNEL_NAME%).
::std::string replace_all(::std::string str, ::std::string_view from, ::std::string_view to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != ::std::string::npos) {
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
        throw ::std::runtime_error("Invalid format specifier: " + ::std::string(format - 1, 2));
    }
    return check_printf(format + 1, rest...);
  }
  // No more format specifiers, but still have arguments left
  throw ::std::runtime_error("No format specifier for the argument of type "
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

template <typename Mdspan, std::size_t... Is>
::std::string stringize_mdspan(const Mdspan& md, ::std::index_sequence<Is...> = ::std::index_sequence<>{})
{
  constexpr std::size_t R = Mdspan::rank();
  if constexpr (R != sizeof...(Is))
  {
    return stringize_mdspan(md, ::std::make_index_sequence<R>{});
  }
  else
  {
    using ET       = typename Mdspan::element_type;
    using Layout   = typename Mdspan::layout_type;
    using Accessor = typename Mdspan::accessor_type;
    using XT       = typename Mdspan::extents_type;

    ::std::ostringstream oss;

    // mdspan<element_type,
    oss << "::cuda::experimental::stf::static_slice<" << type_name<ET>;

    // extents<size_t, e0, e1, ...>,
    if constexpr (R > 0)
    {
      oss << ", ";
    }

    ((oss << (Is ? ", " : "")
          << (XT::static_extent(Is) != ::cuda::std::dynamic_extent ? ::std::to_string(XT::static_extent(Is))
                                                                   : ::std::to_string(md.extent(Is)))),
     ...);

    oss << ">";

    return oss.str();
  }
}



template <typename shape_t, typename ...Args>
::std::string parallel_for_template_generator([[maybe_unused]] shape_t shape, const char *body_template, Args... args)
{
    ::std::ostringstream oss;

    // kernel arguments
    ::std::ostringstream args_prototype_oss;

    // Convert dynamic args to static args
    ::std::ostringstream args_conversion_oss;

    // this will contain "auto targs = make_tuple(...);"
    ::std::ostringstream args_tuple_oss;
    args_tuple_oss << "const auto targs = ::cuda::std::make_tuple(";

    args_prototype_oss << ::std::string(type_name<shape_t>) << " dyn_shape, ";

    size_t arg_index = 0;
    auto emit_arg = [&]([[maybe_unused]] auto &&arg) {
        using raw_arg = ::cuda::std::remove_reference_t<decltype(arg)>;

        if (arg_index > 0) {
             args_tuple_oss << ", ";
             args_prototype_oss << ", ";
        }

        args_conversion_oss << stringize_mdspan(arg) << " static_arg" << arg_index << "{dyn_arg" << arg_index << "};\n";
        args_prototype_oss << ::std::string(type_name<raw_arg>) << " dyn_arg" << arg_index;

        args_tuple_oss << "static_arg" << arg_index;

        arg_index++;
    };

    (emit_arg(args), ...);

    args_tuple_oss << ");\n";

    oss << "extern \"C\"\n";
    oss << "__global__ void %KERNEL_NAME%(" << args_prototype_oss.str() << ")\n";
    oss << "{\n";

    oss << args_conversion_oss.str() << ::std::endl;
    oss << args_tuple_oss.str() << ::std::endl;

    oss << "auto _body_impl = []" << body_template << ";\n";

    oss << R"(
           size_t _i          = blockIdx.x * blockDim.x + threadIdx.x;
           const size_t _step = blockDim.x * gridDim.x;
           const size_t n = dyn_shape.size();

           auto explode_args = [&](auto&&... data) {
             auto const explode_coords = [&](auto&&... coords) {
               _body_impl(coords..., data...);
             };
             // For every linearized index in the shape
             for (; _i < n; _i += _step)
             {
               ::cuda::std::apply(explode_coords, dyn_shape.index_to_coords(_i));
             }
           };
           ::cuda::std::apply(explode_args, ::std::move(targs));
)";

    oss << "}\n";

    return oss.str();
}



} // end namespace cuda::experimental::stf
