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

#include <cuda/experimental/__stf/internal/cuda_kernel_scope.cuh>
#include <cuda/experimental/__stf/internal/slice_core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

#include <atomic>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

namespace cuda::experimental::stf
{

// Replaces all occurrences of 'from' with 'to' in the input string 'str'.
// This is a simple utility for named placeholder substitution (e.g., replacing %KERNEL_NAME%).
inline ::std::string replace_all(::std::string str, ::std::string_view from, ::std::string_view to)
{
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != ::std::string::npos)
  {
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
    {
      throw ::std::runtime_error("Orphan format specifier: " + ::std::string(format - 1, 2));
    }
  }
}

template <typename Head, typename... Tail>
void check_printf(const char* format, Head, Tail... rest)
{
  auto bailout = [&]() {
    throw ::std::runtime_error("Format specifier mismatch: " + ::std::string(format - 1, 2) + " for type "
                               + ::std::string(type_name<Head>) + ".");
  };

  for (; *format; ++format)
  {
    if (*format != '%')
    {
      continue;
    }
    switch (*++format) // Move to the next character after '%'
    {
      case '%':
        continue; // Skip escaped '%'
      case 'a':
      case 'A':
      case 'f':
      case 'F':
      case 'e':
      case 'E':
      case 'g':
      case 'G':
        if (!::std::is_floating_point_v<Head>)
        {
          bailout();
        }
        break;
      case 'd':
      case 'i':
      case 'u':
      case 'o':
      case 'x':
      case 'X':
      case 'c':
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
  throw ::std::runtime_error("No format specifier for the argument of type " + ::std::string(type_name<Head>) + ".");
}

// Lazy cache by string content (can be replaced with hash or stronger keying)
template <typename... Args>
inline CUfunction lazy_jit(
  const char* template_str, const ::std::vector<::std::string>& opts, const char* header_template, const Args&... args)
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
  // differ every time. So we defer the substitution of kernel names later,
  // which introduces a problem with check_printf that will not work with
  // %KERNEL_NAME%.

  // Check if the format string is valid
  //  check_printf(template_with_name.c_str(), make_printfable(args)...);

  // Format code
  auto key = ::std::pair(opts, ::std::string());

  if constexpr (sizeof...(args) == 0)
  {
    // Security warning upon calling snprintf with no arguments
    key.second = header_template;
    key.second += '\n';
    key.second += template_str;
  }
  else
  {
    const int header_size = ::std::strlen(header_template);
    const int size        = ::std::snprintf(nullptr, 0, template_str, make_printfable(args)...);
    // This will be our cache lookup key: a pair of options and the source code string
    key.second = ::std::string(size + header_size + 1, '\0');

    // Write header
    ::std::strcpy(key.second.data(), header_template);
    key.second.data()[header_size] = '\n'; // replace '\0'

    ::std::snprintf(key.second.data() + header_size, key.second.size() + 1, template_str, make_printfable(args)...);
  }

  {
    ::std::lock_guard lock(cache_mutex);

    if (auto it = cache.find(key); it != cache.end())
    {
      return it->second;
    }
  }

  // Select generated kernel name: this cannot be hardcoded because we may instantiate the same template with different
  // values
  static ::std::atomic<int> jit_kernel_cnt = 0;
  ::std::string kernel_name                = "jit_kernel" + ::std::to_string(jit_kernel_cnt++);
  ::std::string template_with_name         = replace_all(key.second.c_str(), "%KERNEL_NAME%", kernel_name.c_str());

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

  CUmodule cuda_module = cuda_try<cuModuleLoadData>(ptx.data());
  CUfunction kernel    = cuda_try<cuModuleGetFunction>(cuda_module, kernel_name.c_str());

  if (getenv("CUDASTF_JIT_DEBUG_PTX") || getenv("CUDASTF_JIT_DEBUG"))
  {
    int num_registers = cuda_try<cuFuncGetAttribute>(CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    ::std::cerr
      << "kernel_name " << kernel_name << " using " << ::std::to_string(num_registers) << " registers" << ::std::endl;
    ::std::cerr << "SOURCE BEGIN:\n";
    ::std::cout << template_with_name << ::std::endl;
    ::std::cerr << "SOURCE END:\n";
  }

  if (getenv("CUDASTF_JIT_DEBUG_PTX"))
  {
    ::std::cerr << "PTX BEGIN:\n";
    ::std::cerr.write(ptx.data(), ptx.size()).flush();
    ::std::cerr << "PTX END:\n";
  }

  {
    ::std::lock_guard lock(cache_mutex);
    cache[mv(key)] = kernel;
  }

  return kernel;
}

inline ::std::string run_command(const char* cmd)
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

template <size_t Dimensions>
::std::string stringize_box(const box<Dimensions>& b)
{
  ::std::ostringstream oss;
  oss << "::cuda::experimental::stf::static_box<";
  for (size_t ind : each(0, Dimensions))
  {
    if (ind > 0)
    {
      oss << ", ";
    }
    oss << b.get_begin(ind) << ", " << b.get_end(ind);
  }
  oss << ">";
  return oss.str();
}

template <typename MDS>
bool is_layout_right(const MDS& mds)
{
  const size_t rank = mds.rank();
  if (rank == 0)
  {
    return true;
  }
  if (mds.mapping().stride(rank - 1) != 1)
  {
    return false;
  }
  for (size_t i = 1; i < rank; ++i)
  {
    if (mds.mapping().stride(i - 1) != mds.mapping().stride(i) * mds.extent(i))
    {
      return false;
    }
  }
  return true;
}

template <typename MDS>
bool is_layout_left(const MDS& mds)
{
  const size_t rank = mds.rank();
  if (rank == 0)
  {
    return true;
  }
  if (mds.mapping().stride(0) != 1)
  {
    return false;
  }
  for (size_t i = 1; i < rank; ++i)
  {
    if (mds.mapping().stride(i) != mds.mapping().stride(i - 1) * mds.extent(i - 1))
    {
      return false;
    }
  }
  return true;
}

template <typename Mdspan, ::std::size_t... Is>
::std::string stringize_mdspan(const Mdspan& md, ::std::index_sequence<Is...> = ::std::index_sequence<>{})
{
  constexpr ::std::size_t R = Mdspan::rank();
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
    oss << "cuda::std::mdspan<" << type_name<ET>;

    // extents<size_t, e0, e1, ...>,
    oss << ", cuda::std::extents<size_t";
    if constexpr (R > 0)
    {
      oss << ", ";
    }

    ((oss << (Is ? ", " : "")
          << (XT::static_extent(Is) != cuda::std::dynamic_extent ? ::std::to_string(XT::static_extent(Is))
                                                                 : ::std::to_string(md.extent(Is)))),
     ...);

    oss << ">";

    // layout   (omit default)
    if constexpr (!std::is_same_v<Layout, cuda::std::layout_right>)
    {
      if (is_layout_right(md))
      {
        oss << ", cuda::std::layout_right";
      }
      else if (is_layout_left(md))
      {
        oss << ", cuda::std::layout_left";
      }
      else
      {
        throw ::std::runtime_error("Unsupported layout for mdspan: " + ::std::string(type_name<Layout>));
      }
    }
    else if constexpr (!std::is_same_v<Layout, cuda::std::layout_right>)
    {
      oss << ", " << type_name<Layout>;
    }

    // accessor (omit default)
    if constexpr (!std::is_same_v<Accessor, cuda::std::default_accessor<ET>>)
    {
      oss << ", " << type_name<Accessor>;
    }

    ::std::string out = oss.str();
    out += '>';

    return out;
  }
}

template <typename Mdspan, ::std::size_t... Is>
::std::string
stringize_mdspan_shape(const shape_of<Mdspan>& md_sh, ::std::index_sequence<Is...> = ::std::index_sequence<>{})
{
  constexpr ::std::size_t R = Mdspan::rank();
  if constexpr (R != sizeof...(Is))
  {
    return stringize_mdspan_shape(md_sh, ::std::make_index_sequence<R>{});
  }
  else
  {
    using ET       = typename Mdspan::element_type;
    using Layout   = typename Mdspan::layout_type;
    using Accessor = typename Mdspan::accessor_type;
    using XT       = typename Mdspan::extents_type;

    ::std::ostringstream oss;

    // mdspan<element_type,
    oss << "::cuda::experimental::stf::static_box<";

    ((oss << (Is ? ", " : "") << "0, "
          << (XT::static_extent(Is) != ::cuda::std::dynamic_extent ? ::std::to_string(XT::static_extent(Is))
                                                                   : ::std::to_string(md_sh.extent(Is)))),
     ...);

    oss << ">";

    return oss.str();
  }
}

namespace reserved
{
template <typename T>
struct jit_helper;

template <typename T, typename... P>
struct jit_helper<mdspan<T, P...>>
{
  using reduced_type = T*;

  static ::std::string stringize(const mdspan<T, P...>& m)
  {
    return stringize_mdspan(m);
  }

  static reduced_type reduce(const mdspan<T, P...>& m)
  {
    return m.data_handle();
  }
};

} // end namespace reserved

template <typename Mdspan>
::std::string shape_stringize(const shape_of<Mdspan>& md_sh)
{
  return stringize_mdspan_shape(md_sh);
}

//// TODO better type detection !
// template <typename Mdspan>
//::std::string stringize(const Mdspan& md) {
//     return stringize_mdspan(md);
//}
//
template <size_t Dimensions>
::std::string shape_stringize(const box<Dimensions>& b)
{
  return stringize_box(b);
}

template <typename shape_t, typename... Args>
::std::string parallel_for_template_generator(
  [[maybe_unused]] shape_t shape, const char* body_template, ::cuda::std::tuple<Args...> targs)
{
  ::std::ostringstream oss;

  // kernel arguments
  ::std::ostringstream args_prototype_oss;

  // Convert dynamic args to static args
  ::std::ostringstream args_conversion_oss;

  // this will contain "auto targs = make_tuple(...);"
  ::std::ostringstream args_tuple_oss;
  args_tuple_oss << "const auto targs = ::cuda::std::make_tuple(";

  each_in_tuple(targs, [&](auto arg_index, const auto& arg) {
    using raw_arg     = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<decltype(arg)>>;
    using reduced_arg = typename reserved::jit_helper<raw_arg>::reduced_type;

    if (arg_index > 0)
    {
      args_tuple_oss << ", ";
      args_prototype_oss << ", ";
    }

    // Pass the reduced version of the dynamic argument as an argument of the kernel
    args_prototype_oss << ::std::string(type_name<reduced_arg>) << " dyn_arg" << arg_index;

    // Convert the dynamic argument into its statically sized counterpart.
    args_conversion_oss << stringize_mdspan(arg) << " static_arg" << arg_index << "{dyn_arg" << arg_index << "};\n";

    args_tuple_oss << "static_arg" << arg_index;
  });

  args_tuple_oss << ");\n";

  // Note that we do not need the dynamic version of the shape at all
  args_conversion_oss << "const " << shape_stringize(shape) << " static_shape;\n";

  oss << "extern \"C\"\n";
  oss << "__global__ void %KERNEL_NAME%(" << args_prototype_oss.str() << ")\n";
  oss << "{\n";

  oss << args_conversion_oss.str() << ::std::endl;
  oss << args_tuple_oss.str() << ::std::endl;

  oss << "auto _body_impl = []" << body_template << ";\n";

  oss << R"(
           size_t _i          = blockIdx.x * blockDim.x + threadIdx.x;
           const size_t _step = blockDim.x * gridDim.x;
           const size_t n = static_shape.size();

           auto explode_args = [&](auto&&... data) {
             auto const explode_coords = [&](auto&&... coords) {
               _body_impl(coords..., data...);
             };
             // For every linearized index in the shape
             for (; _i < n; _i += _step)
             {
               ::cuda::std::apply(explode_coords, static_shape.index_to_coords(_i));
             }
           };
           ::cuda::std::apply(explode_args, ::std::move(targs));
)";

  oss << "}\n";

  return oss.str();
}

inline static const ::std::vector<::std::string>& get_nvrtc_flags()
{
  static ::std::vector<::std::string> flags = [] {
    ::std::vector<::std::string> result;
    result.push_back("-I../../libcudacxx/include");
    result.push_back("-I../../cudax/include/");
    result.push_back("-default-device");

    ::std::string s =
      run_command(R"(echo "" | nvcc -v -x cu - -c 2>&1 | grep '#$ INCLUDES="' | grep -oP '(?<=INCLUDES=").*(?=" *$)')");
    ::std::istringstream iss(s);
    result.insert(result.end(), ::std::istream_iterator<::std::string>{iss}, {});

    int device                = cuda_try<cudaGetDevice>();
    const cudaDeviceProp prop = cuda_try<cudaGetDeviceProperties>(device);
    result.push_back("--gpu-architecture=compute_" + ::std::to_string(prop.major) + ::std::to_string(prop.minor));

    return result;
  }();
  return flags;
}

template <typename context, typename exec_place_t, typename shape_t, typename... deps_ops_t>
struct parallel_for_scope_jit
{
  //  using deps_t = typename reserved::extract_all_first_types<deps_ops_t...>::type;
  // tuple<slice<double>, slice<int>> ...
  using deps_tup_t = ::std::tuple<typename deps_ops_t::dep_type...>;

  parallel_for_scope_jit(context& ctx, exec_place_t e_place, shape_t shape, deps_ops_t... deps)
      : deps(mv(deps)...)
      , ctx(ctx)
      , e_place(mv(e_place))
      , shape(mv(shape))
      , nvrtc_flags(get_nvrtc_flags())
  {}

  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto k = ::std::apply(
      [&](auto&&... unpacked_deps) {
        return ctx.cuda_kernel(::std::forward<decltype(unpacked_deps)>(unpacked_deps)...);
      },
      deps);

    if (!symbol.empty())
    {
      k.set_symbol(symbol);
    }

    k->*[&](auto... args) {
      ::std::pair<::std::string, ::std::string> f_res = f();

      auto gen_template =
        parallel_for_template_generator(shape, f_res.second.c_str(), ::cuda::std::make_tuple(args...));
      // ::std::cout << "->* GEN TEMPLATE ALL\n";
      // ::std::cout << gen_template << ::std::endl;
      // ::std::cout << "->* GEN TEMPLATE END\n";

      CUfunction kernel = lazy_jit(gen_template.c_str(), nvrtc_flags, f_res.first.c_str());
      // We do not pass the arguments directly, but only a "reduced" form
      // which contains all necessary information to build their static
      // counterpart
      return cuda_kernel_desc{kernel, 1280, 128, 0, reduce_for_jit(::std::forward<decltype(args)>(args))...};
    };
  }

private:
  // Get the "reduced" version of an argument so that we don't waste registers
  // passing arguments for which we only need a subset. For example, we only
  // need the data_handle() of an mdspan, not its extents because these are
  // already encoded in the static version of the mdspan.
  template <typename Arg>
  auto reduce_for_jit(Arg&& arg)
  {
    using raw_t = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Arg>>;
    return reserved::jit_helper<raw_t>::reduce(::std::forward<Arg>(arg));
  }

  ::std::tuple<deps_ops_t...> deps;
  context& ctx;
  exec_place_t e_place;
  ::std::string symbol;
  shape_t shape;
  ::std::vector<::std::string> nvrtc_flags;
};

template <typename Arg>
::std::string jit_reduced_type_name(const Arg&)
{
  using raw_arg     = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Arg>>;
  using reduced_arg = typename reserved::jit_helper<raw_arg>::reduced_type;
  return ::std::string(type_name<reduced_arg>);
}

template <typename Arg>
auto jit_reduce(Arg&& arg)
{
  using raw_arg = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Arg>>;
  return reserved::jit_helper<raw_arg>::reduce(::std::forward<Arg>(arg));
}

template <typename Arg>
::std::string jit_typename(const Arg& arg)
{
  using raw_arg = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Arg>>;
  return reserved::jit_helper<raw_arg>::stringize(arg);
}

} // end namespace cuda::experimental::stf
