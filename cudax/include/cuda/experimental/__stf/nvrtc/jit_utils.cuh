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
#include <filesystem>
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
      static_assert(::std::is_arithmetic_v<T> || ::std::is_pointer_v<T>, "Unsupported type for JIT kernel argument");
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
    size_t log_size = cuda_try<nvrtcGetProgramLogSize>(prog);
    ::std::string log(log_size, '\0');
    cuda_try(nvrtcGetProgramLog(prog, log.data()));
    ::std::cerr << "NVRTC compile error:\n" << log << ::std::endl;
    ::std::exit(1);
  }

  size_t ptx_size = cuda_try<nvrtcGetPTXSize>(prog);
  ::std::string ptx(ptx_size, '\0');
  cuda_try(nvrtcGetPTX(prog, ptx.data()));
  cuda_try(nvrtcDestroyProgram(&prog));

  CUmodule cuda_module = cuda_try<cuModuleLoadData>(ptx.data());
  CUfunction kernel    = cuda_try<cuModuleGetFunction>(cuda_module, kernel_name.c_str());

  if (getenv("CUDASTF_JIT_DEBUG_PTX") || getenv("CUDASTF_JIT_DEBUG"))
  {
    int num_registers = cuda_try<cuFuncGetAttribute>(CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    ::std::cerr
      << "kernel_name " << kernel_name << " using " << ::std::to_string(num_registers) << " registers" << ::std::endl;
    ::std::cerr << "SOURCE BEGIN:\n";
    ::std::cerr << template_with_name << ::std::endl;
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

/**
 * @brief JIT adapter for kernel parameters (see specializations below).
 */
template <typename T>
struct jit_adapter;

template <typename T>
jit_adapter(T) -> jit_adapter<T>;

/**
 * @brief JIT adapter specialization for `mdspan` types.
 *
 * This struct defines how to bridge between a statically known `mdspan` type on the caller side,
 * a simplified pointer-based parameter type passed to a JIT-compiled kernel, and a reconstructed
 * `mdspan` type on the kernel side with as many static properties as possible.
 *
 * There are three types involved. First, `caller_side_t` is the type of the data as seen from
 * the code that launches the kernel. Second, `kernel_param_t` is the type of the kernel parameter,
 * which is part of the kernel signature. That type must be trivially copyable because the runtime
 * passes it to the kernel by bitwise copying. Finally, the kernel parameter is converted inside
 * the kernel code to `kernel_side_t_name()` which has extensive static type information. That type
 * is not known during compilation of the caller, and therefore it is accessible only as a string
 * from the caller side. That string will become part of the JITted kernel code. The kernel-side
 * type must be constructible from an object of type `kernel_param_t`.
 *
 * @tparam T Element type of the `mdspan`
 * @tparam P Pack of extents, layout, and accessor types comprising the full `mdspan` signature
 */
template <typename T, typename... P>
struct jit_adapter<mdspan<T, P...>>
{
  /// @brief Type of the argument as seen by the caller (typically with dynamic extents and strides)
  using caller_side_t = mdspan<T, P...>;

  /// @brief Type of the kernel parameter (raw pointer to the element type)
  using kernel_param_t = T*;

  /**
   * @brief Constructs a JIT adapter from the caller-side `mdspan` object.
   *
   * @param md The `mdspan` object representing the argument passed to the kernel.
   */
  jit_adapter(const caller_side_t& md)
      : caller_side_arg(md)
  {}

  /**
   * @brief Returns the string representation of the caller-side `mdspan` type.
   *
   * This is useful for diagnostics or logging, and simply emits the static type as known at the call site.
   *
   * @return A `std::string` with the fully qualified name of the caller-side `mdspan` type.
   */
  static ::std::string caller_side_t_name()
  {
    return ::std::string(type_name<caller_side_t>);
  }

  /**
   * @brief Returns the kernel-side `mdspan` type name with all known compile-time information.
   *
   * The returned string describes a new `mdspan` instantiation with:
   * - the same element type
   * - extents statically specified if they are known at runtime
   * - a static layout (right or left) if the runtime mapping is compatible
   * - the accessor if it differs from the default
   *
   * @throws std::runtime_error if the layout is neither `layout_right` nor `layout_left`
   *
   * @return A `std::string` representing a kernel-side `mdspan` type suitable for code generation
   */
  ::std::string kernel_side_t_name()
  {
    auto const R   = caller_side_arg.rank();
    using ET       = typename caller_side_t::element_type;
    using Layout   = typename caller_side_t::layout_type;
    using Accessor = typename caller_side_t::accessor_type;
    using XT       = typename caller_side_t::extents_type;

    ::std::ostringstream oss;

    // Emit element type
    oss << "cuda::std::mdspan<" << type_name<ET>;

    // Emit extents with as many static values as possible
    oss << ", cuda::std::extents<size_t";
    if (R > 0)
    {
      oss << ", ";
    }

    for (size_t i = 0; i < R; i++)
    {
      oss << (i ? ", " : "")
          << (XT::static_extent(i) != cuda::std::dynamic_extent
                ? ::std::to_string(XT::static_extent(i))
                : ::std::to_string(caller_side_arg.extent(i)));
    }

    oss << ">";

    // Emit layout only if not default
    if constexpr (!std::is_same_v<Layout, cuda::std::layout_right>)
    {
      if (is_layout_right(caller_side_arg))
      {
        oss << ", cuda::std::layout_right";
      }
      else if (is_layout_left(caller_side_arg))
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

    // Emit accessor only if not default
    if constexpr (!std::is_same_v<Accessor, cuda::std::default_accessor<ET>>)
    {
      oss << ", " << type_name<Accessor>;
    }

    ::std::string out = oss.str();
    out += '>';

    return out;
  }

  /**
   * @brief Converts the caller-side object to the kernel parameter form (raw pointer).
   *
   * This extracts the underlying data pointer from the `mdspan` so that it can be passed
   * as a kernel argument via the CUDA driver API.
   *
   * @return The raw pointer to the first element in the `mdspan`.
   */
  kernel_param_t to_kernel_arg()
  {
    return caller_side_arg.data_handle();
  }

  ::std::string kernel_param_t_name()
  {
    return ::std::string(type_name<kernel_param_t>);
  }

private:
  /// The `mdspan` object passed from the host side and stored for introspection
  const caller_side_t caller_side_arg;
};

template <size_t dimensions>
struct jit_adapter<box<dimensions>>
{
  using caller_side_t = box<dimensions>;

  jit_adapter(const caller_side_t& rhs)
      : caller_side_arg(rhs)
  {}

  ::std::string kernel_side_t_name()
  {
    ::std::ostringstream oss;
    oss << "::cuda::experimental::stf::static_box<";
    unroll<dimensions>([&](auto i) {
      if (i > 0)
      {
        oss << ", ";
      }
      oss << caller_side_arg.get_begin(i) << ", " << caller_side_arg.get_end(i);
    });
    oss << ">";
    return oss.str();
  }

private:
  const caller_side_t caller_side_arg;
};

template <typename T, typename... P>
struct jit_adapter<shape_of<mdspan<T, P...>>>
{
  using caller_side_t = shape_of<mdspan<T, P...>>;

  jit_adapter(caller_side_t rhs)
      : caller_side_arg(rhs)
  {}

  ::std::string kernel_side_t_name()
  {
    constexpr auto R = mdspan<T, P...>::rank();
    using ET         = typename mdspan<T, P...>::element_type;
    using Layout     = typename mdspan<T, P...>::layout_type;
    using Accessor   = typename mdspan<T, P...>::accessor_type;
    using XT         = typename mdspan<T, P...>::extents_type;

    ::std::ostringstream oss;

    // mdspan<element_type,
    oss << "::cuda::experimental::stf::static_box<";

    unroll<R>([&](auto i) {
      oss << (i ? ", " : "") << "0, "
          << (XT::static_extent(i) != ::cuda::std::dynamic_extent
                ? ::std::to_string(XT::static_extent(i))
                : ::std::to_string(caller_side_arg.extent(i)));
    });

    oss << ">";

    return oss.str();
  }

private:
  const caller_side_t caller_side_arg;
};

template <typename shape_t, typename... Args>
::std::string parallel_for_template_generator(
  [[maybe_unused]] shape_t shape, const char* body_template, ::cuda::std::tuple<Args...> targs)
{
  ::std::ostringstream oss;
  // kernel arguments
  ::std::ostringstream types_and_params_list;
  // Convert dynamic args to static args
  ::std::ostringstream types_list;
  // this will contain "auto targs = static_tuple(...);"
  ::std::ostringstream param_list;

  each_in_tuple(targs, [&](auto i, const auto& arg) {
    using raw_arg        = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<decltype(arg)>>;
    using kernel_param_t = typename jit_adapter<raw_arg>::kernel_param_t;

    if (i > 0)
    {
      param_list << ", ";
      types_and_params_list << ", ";
      types_list << ", ";
    }

    // Pass the to_kernel_argd version of the dynamic argument as an argument of the kernel
    types_and_params_list << ::std::string(type_name<kernel_param_t>) << " dyn_arg" << i;
    // Convert the dynamic argument into its statically sized counterpart.
    types_list << jit_adapter<raw_arg>(arg).kernel_side_t_name();
    param_list << "dyn_arg" << i;
  });

  // Note that we do not need the dynamic version of the shape at all

  oss
    << "#include <cuda/experimental/__stf/nvrtc/jit_loop.cuh>\n"
    << "extern \"C\"\n"
    << "__global__ void %KERNEL_NAME%(" << types_and_params_list.str() << ")\n"
    << "{\n"
    << "  using static_shape_t = " << jit_adapter<shape_t>(shape).kernel_side_t_name() << ";\n"
    << "  using static_tuple = ::cuda::std::tuple<" << types_list.str() << ">;\n"
    << "  ::cuda::experimental::stf::reserved::jit_loop<static_shape_t>([]" << body_template << ",\n    static_tuple("
    << param_list.str() << "));\n"
    << "}\n";

  ::std::cerr << oss.str();

  return oss.str();
}

inline static const ::std::vector<::std::string>& get_nvrtc_flags()
{
  static ::std::vector<::std::string> flags = [] {
    namespace fs = ::std::filesystem;

    // Returns the n-th parent of a path.
    // For example: with path "foo/bar/baz/bla.cu" and n = 2, returns "foo/bar"
    const auto get_nth_parent = [](::std::filesystem::path p, int n) -> ::std::filesystem::path {
      while (n-- > 0 && !p.empty())
      {
        p = p.parent_path();
      }
      return p;
    };

    // cudax/include/cuda/experimental/__stf/nvrtc/jit_utils.cuh
    fs::path this_file     = __FILE__;
    fs::path cccl_root_dir = get_nth_parent(this_file, 7);

    ::std::vector<::std::string> result;
    result.push_back("-I" + (cccl_root_dir / "libcudacxx/include").string());
    result.push_back("-I" + (cccl_root_dir / "cudax/include").string());
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

  const char* default_header_template = R"(
      #include <cuda/experimental/__stf/nvrtc/slice.cuh>
  )";

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

    ::std::string header;
    ::std::string body;

    k->*[&](auto... args) {
      using f_return_t = decltype(f());
      // If f only returns a string, use a default header
      if constexpr (::std::is_same_v<f_return_t, ::std::string>)
      {
        body   = f();
        header = ::std::string(default_header_template);
      }
      else
      {
        auto&& [header_, body_] = f();
        header                  = mv(header_);
        body                    = mv(body_);
      }

      auto gen_template = parallel_for_template_generator(shape, body.c_str(), ::cuda::std::make_tuple(args...));
      // ::std::cout << "->* GEN TEMPLATE ALL\n";
      // ::std::cout << gen_template << ::std::endl;
      // ::std::cout << "->* GEN TEMPLATE END\n";

      CUfunction kernel = lazy_jit(gen_template.c_str(), nvrtc_flags, header.c_str());
      // We do not pass the arguments directly, but only a "to_kernel_argd" form
      // which contains all necessary information to build their static
      // counterpart
      return cuda_kernel_desc{kernel, 1280, 128, 0, to_kernel_arg_for_jit(::std::forward<decltype(args)>(args))...};
    };
  }

private:
  // Get the "to_kernel_argd" version of an argument so that we don't waste registers
  // passing arguments for which we only need a subset. For example, we only
  // need the data_handle() of an mdspan, not its extents because these are
  // already encoded in the static version of the mdspan.
  template <typename Arg>
  auto to_kernel_arg_for_jit(Arg&& arg)
  {
    using raw_t = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Arg>>;
    return jit_adapter<raw_t>(::std::forward<Arg>(arg)).to_kernel_arg();
  }

  ::std::tuple<deps_ops_t...> deps;
  context& ctx;
  exec_place_t e_place;
  ::std::string symbol;
  shape_t shape;
  ::std::vector<::std::string> nvrtc_flags;
};

} // end namespace cuda::experimental::stf
