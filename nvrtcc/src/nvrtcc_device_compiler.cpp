//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <nv_decode.h>
#include <nvrtc.h>

//! @brief Gets the dynamic handle to nvrtc library. The first call must supply the path to the library.
static void* get_nvrtc_lib(const char* nvrtc_lib = nullptr)
{
  static const auto handle = dlopen(nvrtc_lib, RTLD_NOW);
  if (handle == nullptr)
  {
    std::fprintf(stderr, "Failed to open nvrtc lib.\n");
    std::exit(-1);
  }
  return handle;
}

//! @brief Gets the pointer to a nvrtc function.
template <class Signature>
[[nodiscard]] static Signature* get_nvrtc_api(const char* api)
{
  const auto handle = dlsym(get_nvrtc_lib(), api);
  if (handle == nullptr)
  {
    std::fprintf(stderr, "Failed to get nvrtc function: %s.\n", api);
    std::exit(-1);
  }
  return reinterpret_cast<Signature*>(handle);
}

#define CALL_NVRTC_UNCHECKED(API, ...) get_nvrtc_api<decltype(API)>(#API)(__VA_ARGS__)

#define CALL_NVRTC(API, ...)                                                                                       \
  do                                                                                                               \
  {                                                                                                                \
    nvrtcResult _ret = CALL_NVRTC_UNCHECKED(API, __VA_ARGS__);                                                     \
    if (_ret != NVRTC_SUCCESS)                                                                                     \
    {                                                                                                              \
      std::fprintf(                                                                                                \
        stderr, "%s(%d): NVRTC error: %s\n", __FILE__, __LINE__, CALL_NVRTC_UNCHECKED(nvrtcGetErrorString, _ret)); \
      std::exit(-1);                                                                                               \
    }                                                                                                              \
  } while (false)

//! @brief Read the file's contents to a string.
[[nodiscard]] static std::string read_input(const char* file)
{
  std::ifstream ifs{file};
  ifs.seekg(0, std::ios::end);
  const std::size_t size = ifs.tellg();

  std::string buffer(size, '\0');

  ifs.seekg(0);
  ifs.read(buffer.data(), size);
  buffer[size] = '\0';

  return buffer;
}

//! @brief Returns a view to a line from a string_view.
[[nodiscard]] static std::string_view get_line(std::string_view s) noexcept
{
  const auto pos = s.find('\n');
  return (pos == std::string_view::npos) ? s : s.substr(0, pos + 1);
}

//! @brief Returns \c true if \c s starts with \c with.
[[nodiscard]] static bool starts_with(std::string_view s, std::string_view with)
{
  return s.substr(0, std::min(s.size(), with.size())) == with;
}

//! @brief Type of PTX symbol.
enum class SymbolType
{
  none, //!< Empty/invalid state.
  variable, //!< Variable (.global).
  kernel, //!< Kernel (.entry).
};

//! @brief Structure with data for PTX symbol.
struct Symbol
{
  SymbolType type; //!< Symbol type.
  std::string_view name; //!< Symbol name.
};

//! @brief Extracts PTX symbol from line.
[[nodiscard]] static Symbol extract_symbol_from_line(std::string_view line) noexcept
{
  constexpr std::string_view variable_prefix = ".global ";
  constexpr std::string_view kernel_prefix   = ".entry ";

  std::string_view symbol{};
  SymbolType symbol_type{SymbolType::none};

  if (starts_with(line, variable_prefix))
  {
    symbol_type = SymbolType::variable;

    // Globals have format of ".global .align 4 .u32 _Z3varIiE[] = { ... };".

    // 1. Remove everything after ';'.
    symbol = line.substr(0, line.find(';'));

    // 2. If the variable is an array, we find the symbol end by searching for '['.
    symbol = symbol.substr(0, symbol.find('['));

    // 3. If the variable is not an array is initialized, find the symbol end by searching for ' ='.
    symbol = symbol.substr(0, symbol.find(" ="));

    // 4. We should have the end of the symbol, remove everything in front of it.
    symbol.remove_prefix(symbol.rfind(' ') + 1);

    // NVVM emits $str and __unnamed_N symbols for some debug data, let's ignore those.
    if (starts_with(symbol, "$") || starts_with(symbol, "__unnamed"))
    {
      symbol_type = SymbolType::none;
    }
  }
  else if (line.find(kernel_prefix) != std::string_view::npos)
  {
    symbol_type = SymbolType::kernel;

    // Entries have format of ".visible .entry _Z6squareIiEvPT_i(" and end with ")" if there are no parameters.

    // 1. Discard everything after last "("
    symbol = line.substr(0, line.find_first_of('('));

    // 2. Remove everything in front of the symbol.
    symbol.remove_prefix(symbol.rfind(' ') + 1);
  }

  return {symbol_type, symbol};
}

//! @brief Extracts name expression in form `symbol<template_args>` from symbol in `void symbol<template_args>(args)`
//! form.
//!
//! @return Pointer to first character of a zero terminated string or `nullptr` if the symbol is not a template.
//!
//! @note This function modifies the symbol buffer.
[[nodiscard]] static char* extract_name_expr(char* symbol, SymbolType symbol_type) noexcept
{
  if (symbol_type == SymbolType::kernel)
  {
    // Remove 'void ' prefix.
    symbol += 5;
  }

  // Iterate over the name
  std::size_t curly_parens{0};
  std::size_t template_parens{0};
  bool is_template{false};
  for (char* it = symbol; *it != '\0'; ++it)
  {
    switch (*it)
    {
      case '(':
        if (template_parens == 0)
        {
          *it = '\0';
          return (is_template) ? symbol : nullptr;
        }
        ++curly_parens;
        break;
      case ')':
        --curly_parens;
        break;
      case '<':
        ++template_parens;
        is_template = true;
        break;
      case '>':
        --template_parens;

        // When we are outside of template parens, and next 2 characters are ::, skip them and reset is_template,
        // because we were parsing type namespace
        if (template_parens == 0)
        {
          if (it[1] == ':' && it[2] == ':')
          {
            is_template = false;
            it += 2;
          }
        }
        break;
      default:
        break;
    }
  }
  return nullptr;
}

//! @brief Adds symbol to the program.
static void add_symbol(nvrtcProgram prog, Symbol symbol)
{
  struct CuDemangleBuffer
  {
    char* ptr{};
    std::size_t size{};

    ~CuDemangleBuffer()
    {
      std::free(ptr);
    }
  };

  static std::string symbol_copy{};
  static CuDemangleBuffer buffer{};

  // If the symbol is not mangled, it's not a template, thus we even needn't to add it as a name expression.
  if (!starts_with(symbol.name, "_Z"))
  {
    return;
  }

  symbol_copy = symbol.name;

  int status;
  buffer.ptr = __cu_demangle(symbol_copy.c_str(), buffer.ptr, &buffer.size, &status);
  switch (status)
  {
    case 0:
      break;
    case -1:
      throw std::bad_alloc{};
    case -2:
      throw std::invalid_argument{std::string{"invalid symbol '"} + std::string{symbol.name} + "'"};
    case -3:
      throw std::invalid_argument{"invalid __cu_demangle parameter"};
    default:
      throw std::runtime_error{"unknown __cu_demangle error"};
  }

  char* name_expr = extract_name_expr(buffer.ptr, symbol.type);
  if (name_expr != nullptr)
  {
    CALL_NVRTC(nvrtcAddNameExpression, prog, name_expr);
  }
}

//! @brief Adds all symbols from a given PTX input to the program.
static void add_symbols(nvrtcProgram prog, std::string_view ptx_input)
{
  if (ptx_input.find(-1))
  {
    while (!ptx_input.empty())
    {
      const auto line = get_line(ptx_input);

      // All symbols start with "." as the first character on line.
      if (!line.empty() && line[0] == '.')
      {
        const auto symbol = extract_symbol_from_line(line);

        if (symbol.type != SymbolType::none)
        {
          add_symbol(prog, symbol);
        }
      }

      ptx_input.remove_prefix(line.size());
    }
  }
}

int main(int argc, const char* const* argv)
{
  auto arg_it = argv + 1;

  // Extract positional arguments.
  const auto nvrtc_lib   = *arg_it++;
  const auto input_file  = *arg_it++;
  const auto input_name  = *arg_it++;
  const auto output_file = *arg_it++;
  const auto arch_list   = *arg_it++;

  // Open nvrtc_lib shared object.
  get_nvrtc_lib(nvrtc_lib);

  // Read the source file.
  const auto input = read_input(input_file);

  // Read the input PTX file.
  const auto ptx_input = read_input(output_file);

  // Remove the input PTX file.
  std::remove(output_file);

  // Create arch list include.
  const auto arch_list_include = std::string{"#undef __CUDA_ARCH_LIST__\n#define __CUDA_ARCH_LIST__ "} + arch_list;

  constexpr auto arch_list_include_name = "__nvrtcc_arch_list_include";

  // Create list of options to be passed to nvrtc.
  std::vector<const char*> opts{};
  opts.reserve(argv + argc - arg_it + 2);
  opts.push_back("-include");
  opts.push_back(arch_list_include_name);
  opts.insert(opts.end(), arg_it, argv + argc);

  // Create list of headers to be used by NVRTC.
  std::array headers{arch_list_include.c_str()};
  std::array headers_names{arch_list_include_name};
  static_assert(headers.size() == headers_names.size());

  // Create nvrtc program.
  nvrtcProgram prog{};
  CALL_NVRTC(nvrtcCreateProgram,
             &prog,
             input.data(),
             input_name,
             static_cast<int>(headers.size()),
             headers.data(),
             headers_names.data());

  // Add symbols to the program.
  add_symbols(prog, ptx_input);

  // Compile the program.
  const auto compile_result =
    CALL_NVRTC_UNCHECKED(nvrtcCompileProgram, prog, static_cast<int>(opts.size()), opts.data());

  // Obtain the log size.
  std::size_t log_size{};
  CALL_NVRTC(nvrtcGetProgramLogSize, prog, &log_size);

  // Get the log and output it to the stderr. The log always contains EOF, so we check for > 1.
  if (log_size > 1)
  {
    auto log = std::make_unique<char[]>(log_size);
    CALL_NVRTC(nvrtcGetProgramLog, prog, log.get());
    std::fprintf(stderr, "%s\n", log.get());
  }

  // If the compilation failed, exit.
  if (compile_result != NVRTC_SUCCESS)
  {
    std::exit(1);
  }

  // Get the ptx size.
  std::size_t ptx_size{};
  CALL_NVRTC(nvrtcGetPTXSize, prog, &ptx_size);

  // Get the ptx code.
  auto ptx = std::make_unique<char[]>(ptx_size);
  CALL_NVRTC(nvrtcGetPTX, prog, ptx.get());

  // Write the ptx to the output file. The code contains EOF, so we write one character less.
  std::ofstream ofs{output_file};
  ofs.write(ptx.get(), ptx_size - 1);

  // Destroy the program.
  CALL_NVRTC(nvrtcDestroyProgram, &prog);
}
