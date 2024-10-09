//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>

#include <nvJitLink.h>
#include <nvrtc.h>
#include <util/errors.h>

struct nvrtc_cubin
{
  std::unique_ptr<char[]> cubin{};
  size_t size;
};

struct nvrtc_translation_unit
{
  std::string_view program;
  std::string_view name;
};
struct nvrtc_expression
{
  std::string_view expression;
};
struct nvrtc_get_name
{
  std::string_view name;
  std::string& lowered_name;
};
struct nvrtc_compile
{
  const char** args;
  size_t num_args;
};
struct nvrtc_program_cleanup
{};
struct nvrtc_ltoir
{
  const char* ltoir;
  int ltsz;
};
struct nvrtc_jitlink_cleanup
{
  nvrtc_cubin& cubin_ref;
};

struct nvrtc_jitlink
{
  nvJitLinkHandle handle;

  nvrtc_jitlink(uint32_t numOpts, const char** opts)
  {
    nvJitLinkCreate(&handle, numOpts, opts);
  }

  ~nvrtc_jitlink()
  {
    nvJitLinkDestroy(&handle);
  }
};

struct nvrtc_command_list_visitor
{
  nvrtc_jitlink& jitlink;
  std::string_view program_name = "test";
  nvrtcProgram program{};

  template <typename T, typename... Tx>
  void operator()(T&& t, Tx&&... rest)
  {
    execute(std::forward<T>(t));
    operator()(std::forward<Tx>(rest)...);
  }
  void operator()() {}

  void execute(nvrtc_translation_unit p)
  {
    check(nvrtcCreateProgram(&program, p.program.data(), p.name.data(), 0, nullptr, nullptr));
  }
  void execute(nvrtc_expression e)
  {
    check(nvrtcAddNameExpression(program, e.expression.data()));
  }
  void execute(nvrtc_compile c)
  {
    auto result = nvrtcCompileProgram(program, c.num_args, c.args);

    size_t log_size{};
    check(nvrtcGetProgramLogSize(program, &log_size));
    if (log_size > 1)
    {
      std::unique_ptr<char[]> log{new char[log_size]};
      check(nvrtcGetProgramLog(program, log.get()));
      std::cerr << log.get() << std::endl;
    }

    check(result);
  }
  void execute(nvrtc_get_name gn)
  {
    const char* lowered_name;
    check(nvrtcGetLoweredName(program, gn.name.data(), &lowered_name));
    gn.lowered_name = lowered_name;
  }
  void execute(nvrtc_program_cleanup)
  {
    std::size_t ltoir_size{};
    check(nvrtcGetLTOIRSize(program, &ltoir_size));
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    check(nvrtcGetLTOIR(program, ltoir.get()));

    check(nvJitLinkAddData(jitlink.handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, program_name.data()));

    nvrtcDestroyProgram(&program);
  }
  void execute(nvrtc_ltoir lto)
  {
    check(nvJitLinkAddData(
      jitlink.handle, NVJITLINK_INPUT_LTOIR, (const void*) lto.ltoir, (size_t) lto.ltsz, program_name.data()));
  }
  void execute(nvrtc_jitlink_cleanup cleanup)
  {
    auto jitlink_error = nvJitLinkComplete(jitlink.handle);

    size_t log_size{};
    check(nvJitLinkGetErrorLogSize(jitlink.handle, &log_size));
    if (log_size > 1)
    {
      std::unique_ptr<char[]> log{new char[log_size]};
      check(nvJitLinkGetErrorLog(jitlink.handle, log.get()));
      std::cerr << log.get() << std::endl;
    }

    check(jitlink_error);

    check(nvJitLinkGetLinkedCubinSize(jitlink.handle, &cleanup.cubin_ref.size));
    cleanup.cubin_ref.cubin = std::unique_ptr<char[]>(new char[cleanup.cubin_ref.size]);
    check(nvJitLinkGetLinkedCubin(jitlink.handle, cleanup.cubin_ref.cubin.get()));
  }
};

template <typename... Tx, typename T>
std::tuple<Tx..., T> nvrtc_command_list_append(std::tuple<Tx...>&& tup, T&& a)
{
  return std::tuple_cat(std::forward<std::tuple<Tx...>>(tup), std::make_tuple(std::forward<T>(a)));
}

template <typename... Tx>
struct nvrtc_sm_top_level;
template <typename... Tx>
struct nvrtc_sm_cleanup_tu;

template <typename... Tx>
struct nvrtc_sm_compilation_unit
{
  using command_list = std::tuple<Tx...>;
  command_list cl{};

  // Add expression before compiling (instantiates global kernel declared in unit)
  nvrtc_sm_compilation_unit<Tx..., nvrtc_expression> add_expression(nvrtc_expression arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }
  // Compile program
  nvrtc_sm_cleanup_tu<Tx..., nvrtc_compile> compile_program(nvrtc_compile arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }
};

template <typename... Tx>
struct nvrtc_sm_cleanup_tu
{
  using command_list = std::tuple<Tx...>;
  command_list cl{};

  nvrtc_sm_cleanup_tu<Tx..., nvrtc_get_name> get_name(nvrtc_get_name arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }
  // Compile program
  nvrtc_sm_top_level<Tx..., nvrtc_program_cleanup> cleanup_program()
  {
    return {nvrtc_command_list_append(std::move(cl), nvrtc_program_cleanup{})};
  }
};

template <typename... Tx>
struct nvrtc_sm_top_level
{
  using command_list = std::tuple<Tx...>;
  command_list cl{};

  // Multiple programs may be linked together
  nvrtc_sm_compilation_unit<Tx..., nvrtc_translation_unit> add_program(nvrtc_translation_unit arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }
  // Add linkable unit to whole program
  nvrtc_sm_top_level<Tx..., nvrtc_ltoir> add_link(nvrtc_ltoir arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }

  // Execute steps and link unit
  nvrtc_cubin finalize_program(uint32_t numLtoOpts, const char** ltoOpts)
  {
    nvrtc_cubin cubin{};
    nvrtc_jitlink_cleanup cleanup{cubin};
    nvrtc_jitlink jl(numLtoOpts, ltoOpts);
    std::apply(nvrtc_command_list_visitor{jl}, nvrtc_command_list_append(std::move(cl), std::move(cleanup)));
    return cubin;
  }
};

static nvrtc_sm_top_level<> make_nvrtc_command_list()
{
  return {};
}
