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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <nvrtc.h>

#include <nvrtc/nvjitlink_helper.h>
#include <util/errors.h>

struct nvrtc_ptx
{
  std::unique_ptr<char[]> ptx{};
  size_t size;
};
struct nvrtc_link_result
{
  std::unique_ptr<char[]> data{};
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

  nvrtc_get_name() = delete;
  nvrtc_get_name(std::string_view name, std::string& lowered_name)
      : name(name)
      , lowered_name(lowered_name)
  {}
  ~nvrtc_get_name() noexcept {};

  nvrtc_get_name(const nvrtc_get_name&) = delete;
  nvrtc_get_name(nvrtc_get_name&& other) noexcept
      : name(other.name)
      , lowered_name(other.lowered_name)
  {}

  nvrtc_get_name& operator=(const nvrtc_get_name&) = delete;
  nvrtc_get_name& operator=(nvrtc_get_name&&)      = delete;
};
struct nvrtc_compile
{
  const char** args;
  size_t num_args;
};
struct nvrtc_get_ptx
{
  nvrtc_ptx& ptx_ref;
};
struct nvrtc_program_cleanup
{};
struct nvrtc_ltoir
{
  const char* ltoir;
  size_t ltsz;
};
using nvrtc_ltoir_list = std::vector<nvrtc_ltoir>;
struct nvrtc_jitlink_cleanup
{
  nvrtc_link_result& link_result_ref;
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

struct nvrtc_compile_command_list_visitor
{
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
  void execute(nvrtc_get_ptx ptx)
  {
    std::size_t ptx_size{};
    check(nvrtcGetPTXSize(program, &ptx_size));
    ptx.ptx_ref.ptx = std::unique_ptr<char[]>{new char[ptx_size]};
    check(nvrtcGetPTX(program, ptx.ptx_ref.ptx.get()));
  }
  void execute(nvrtc_program_cleanup)
  {
    nvrtcDestroyProgram(&program);
  }
};

struct nvrtc_link_command_list_visitor
{
  nvrtc_jitlink& jitlink;
  std::string_view program_name              = "test";
  nvrtc_compile_command_list_visitor compile = {};

  template <typename T, typename... Tx>
  void operator()(T&& t, Tx&&... rest)
  {
    execute(std::forward<T>(t));
    operator()(std::forward<Tx>(rest)...);
  }
  void operator()() {}

  void execute(nvrtc_translation_unit p)
  {
    compile.execute(p);
  }
  void execute(nvrtc_expression e)
  {
    compile.execute(e);
  }
  void execute(nvrtc_compile c)
  {
    compile.execute(c);
  }
  void execute(nvrtc_get_name gn)
  {
    compile.execute(std::move(gn));
  }
  void execute(nvrtc_program_cleanup cl)
  {
    std::size_t ltoir_size{};
    check(nvrtcGetLTOIRSize(compile.program, &ltoir_size));
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    check(nvrtcGetLTOIR(compile.program, ltoir.get()));

    check(nvJitLinkAddData(jitlink.handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, program_name.data()));

    compile.execute(cl);
  }
  void execute(nvrtc_ltoir lto)
  {
    check(
      nvJitLinkAddData(jitlink.handle, NVJITLINK_INPUT_LTOIR, (const void*) lto.ltoir, lto.ltsz, program_name.data()));
  }
  void execute(const nvrtc_ltoir_list& lto_list)
  {
    for (auto lto : lto_list)
    {
      execute(lto);
    }
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

    bool output_ptx = false;
    auto result     = nvJitLinkGetLinkedCubinSize(jitlink.handle, &cleanup.link_result_ref.size);
    if (result != NVJITLINK_SUCCESS)
    {
      output_ptx = true;
      check(nvJitLinkGetLinkedPtxSize(jitlink.handle, &cleanup.link_result_ref.size));
    }
    cleanup.link_result_ref.data = std::unique_ptr<char[]>(new char[cleanup.link_result_ref.size]);

    if (output_ptx)
    {
      check(nvJitLinkGetLinkedPtx(jitlink.handle, cleanup.link_result_ref.data.get()));
    }
    else
    {
      check(nvJitLinkGetLinkedCubin(jitlink.handle, cleanup.link_result_ref.data.get()));
    }
  }
};

template <typename... Tx, typename... Ts>
std::tuple<Tx..., Ts...> nvrtc_command_list_append(std::tuple<Tx...>&& tup, Ts&&... as)
{
  return std::tuple_cat(std::forward<std::tuple<Tx...>>(tup), std::forward_as_tuple(std::forward<Ts>(as)...));
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
  // Compile program to ptx
  // This ends the chain, similarly to finalize_program
  nvrtc_ptx compile_program_to_ptx(nvrtc_compile arg)
  {
    nvrtc_ptx ret;
    nvrtc_get_ptx get_ptx{ret};
    std::apply(nvrtc_compile_command_list_visitor{},
               nvrtc_command_list_append(std::move(cl), arg, std::move(get_ptx), nvrtc_program_cleanup{}));

    return ret;
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
  // Add linkable units to whole program
  nvrtc_sm_top_level<Tx..., nvrtc_ltoir_list> add_link_list(nvrtc_ltoir_list arg)
  {
    return {nvrtc_command_list_append(std::move(cl), std::move(arg))};
  }

  // Execute steps and link unit
  nvrtc_link_result finalize_program(uint32_t numLtoOpts, const char** ltoOpts)
  {
    nvrtc_link_result link_result{};
    nvrtc_jitlink_cleanup cleanup{link_result};
    nvrtc_jitlink jl(numLtoOpts, ltoOpts);
    std::apply(nvrtc_link_command_list_visitor{jl}, nvrtc_command_list_append(std::move(cl), std::move(cleanup)));
    return link_result;
  }
};

static nvrtc_sm_top_level<> make_nvrtc_command_list()
{
  return {};
}
