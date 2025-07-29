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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

#include <nvrtc.h>

#include <nvrtc/command_list_mixins.h>
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
  nvJitLinkHandle handle{};

  nvrtc_jitlink()                     = delete;
  nvrtc_jitlink(const nvrtc_jitlink&) = delete;
  nvrtc_jitlink(nvrtc_jitlink&& rhs)
  {
    handle     = rhs.handle;
    rhs.handle = nullptr;
  };

  nvrtc_jitlink(uint32_t numOpts, const char** opts)
      : handle(nullptr)
  {
    if (opts)
    {
      nvJitLinkCreate(&handle, numOpts, opts);
    }
  }

  ~nvrtc_jitlink()
  {
    if (handle)
    {
      nvJitLinkDestroy(&handle);
    }
  }
};

struct nvrtc2_top_level;
struct nvrtc2_link_result;
struct nvrtc2_pre_build;
struct nvrtc2_post_build;

struct nvrtc2_lto_context
{
  nvrtc_jitlink jit;
  nvrtcProgram program{};
  std::string_view program_name = "test";
};

using nvrtc2_top_level_nl   = node_list<nvrtc2_lto_context, nvrtc2_top_level>;
using nvrtc2_pre_build_nl   = node_list<nvrtc2_lto_context, nvrtc2_pre_build>;
using nvrtc2_link_result_nl = node_list<nvrtc2_lto_context, nvrtc2_link_result>;
using nvrtc2_post_build_nl  = node_list<nvrtc2_lto_context, nvrtc2_post_build>;

struct nvrtc2_post_build
{
  nvrtc2_lto_context& context;
  nvrtc2_post_build(nvrtc2_lto_context& c_)
      : context(c_)
  {}

  inline nvrtc2_post_build_nl get_name(nvrtc_get_name gn)
  {
    const char* lowered_name;
    check(nvrtcGetLoweredName(context.program, gn.name.data(), &lowered_name));
    gn.lowered_name = lowered_name;
    return {std::move(context)};
  }

  inline nvrtc_ptx get_program_ptx()
  {
    nvrtc_ptx ret;
    check(nvrtcGetPTXSize(context.program, &ret.size));
    ret.ptx = std::unique_ptr<char[]>{new char[ret.size]};
    check(nvrtcGetPTX(context.program, ret.ptx.get()));

    cleanup_program();
    return ret;
  }

  inline nvrtc2_top_level_nl link_program()
  {
    std::size_t ltoir_size{};
    check(nvrtcGetLTOIRSize(context.program, &ltoir_size));
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    check(nvrtcGetLTOIR(context.program, ltoir.get()));
    check(nvJitLinkAddData(
      context.jit.handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, context.program_name.data()));

    return cleanup_program();
  }

private:
  inline nvrtc2_top_level_nl cleanup_program()
  {
    nvrtcDestroyProgram(&context.program);
    return {std::move(context)};
  }
};

struct nvrtc2_pre_build
{
  nvrtc2_lto_context& context;
  nvrtc2_pre_build(nvrtc2_lto_context& c_)
      : context(c_)
  {}

  // Add expression before compiling (instantiates global kernel declared in unit)
  inline nvrtc2_pre_build_nl add_expression(nvrtc_expression arg)
  {
    check(nvrtcAddNameExpression(context.program, arg.expression.data()));
    return nvrtc2_pre_build_nl{std::move(context)};
  }
  // Compile program
  inline nvrtc2_post_build_nl compile_program(nvrtc_compile compile_args)
  {
    size_t n_actual_args = std::distance(
      compile_args.args,
      std::remove_if(compile_args.args, compile_args.args + compile_args.num_args, [](const char* ptr) -> bool {
        return (ptr == nullptr);
      }));
    nvrtcResult result = nvrtcCompileProgram(context.program, n_actual_args, compile_args.args);

    size_t log_size{};
    check(nvrtcGetProgramLogSize(context.program, &log_size));
    if (log_size > 1)
    {
      std::unique_ptr<char[]> log{new char[log_size]};
      check(nvrtcGetProgramLog(context.program, log.get()));
      std::cerr << log.get() << std::endl;
    }
    check(result);

    return {std::move(context)};
  }
};

struct nvrtc2_top_level
{
  nvrtc2_lto_context& context;
  nvrtc2_top_level(nvrtc2_lto_context& c_)
      : context(c_)
  {}

  // Compile and link program
  inline nvrtc2_pre_build_nl add_program(nvrtc_translation_unit tu)
  {
    check(nvrtcCreateProgram(&context.program, tu.program.data(), tu.name.data(), 0, nullptr, nullptr));
    return {std::move(context)};
  }

  // Add linkable unit to whole program
  inline nvrtc2_top_level_nl add_link(nvrtc_ltoir arg)
  {
    check(nvJitLinkAddData(
      context.jit.handle, NVJITLINK_INPUT_LTOIR, (const void*) arg.ltoir, arg.ltsz, context.program_name.data()));
    return {std::move(context)};
  }

  // Add linkable units to whole program
  inline nvrtc2_top_level_nl add_link_list(nvrtc_ltoir_list list)
  {
    for (const auto& lto : list)
    {
      check(nvJitLinkAddData(
        context.jit.handle, NVJITLINK_INPUT_LTOIR, (const void*) lto.ltoir, lto.ltsz, context.program_name.data()));
    }
    return {std::move(context)};
  }

  // Execute steps and link unit
  inline nvrtc_link_result finalize_program()
  {
    nvrtc_link_result link_result{};

    auto jitlink_error = nvJitLinkComplete(context.jit.handle);
    size_t log_size{};
    check(nvJitLinkGetErrorLogSize(context.jit.handle, &log_size));
    if (log_size > 1)
    {
      std::unique_ptr<char[]> log{new char[log_size]};
      check(nvJitLinkGetErrorLog(context.jit.handle, log.get()));
      std::cerr << log.get() << std::endl;
    }

    check(jitlink_error);

    bool output_ptx = false;
    auto result     = nvJitLinkGetLinkedCubinSize(context.jit.handle, &link_result.size);

    if (result != NVJITLINK_SUCCESS)
    {
      output_ptx = true;
      check(nvJitLinkGetLinkedPtxSize(context.jit.handle, &link_result.size));
    }
    link_result.data = std::unique_ptr<char[]>(new char[link_result.size]);

    if (output_ptx)
    {
      check(nvJitLinkGetLinkedPtx(context.jit.handle, link_result.data.get()));
    }
    else
    {
      check(nvJitLinkGetLinkedCubin(context.jit.handle, link_result.data.get()));
    }

    return link_result;
  }
};

inline nvrtc2_top_level_nl begin_linking_nvrtc_program(uint32_t numLtoOpts, const char** ltoOpts)
{
  nvrtc2_lto_context context{nvrtc_jitlink(numLtoOpts, ltoOpts)};

  return {std::move(context)};
}
