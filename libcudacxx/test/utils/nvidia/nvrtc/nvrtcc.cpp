//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <deque>
#include <functional>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "nvrtcc_build.h"
#include "nvrtcc_run.h"
#include "utils/platform.h"
#include <stdio.h>

ArgList nvrtcArguments;
ArgList ignoredArguments;
std::string outputDir;
std::string outputFile;
std::string inputFile;

bool skipOutput = false;
bool building   = false;
bool execute    = false;

ExecutionConfig executionConfig;

enum ArgProcessorState
{
  NORMAL,
  GREEDY,
  ABORT,
};

// Handlers may increment the iterator of the argument parser if they need multiple arguments
// First argument is incrementable, last argument is the end of the list
using ArgHandler    = std::function<ArgProcessorState(const std::smatch&)>;
using ArgPair       = std::pair<std::regex, ArgHandler>;
using ArgHandlerMap = std::vector<ArgPair>;

int g_argc;
char** g_argv;

// Ignore PTX arch, only capture output version since PTX only compilation *must* be the same
std::regex real_capture("^.*arch=.*,code=(sm_[0-9]+a?)$");
std::regex virtual_capture("^.*arch=.*,code=(compute_[0-9]+a?)$");

// Arch list is a set of unique pairs of strings and bools
// e.x. { compute_arch, real_or_virtual }
// { "sm_80", true } { "compute_80", false }
using ArchList = std::set<ArchConfig>;
ArchList buildList;

// Input example: arch=compute_80,code=sm_80
static ArchConfig translate_gpu_arch(const std::string& arch)
{
  std::smatch real;
  std::smatch virt;
  std::regex_match(arch, real, real_capture);
  std::regex_match(arch, virt, virtual_capture);

  // Safe default of "compute_60" in case parsing fails
  ArchConfig config = (real.size()) ? ArchConfig{real[1].str(), true}
                    : (virt.size()) ? ArchConfig{virt[1].str(), false}
                                    : ArchConfig{"compute_60", false};

  return config;
}

// Greedy handlers inform the argument processor to expect more arguments
constexpr auto make_greedy_handler = [](char const* match) {
  return ArgPair{std::regex(match), [](const std::smatch&) {
                   return GREEDY;
                 }};
};

ArgPair argHandlers[] = {
  {// Forward all arguments to NVCC
   std::regex("^-c$"),
   [](const std::smatch&) {
     building = true;
     // We're compiling, maybe do something useful
     return NORMAL; // Unreachable
   }},
  {// Forward all arguments to NVCC
   std::regex("^-E$"),
   [](const std::smatch&) {
     platform_exec("nvcc", g_argv, g_argc);
     return ABORT; // Unreachable
   }},
  {// Greed input file type flag
   make_greedy_handler("^-x$")},
  {// Matches for CUDA input type
   std::regex("^-x ?cu$"),
   [](const std::smatch& match) {
     ignoredArguments.emplace_back(match[0].str());
     return NORMAL;
   }},
  {// Matches anything other than CUDA as the CUDA flag is captured before this one
   std::regex("^-x ?(.*)$"),
   [](const std::smatch&) {
     // If we're building with something else just add the default arch
     buildList.emplace(translate_gpu_arch(""));
     return NORMAL;
   }},
  {// The include flag is improperly formatted, greed append
   make_greedy_handler("^-I$")},
  {std::regex("^-I ?(.+)$"),
   [](const std::smatch& match) {
     nvrtcArguments.emplace_back(match[0].str());
     return NORMAL;
   }},
  {make_greedy_handler("^(-include|-isystem)$")},
  {// Matches any force include or system include directories
   // Might need to figure out if we need to force include a file manually
   std::regex("^-include ?(.+)$"),
   [](const std::smatch& match) {
     nvrtcArguments.emplace_back("--pre-include=" + match[1].str());
     return NORMAL;
   }},
  {make_greedy_handler("^-o$")},
  {// Matches '-o nul' which is used for syntax only testing (i.e. .fail.cpp tests)
   std::regex("^-o (?:.*?dev)?.*nul$"),
   [](const std::smatch&) {
     skipOutput = true;
     return NORMAL;
   }},
  {// Matches '-o object' and obtains the output directory
   // \\\\ skip C++ escape, and skip regex escape to match \ on Windows
   // The second match grouping catches the name sorta of the file. i.e. test.pass.cpp -> test.pass
   std::regex("^-o (.+)[\\\\/]([^\\\\/]+)\\..+$"),
   [](const std::smatch& match) {
     outputDir  = match[1].str();
     outputFile = match[2].str();
     return NORMAL;
   }},
  {make_greedy_handler("^-gencode$")},
  {// Matches '-gencode=' or '-gencode ...'
   std::regex("^-gencode[= ]?(.+)$"),
   [](const std::smatch& match) {
     buildList.emplace(translate_gpu_arch(match[1].str().data()));
     return NORMAL;
   }},
  {// Matches the many various versions of dialect switch and normalizes it
   std::regex("^[-/]std[:=](.+)$"),
   [](const std::smatch& match) {
     nvrtcArguments.emplace_back("-std=" + match[1].str());
     return NORMAL;
   }},
  {// Capture an argument that is just '-'. If no input file is listed input is on stdin
   std::regex("^-$"),
   [](const std::smatch& match) {
     inputFile = match[0].str();
     return NORMAL;
   }},
  {// If an input lists a .gpu file, run that file instead
   std::regex("^([^-].*).gpu$"),
   [](const std::smatch& match) {
     execute         = true;
     executionConfig = ExecutionConfig{RunConfig{1, 0}, {match[0].str()}};

     return NORMAL;
   }},
  {// If an input is a .exe file, search for other builds and run those
   std::regex("^([^-].*).exe$"),
   [](const std::smatch& match) {
     execute         = true;
     executionConfig = load_execution_config_from_file(match[1].str() + ".build.yml");
     assert(executionConfig.builds.size());
     return NORMAL;
   }},
  {// Capture any argument not starting with '-' as the input file
   std::regex("^([^-].+)[\\\\/].+$"),
   [](const std::smatch& match) {
     inputFile = match[0].str();
     // Capture directory of input file as an include path
     nvrtcArguments.emplace_back("-I " + match[1].str());
     return NORMAL;
   }},
  {// Throw away remaining arguments
   std::regex("^-.+$"),
   [](const std::smatch& match) {
     ignoredArguments.emplace_back(match[0].str());
     return NORMAL;
   }},
};

int main(int argc, char** argv)
{
  // Greedily take off first arg
  g_argc = argc - 1;
  g_argv = argv + 1;

  ArgProcessorState argState = NORMAL;

  // Start by parsing arguments and building the configuration
  std::string c_arg{};
  for (auto a = g_argv; a < g_argv + g_argc; a++)
  {
    // If the argument was greedy, we'll retry with an appended argument
    c_arg = (argState == GREEDY) ? c_arg + " " + *a : *a;

    for (auto& h : argHandlers)
    {
      auto& regex   = h.first;
      auto& handler = h.second;

      std::smatch matches;
      std::regex_match(c_arg, matches, regex);

      if (matches.size())
      {
        argState = handler(matches);
        break;
      }
    }
  }

  fprintf(stderr, "NVRTCC Configuration:\r\n");
  fprintf(stderr, "  Output dir: %s\r\n", outputDir.c_str());
  fprintf(stderr, "  Output file: %s\r\n", outputFile.c_str());
  fprintf(stderr, "  Input file: %s\r\n", inputFile.c_str());
  fprintf(stderr, "  Building: %s\r\n", building ? "true" : "false");
  fprintf(stderr, "  Skipping output: %s\r\n", skipOutput ? "true" : "false");
  fprintf(stderr, "  Executing: %s\r\n", execute ? "true" : "false");

  // Load the input file and execute
  if (execute)
  {
    fprintf(
      stderr, "Executing %s with %i threads\r\n", executionConfig.builds[0].c_str(), executionConfig.rc.threadCount);
    load_and_run_gpu_code(executionConfig.builds[0], executionConfig.rc);
    fprintf(stderr, "Execution Passed\r\n");
    return 0;
  }

  // Linking exits and does nothing
  if (!building)
  {
    return 0;
  }

  // Rebuild the output file template based on the filename
  // Check for nul - do not write files
  std::string outputTemplate;

  if (outputDir.size() && outputFile.size())
  {
    outputTemplate = outputDir + "/" + outputFile;
  }
  else
  {
    outputTemplate = "temp";
  }

  // load input test file and prepend fakemain
  std::string testCu = program + load_input_file(inputFile);

  // Write any needed kernel launch data to file for later
  RunConfig runConfig = parse_run_config(testCu);

  nvrtcArguments.emplace_back("-DCCCL_ENABLE_ASSERTIONS");
  if (!skipOutput)
  {
    std::ofstream ostr(outputTemplate + ".build.yml");
    ostr << "cuda_thread_count: " << runConfig.threadCount << '\n';
    ostr << "cuda_block_shmem_size: " << runConfig.shmemSize << '\n';

    // Do a build for each arch and add it to the build list
    ostr << "builds:\n";
    for (const auto& build : buildList)
    {
      auto gpuCode            = nvrtc_build_prog(testCu, build, nvrtcArguments);
      std::string gpuCodeFile = outputTemplate + "." + archString(build) + ".gpu";
      write_output_file(gpuCode.data(), gpuCode.size(), gpuCodeFile);
      ostr << "  - ";
      ostr << '\'' << gpuCodeFile << '\'';
      ostr << '\n';
    }
    ostr.close();
  }
  else
  {
    for (const auto& build : buildList)
    {
      auto gpuCode = nvrtc_build_prog(testCu, build, nvrtcArguments);
    }
  }

  return 0;
}
