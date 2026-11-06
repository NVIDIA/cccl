#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Opt-in diagnostics for the PowerShell/Bash Thrust comparison. Files travel with
# the existing job-result artifact, whose staging directory is mounted on the host.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
output="${ARTIFACT_UPLOAD_STAGE:?}/jobs/${JOB_ID:?}/experiment"
mkdir -p "$output"

case "${1:?Expected preflight or collect}" in
  preflight)
    [[ "$OSTYPE" == msys* || "$OSTYPE" == mingw* ]]
    actual_sha=$(git rev-parse HEAD)
    [[ "$actual_sha" == "${GITHUB_SHA:?}" ]]
    [[ "${CCCL_WINDOWS_BASH_IMAGE:?}" == *@sha256:* ]]
    for tool in bash cygpath cmake ctest ninja cl nvcc sccache jq tar zstd; do
      command -v "$tool"
    done
    native_cl=$(cygpath -am "$(command -v cl)")
    native_nvcc=$(cygpath -am "$(command -v nvcc)")
    jq -n --arg sha "$actual_sha" --arg image "$CCCL_WINDOWS_BASH_IMAGE" \
      --arg command "${COMMAND:?}" --arg cl "$native_cl" --arg nvcc "$native_nvcc" \
      --arg processors "${NUMBER_OF_PROCESSORS:?}" --arg bash "$BASH_VERSION" \
      '{sha: $sha, image: $image, command: $command, cl: $cl, nvcc: $nvcc,
        processors: $processors, bash: $bash}' > "$output/environment.json"
    {
      uname -a
      cmake --version
      ninja --version
      nvcc --version
      cl 2>&1 || :
      sccache --version
      tar --version
      zstd --version
    } > "$output/versions.txt"

    # Exercise native tools through Git Bash with spaces in source/output paths.
    # Compile CUDA but do not run device code on the CPU runner.
    probe="build/windows bash preflight/${JOB_ID}"
    mkdir -p "$probe"
    native_probe=$(cygpath -am "$probe")
    printf 'int main() { return 0; }\n' > "$probe/probe.cpp"
    printf '__global__ void probe_kernel() {}\n' > "$probe/probe.cu"
    (
      cd "$probe"
      # Every compiler path below is already native. In particular, Git Bash's
      # automatic conversion corrupts MSVC's attached -FeC:/... argument.
      export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
      cl -nologo -EHsc "-Fe${native_probe}/probe.exe" "${native_probe}/probe.cpp"
      ./probe.exe
      nvcc -ccbin "$native_cl" -std=c++20 -arch=sm_89 -c \
        "${native_probe}/probe.cu" -o "${native_probe}/probe.obj"
    ) 2>&1 | tee "$output/native-probe.log"
    ;;
  collect)
    status="${2:?Expected command exit code}"
    printf '%s\n' "$status" > "$output/command-exit-code.txt"
    if [[ -f build/thrust/CMakeCache.txt ]]; then
      cp build/thrust/CMakeCache.txt "$output/"
      ninja -C build/thrust -t targets all > "$output/targets.txt"
      # CTest's inventory command overwrites LastTest.log. Preserve execution
      # evidence first, especially for the PowerShell arm without ctest.log.
      if [[ -d build/thrust/Testing ]]; then
        cp -R build/thrust/Testing "$output/"
      fi
      ctest --preset thrust-cpu --show-only=json-v1 > "$output/tests.json"
      jq -e '.tests | length > 0' "$output/tests.json"
      find build/thrust/bin -type f -name '*.exe' | sort > "$output/executables.txt"
      if [[ -f build/thrust/compile_commands.json ]]; then
        cp build/thrust/compile_commands.json "$output/"
      fi
      if [[ -f build/thrust-cpu/ctest.log ]]; then
        cp build/thrust-cpu/ctest.log "$output/"
      fi
      sccache --show-adv-stats --stats-format=json > "$output/sccache.json" || :
    elif [[ "$status" == 0 ]]; then
      echo 'Successful experiment command did not produce a Thrust build tree.' >&2
      exit 1
    fi
    ;;
  *) echo "Unknown experiment phase: $1" >&2; exit 1 ;;
esac
