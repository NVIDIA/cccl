# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measure source-built CUB and cuda.coop.cutlass kernels with NVBench."""

import argparse
import fnmatch
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from _cases import all_cases, assert_result, make_arrays


def checked(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_case(state, case, args, artifacts):
    """Compile and check one state before entering NVBench's timed region."""
    from _kernels import capture, prepare

    from cuda.bindings import driver

    blocks = state.get_int64("Blocks")
    launches = state.get_int64("Launches")
    if not 1 <= blocks <= 2**20 or launches not in (1, 16):
        raise ValueError(
            "Blocks must be in [1, 2^20]; Launches must be 1 or 16"
        )
    device = checked(driver.cuCtxGetDevice())
    if state.get_device() is None or int(device) != int(state.get_device()):
        raise RuntimeError("Current CUDA context differs from NVBench's device")
    stream = driver.CUstream(state.get_stream().addressof())
    inputs, expected = make_arrays(case, blocks)
    allocations = []
    graphs = []
    kernel = None
    guard_bytes = 256
    case_dir = artifacts / f"d{int(device)}-{case.name}-b{blocks}-l{launches}"
    case_dir.mkdir(parents=True, exist_ok=False)

    def allocate(values, padding=0):
        pointer = checked(driver.cuMemAlloc(values.nbytes + padding))
        allocations.append(pointer)
        return pointer

    try:
        sources = {key: allocate(value) for key, value in inputs.items()}
        outputs = {
            key: allocate(value, guard_bytes) for key, value in expected.items()
        }
        for key, value in inputs.items():
            checked(
                driver.cuMemcpyHtoD(
                    sources[key], value.ctypes.data, value.nbytes
                )
            )
        if getattr(args, "compiled_kernels", None) is None:
            kernel = prepare(
                case, sources, outputs, stream, blocks, args.provider, case_dir
            )
        else:
            from _replay import load

            kernel = load(
                case,
                sources,
                outputs,
                stream,
                blocks,
                args.provider,
                case_dir,
                args.compiled_kernels,
            )

        def poison():
            for key, value in expected.items():
                checked(
                    driver.cuMemsetD8Async(
                        outputs[key], 0xCD, value.nbytes, stream
                    )
                )
                checked(
                    driver.cuMemsetD8Async(
                        int(outputs[key]) + value.nbytes,
                        0xA5,
                        guard_bytes,
                        stream,
                    )
                )

        def verify():
            checked(driver.cuStreamSynchronize(stream))
            observed = {}
            for key, value in expected.items():
                result = np.empty_like(value)
                checked(
                    driver.cuMemcpyDtoH(
                        result.ctypes.data, outputs[key], result.nbytes
                    )
                )
                observed[key] = result
                guard = np.empty(guard_bytes, dtype=np.uint8)
                checked(
                    driver.cuMemcpyDtoH(
                        guard.ctypes.data,
                        int(outputs[key]) + value.nbytes,
                        guard.nbytes,
                    )
                )
                np.testing.assert_array_equal(guard, 0xA5)
            assert_result(case, observed, expected)
            for key, value in inputs.items():
                observed_input = np.empty_like(value)
                checked(
                    driver.cuMemcpyDtoH(
                        observed_input.ctypes.data, sources[key], value.nbytes
                    )
                )
                np.testing.assert_array_equal(observed_input, value)

        source_launch = getattr(kernel, "compiled_launch", None)
        if source_launch is not None:
            poison()
            source_launch()
            verify()
        poison()
        kernel.launch()
        verify()
        execute = kernel.launch
        if launches > 1:
            graph = capture(stream, kernel.launch, launches)
            graphs.append([graph, None])
            _, count = checked(driver.cuGraphGetNodes(graph, 0))
            if count != launches:
                raise RuntimeError(
                    f"Expected {launches} graph nodes, got {count}"
                )
            executable = checked(driver.cuGraphInstantiate(graph, 0))
            graphs[-1][1] = executable

            def execute():
                checked(driver.cuGraphLaunch(executable, stream))

            poison()
            execute()
            verify()

        state.add_element_count(inputs["input"].size * launches)
        state.add_summary("provider", args.provider)
        state.add_summary("timing", args.timing)
        state.add_summary("time_unit", "seconds per Launches kernels")
        if not args.validate_only:

            def launcher(execution):
                if execution.get_stream().addressof() != int(stream):
                    raise RuntimeError("NVBench changed the launch stream")
                execute()

            poison()
            state.exec(launcher, batched=args.timing == "batch")
            if state.is_skipped():
                raise RuntimeError(state.get_skip_reason())
            verify()
        result = {
            "case": case.name,
            "blocks": blocks,
            "launches": launches,
            "device": int(device),
            "provider": args.provider,
            "correct": True,
            "source_launch_checked": source_launch is not None,
            "kernel": kernel.metadata,
        }
        if args.export_inputs:
            result["arrays"] = {}
            for role, arrays in (("inputs", inputs), ("expected", expected)):
                result["arrays"][role] = {}
                for key, value in arrays.items():
                    path = case_dir / f"{key}.bin"
                    value.tofile(path)
                    result["arrays"][role][key] = {
                        "path": str(path),
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "bytes": value.nbytes,
                    }
        return result
    finally:
        for graph, executable in reversed(graphs):
            if executable is not None:
                checked(driver.cuGraphExecDestroy(executable))
            checked(driver.cuGraphDestroy(graph))
        if kernel is not None:
            kernel.close()
        for pointer in reversed(allocations):
            checked(driver.cuMemFree(pointer))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--provider", choices=("cub", "cutlass"), default="cutlass"
    )
    parser.add_argument("--case", action="append", help="Case glob; repeatable")
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 1024])
    parser.add_argument("--launches", type=int, nargs="+", default=[1])
    parser.add_argument("--timing", choices=("cold", "batch"), default="cold")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument(
        "--compiled-kernels",
        type=Path,
        help="Replay qualified cubins from a portable artifact directory",
    )
    parser.add_argument("--export-inputs", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--runner-help", action="help")
    args, nvbench_args = parser.parse_known_args()
    cases = all_cases()
    if args.case:
        cases = [
            case
            for case in cases
            if any(
                fnmatch.fnmatchcase(case.name, pattern) for pattern in args.case
            )
        ]
    if not cases:
        parser.error("No cases matched")
    if args.list_cases:
        print("\n".join(case.name for case in cases))
        return
    if args.export_inputs and args.artifacts is None:
        parser.error("--export-inputs requires --artifacts")
    from cuda import bench

    with tempfile.TemporaryDirectory(prefix="coop-nvbench-") as temporary:
        artifacts = args.artifacts or Path(temporary)
        if args.artifacts:
            artifacts.mkdir(parents=True, exist_ok=False)
        results, errors = [], []
        by_name = {case.name: case for case in cases}

        def benchmark(state):
            try:
                results.append(
                    run_case(
                        state,
                        by_name[state.get_string("Case")],
                        args,
                        artifacts,
                    )
                )
            except Exception as error:
                errors.append(f"{state.get_axis_values_as_string()}: {error}")
                raise

        registration = bench.register(benchmark)
        registration.set_name("coop")
        registration.add_string_axis("Case", list(by_name))
        registration.add_int64_axis("Blocks", args.blocks)
        registration.add_int64_axis("Launches", args.launches)
        # The CuTe DSL parses sys.argv during compilation, and argparse would
        # read NVBench's "-d" as an abbreviation of the DSL's "-diagnostic".
        argv = sys.argv
        sys.argv = argv[:1]
        try:
            bench.run_all_benchmarks([argv[0], *nvbench_args])
        finally:
            sys.argv = argv
        if args.artifacts:

            def version(name):
                try:
                    return importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    return None

            receipt = {
                "argv": sys.argv,
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_HOME",
                        "CUDA_PATH",
                        "CUTE_DSL_USE_EXTENSION_COMPILER",
                    )
                },
                "versions": {
                    name: version(name)
                    for name in (
                        "cuda-bench",
                        "cuda-core",
                        "cuda-bindings",
                        "nvidia-cutlass-dsl",
                        "numpy",
                    )
                },
                "results": results,
                "errors": errors,
            }
            (artifacts / "qualification.json").write_text(
                json.dumps(receipt, indent=2, default=str) + "\n"
            )
        if errors:
            raise RuntimeError("Benchmark states failed:\n" + "\n".join(errors))
        listing = any(
            flag in nvbench_args
            for flag in ("--help", "-h", "--list", "-l", "--list-benches")
        )
        if not results and not listing:
            raise RuntimeError("NVBench executed no benchmark states")


if __name__ == "__main__":
    main()
