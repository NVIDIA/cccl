# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile private Direct validity controls and resource-epoch contrasts."""

import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
output = args.output.resolve()
record = json.loads((output / "command.json").read_text())
source = output / "fixture/carrier/src/validity.rs"
original = source.read_text()
command = record["argv"].copy()
command[command.index("run")] = "build"
# build forwards binary selection to Cargo; run has its own --bin option.
command.insert(command.index("--bin"), "--")
env = os.environ.copy()
env.update(
    CCCL_SOURCE=record.get("cccl_source", str(Path(__file__).resolve().parents[3])),
    CUDA_OXIDE_BACKEND=record["backend"],
    CUDA_HOME=record["cuda_root"],
    CUDA_PATH=record["cuda_root"],
)
prefix = """// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
use coop_marker_fixture_api::{ThreadData, this_block, validity};
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
pub fn probe(input: &[i32], mut output: DisjointSlice<i32, CollectiveTiles<4>>, valid: usize) {
    let group = this_block();
    let offset = thread::blockIdx_x() as usize * 512;
    let mut values = ThreadData::<_, 4>::new();
"""
suffix = "\n}\nfn main() {}\n"
cases = {
    "varying_count": (
        "validity::load_masked(group, input, offset, thread::threadIdx_x() as usize, &mut values);",
        "not proved block-uniform",
    ),
    "varying_default": (
        "validity::load_filled(group, input, offset, valid, thread::threadIdx_x() as i32, &mut values);",
        "not proved block-uniform",
    ),
    "varying_offset": (
        "validity::load_masked(group, input, offset + thread::threadIdx_x() as usize, valid, &mut values);",
        "not proved block-uniform",
    ),
    "missing_barrier": (
        "validity::load_filled(group, input, offset, valid, 0, &mut values);\n"
        "validity::store_partial(group, &mut output, offset, valid, &mut values);\n"
        "validity::reload_filled(group, &mut output, offset, valid, 0, &mut values);",
        "requires a barrier covering both accesses",
    ),
    "varying_participation": (
        "if thread::threadIdx_x() & 1 == 0 {\n"
        "validity::load_masked(group, input, offset, valid, &mut values);\n}",
        "lane-varying",
    ),
    "loop_without_completion": (
        "validity::load_filled(group, input, offset, valid, 0, &mut values);\n"
        "let mut count = 2; while count > 0 {\n"
        "validity::store_partial(group, &mut output, offset, valid, &mut values);\n"
        "count -= 1; }",
        "requires a barrier covering both accesses",
    ),
    "explicit_completion": (
        "validity::load_filled(group, input, offset, valid, 0, &mut values);\n"
        "validity::store_partial(group, &mut output, offset, valid, &mut values);\n"
        "thread::sync_threads();\n"
        "validity::reload_filled(group, &mut output, offset, valid, 0, &mut values);",
        None,
    ),
    "reconverged_payload": (
        "let mut other = ThreadData::<_, 4>::new();\n"
        "values.write(0, 7); other.write(0, 9);\n"
        "let mut selected = if thread::threadIdx_x() & 1 == 0 { values } else { other };\n"
        "validity::store_partial(group, &mut output, offset, 1, &mut selected);",
        None,
    ),
}
results = []
try:
    for name, (body, diagnostic) in cases.items():
        source.write_text(prefix + body + suffix)
        result = subprocess.run(
            command,
            cwd=record["cwd"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output / f"validity-{name}.log").write_text(result.stdout)
        passed = (
            result.returncode == 0
            if diagnostic is None
            else result.returncode != 0 and diagnostic in result.stdout
        )
        results.append(
            {
                "case": name,
                "expected": diagnostic,
                "returncode": result.returncode,
                "passed": passed,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        (output / "validity-rejections.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
finally:
    source.write_text(original)
raise SystemExit(0 if all(row["passed"] for row in results) else 1)
