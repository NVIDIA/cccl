# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile resource/epoch counterexamples in a generated --carrier fixture."""

import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--case", action="append", default=[])
args = parser.parse_args()
output = args.output.resolve()
record = json.loads((output / "command.json").read_text())
root = output / "fixture/carrier/src/main.rs"
original = root.read_text()
api = output / "fixture/api/src/carrier.rs"
original_api = api.read_text()
command = record["argv"].copy()
command[command.index("run")] = "build"
env = os.environ.copy()
env.update(
    CUDA_OXIDE_BACKEND=record["backend"],
    CUDA_HOME=record["cuda_root"],
    CUDA_PATH=record["cuda_root"],
)
prefix = """// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
use coop_marker_fixture_api::carrier;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
type Buffer<'a> = DisjointSlice<'a, i32, CollectiveTiles<4>>;
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
pub fn probe<'a>(input: &'a [i32], mut output: Buffer<'a>, mut other: Buffer<'a>, valid: usize) {
    let offset = thread::blockIdx_x() as usize * 512;
    let mut values = [0_i32; 4];
"""
suffix = "\n}\nfn main() {}\n"
cases = {
    "missing_barrier": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "carrier::reload(&mut output, offset, valid, &mut values);",
        "requires a full-block barrier",
    ),
    "loop_backedge": (
        "let mut remaining = 2;\nwhile remaining > 0 {\n"
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "remaining -= 1;\n}",
        "requires a full-block barrier",
    ),
    "conditional_barrier": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "if valid == 0 { thread::sync_threads(); }\n"
        "carrier::reload(&mut output, offset, valid, &mut values);",
        "requires a full-block barrier",
    ),
    "conditional_access": (
        "if valid > 0 { carrier::store(&mut output, offset, valid, &mut values); }\n"
        "carrier::reload(&mut output, offset, valid, &mut values);",
        "requires a full-block barrier",
    ),
    "divergent_barrier": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "if thread::threadIdx_x() & 1 == 0 { thread::sync_threads(); }",
        "barrier is reached through lane-varying",
    ),
    "varying_count": (
        "carrier::store(&mut output, offset, thread::threadIdx_x() as usize, &mut values);",
        "not proved block-uniform",
    ),
    "varying_carrier": (
        "let selected = if thread::threadIdx_x() & 1 == 0 { &mut output } else { &mut other };\n"
        "carrier::store(selected, offset, valid, &mut values);",
        "not proved block-uniform",
    ),
    "ambiguous_origin": (
        "let selected = if valid == 0 { &mut output } else { &mut other };\n"
        "carrier::store(selected, offset, valid, &mut values);",
        "not proved block-uniform",
    ),
    "descriptor_mutation": (
        "replace(&mut output, other);\n"
        "carrier::store(&mut output, offset, valid, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "readonly_descriptor_mutation": (
        "let mut selected = input;\n"
        "redirect_read(&mut selected, input);\n"
        "carrier::load(selected, offset, valid, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "enum_descriptor_mutation": (
        "let mut selected = Some(output);\n"
        "replace_optional(&mut selected, other);\n"
        "if let Some(mut target) = selected {\n"
        "    carrier::store(&mut target, offset, valid, &mut values);\n"
        "}",
        "mutation through a reference to a resource descriptor",
    ),
    "descriptor_swap": (
        "carrier::store(&mut other, offset, valid, &mut values);\n"
        "core::mem::swap(&mut output, &mut other);\n"
        "carrier::store(&mut output, offset, valid, &mut values);",
        "intrinsic is outside the supported checked block subset",
    ),
    "descriptor_copy": (
        "unsafe { core::ptr::copy_nonoverlapping(&other, &mut output, 1); }\n"
        "carrier::store(&mut output, offset, valid, &mut values);",
        "bulk memory mutation",
    ),
    "aggregate_reference_mutation": (
        "carrier::store(&mut other, offset, valid, &mut values);\n"
        "let mut refs = (&mut output,);\n"
        "redirect(&mut refs, &mut other);\n"
        "carrier::store(refs.0, offset, valid, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "projected_reborrow": (
        "carrier::store(&mut other, offset, valid, &mut values);\n"
        "let mut refs = (&mut output,);\n"
        "let target = &mut refs.0;\n*target = &mut other;\n"
        "carrier::store(refs.0, offset, valid, &mut values);",
        "requires a full-block barrier",
    ),
    "joined_owner_replacement": (
        "let mut first = Owner { buf: output };\n"
        "let mut second = Owner { buf: other };\n"
        "let mut replacement = Owner { buf: third };\n"
        "let mut one = [7_i32; 1];\n"
        "carrier::store(&mut replacement.buf, offset, 4, &mut one);\n"
        "let selected = if choose { &mut first } else { &mut second };\n"
        "*selected = replacement;\n"
        "carrier::store(&mut first.buf, offset, 4, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "joined_owner_helper": (
        "let mut first = Owner { buf: output };\n"
        "let mut second = Owner { buf: other };\n"
        "let mut replacement = Owner { buf: third };\n"
        "let mut one = [7_i32; 1];\n"
        "carrier::store(&mut replacement.buf, offset, 4, &mut one);\n"
        "let selected = if choose { &mut first } else { &mut second };\n"
        "replace_owner(selected, replacement);\n"
        "carrier::store(&mut first.buf, offset, 4, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "tuple_owner_replacement": (
        "let mut first = Owner { buf: output };\n"
        "let mut replacement = Owner { buf: other };\n"
        "let mut one = [7_i32; 1];\n"
        "carrier::store(&mut replacement.buf, offset, 4, &mut one);\n"
        "let references = (&mut first,);\n"
        "*references.0 = replacement;\n"
        "carrier::store(&mut first.buf, offset, 4, &mut values);",
        "mutation through a reference to a resource descriptor",
    ),
    "warp_barrier": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "cuda_device::warp::sync_mask(u32::MAX);\n"
        "carrier::reload(&mut output, offset, valid, &mut values);",
        "intrinsic is outside the supported checked block subset",
    ),
    "changed_items_without_barrier": (
        "let mut one = [0_i32; 1];\n"
        "carrier::store(&mut output, offset, 128, &mut one);\n"
        "carrier::store(&mut output, offset, 512, &mut values);",
        "requires a full-block barrier",
    ),
    "slice_write_witness": (
        "carrier::load(input, offset, valid, &mut values);",
        "resource witness requires DisjointSlice<T, CollectiveTiles<C>>",
    ),
    "immutable_write_witness": (
        "carrier::store(&mut output, offset, valid, &mut values);",
        "resource writes require a mutable carrier witness",
    ),
    "wrong_element_witness": (
        "carrier::store(&mut output, offset, valid, &mut values);",
        "resource witness element type disagrees with the library element type",
    ),
    "zero_capacity": (
        "carrier::store(&mut output, offset, valid, &mut values);",
        "resource carriers require positive capacity",
    ),
    "mixed_marker": (
        "unsafe { coop_marker_fixture_api::store::<4, false>(output.as_mut_ptr(), values.as_mut_ptr(), valid as i32, true); }\n"
        "carrier::reload(&mut output, offset, valid, &mut values);",
        "untracked and resource-checked",
    ),
    "scratch_provider": (
        "carrier::store(&mut output, offset, valid, &mut values);",
        "require a scratch-free provider",
    ),
    "lane_reference": (
        "let _ = output.get_mut(thread::index_1d());",
        "FlatIndexSpace",
    ),
    "distinct_allocations": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "carrier::store(&mut other, offset, valid, &mut values);",
        None,
    ),
    "aggregate_forward": (
        "let owned = Owner { buf: output };\n"
        "let mut moved = owned;\n"
        "carrier::store(&mut moved.buf, offset, valid, &mut values);\n"
        "thread::sync_threads();\n"
        "carrier::reload(&mut moved.buf, offset, valid, &mut values);",
        None,
    ),
    "payload_reference_write": (
        "let references = (&mut values,);\n"
        "references.0[0] = 37;\n"
        "carrier::store(&mut output, offset, valid, &mut values);",
        None,
    ),
    "same_origin_join": (
        "let selected = if valid == 0 { &mut output } else { &mut output };\n"
        "carrier::store(selected, offset, valid, &mut values);",
        None,
    ),
    "helper_barrier": (
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "complete();\ncarrier::reload(&mut output, offset, valid, &mut values);",
        None,
    ),
    "loop_barrier": (
        "let mut remaining = 2;\nwhile remaining > 0 {\n"
        "carrier::store(&mut output, offset, valid, &mut values);\n"
        "thread::sync_threads();\nremaining -= 1;\n}",
        None,
    ),
}
helpers = """
struct Owner<'a> { buf: Buffer<'a> }
#[inline(never)]
fn replace_owner<'a>(target: &mut Owner<'a>, value: Owner<'a>) { *target = value; }
#[inline(never)]
fn complete() { thread::sync_threads(); }
#[inline(never)]
fn replace<'a>(target: &mut Buffer<'a>, value: Buffer<'a>) { *target = value; }
#[inline(never)]
fn replace_optional<'a>(target: &mut Option<Buffer<'a>>, value: Buffer<'a>) { *target = Some(value); }
#[inline(never)]
fn redirect<'a, 'b>(target: &mut (&'a mut Buffer<'b>,), value: &'a mut Buffer<'b>) {
    target.0 = value;
}
#[inline(never)]
fn redirect_read<'a>(target: &mut &'a [i32], value: &'a [i32]) { *target = value; }
"""
results = {}
unknown = set(args.case) - cases.keys()
if unknown:
    parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
try:
    for name, (body, diagnostic) in cases.items():
        if args.case and name not in args.case:
            continue
        case_prefix = prefix
        if name.startswith("joined_owner"):
            case_prefix = prefix.replace(
                "valid: usize)", "valid: usize, third: Buffer<'a>, choose: bool)"
            )
        if name == "zero_capacity":
            case_prefix = prefix.replace("CollectiveTiles<4>", "CollectiveTiles<0>")
        root.write_text(case_prefix + body + suffix + helpers)
        case_api = original_api
        if name == "scratch_provider":
            case_api = original_api.replace(
                "super::direct_store_partial", "super::transpose_store_partial"
            )
        elif name == "slice_write_witness":
            case_api = original_api.replace("ITEMS, false>", "ITEMS, true>")
        elif name == "immutable_write_witness":
            case_api = original_api.replace(
                "            target,", "            &*target,"
            )
        elif name == "wrong_element_witness":
            case_api = (
                original_api.replace(
                    "::<i32, _, _, _, _, ITEMS, true>",
                    "::<u32, _, _, _, _, ITEMS, true>",
                )
                .replace(
                    "(pointer.add(offset), valid as i32)",
                    "(pointer.add(offset).cast::<u32>(), valid as i32)",
                )
                .replace(
                    "(values.as_mut_ptr(),)", "(values.as_mut_ptr().cast::<u32>(),)"
                )
            )
        elif name == "zero_capacity":
            # Keep the marker reachable: this case tests structural witness
            # validation, independently of the wrapper's runtime containment.
            case_api = original_api.replace(
                "    containment::<CAPACITY>(offset, valid);\n", ""
            )
        api.write_text(case_api)
        log_path = output / f"carrier-{name}.log"
        # Each program gets its own artifacts, including wrapper dependencies.
        case_env = env.copy()
        case_env["CARGO_TARGET_DIR"] = str(output / "carrier-targets" / name)
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=record["cwd"],
                env=case_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log = log_path.read_text()
        passed = (
            result.returncode == 0
            if diagnostic is None
            else result.returncode != 0 and diagnostic in log
        )
        results[name] = {"passed": passed, "returncode": result.returncode}
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
finally:
    root.write_text(original)
    api.write_text(original_api)
    (output / "carrier-checks.json").write_text(json.dumps(results, indent=2) + "\n")
raise SystemExit(0 if all(item["passed"] for item in results.values()) else 1)
