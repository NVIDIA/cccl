# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check rejected source programs in a copy made by run_oxide_fixture.py --checked.

All edits are confined to that generated fixture and restored after each case.
Negative cases fail in the checked collective diagnostic, before launch. A
separate-site geometry case must compile with two shapes and one marker Instance.
"""

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
fixture = output / "fixture"
paths = {
    "helper": fixture / "helper/src/checked.rs",
    "api": fixture / "api/src/lib.rs",
    "root": fixture / "consumer/src/main.rs",
}
original = {key: path.read_text() for key, path in paths.items()}
command = record["argv"].copy()
assert any(feature in command for feature in ["checked", "root-context"]), (
    "first run the fixture with --checked or --root-context"
)
command[command.index("run")] = "build"
env = os.environ.copy()
env.update(
    CCCL_SOURCE=str(Path(__file__).resolve().parents[3]),
    CUDA_OXIDE_BACKEND=record["backend"],
)
if "cuda_root" in record:
    env.update(CUDA_HOME=record["cuda_root"], CUDA_PATH=record["cuda_root"])
env.pop("CUDA_OXIDE_UNCHECKED_INDEXING", None)
anchor = "    let mut first = [0_i32; ITEMS_PER_THREAD];"


def in_helper(code):
    return {"helper": original["helper"].replace(anchor, code + "\n" + anchor)}


shape_root = original["root"].replace("block = (128, 1, 1)", "block = (64, 1, 1)", 1)
for before, after in [
    (
        "roundtrip::<1, false>(input, output, post, valid, mode)",
        "shape_probe::<0>(input)",
    ),
    (
        "roundtrip::<4, false>(input, output, post, valid, mode)",
        "shape_probe::<1>(input)",
    ),
]:
    shape_root = shape_root.replace(before, after)
shape_api = (
    original["api"]
    + """
#[inline(never)]
pub unsafe fn shape_probe<const TAG: usize>(input: *const i32, values: *mut i32) {
    unsafe { cuda_device::library::checked_block_call_v1::<i32, _, _, _, 1>(direct_load_full, (input,), (values,)); }
}
"""
)
shape_helper = (
    original["helper"]
    + """
#[inline(never)]
pub unsafe fn shape_probe<const TAG: usize>(input: *const i32) {
    let mut value = 0;
    unsafe { coop_marker_fixture_api::shape_probe::<TAG>(input, &mut value); }
}
"""
)
# Expose the additional helper only in the generated fixture copy.
helper_lib = fixture / "helper/src/lib.rs"
helper_lib_original = helper_lib.read_text()
cases = {
    "distinct_shapes": (
        {"root": shape_root, "api": shape_api, "helper": shape_helper},
        None,
    ),
    "count": (
        in_helper("    let valid = valid + (lane & 1) as i32;"),
        "not proved block-uniform",
    ),
    "carrier": (
        in_helper(
            "    let input = if lane & 1 == 0 { input } else { output as *const i32 };"
        ),
        "not proved block-uniform",
    ),
    "offset": (
        in_helper("    let input = unsafe { input.add(lane) };"),
        "not proved block-uniform",
    ),
    "maybe_uninit_value": (
        {
            "helper": in_helper(
                "    let initialized = initialized_value((valid, 0_u8));"
            )["helper"].replace(
                "payload[i] += CONTEXT_INCREMENT;",
                "payload[i] += CONTEXT_INCREMENT + initialized.0;",
            )
            + """
#[inline(never)]
fn initialized_value<T: Copy>(value: T) -> T {
    unsafe { core::mem::MaybeUninit::new(value).assume_init() }
}
"""
        },
        None,
    ),
    "maybe_uninit_uninhabited": (
        in_helper(
            "    if mode & 8 != 0 {\n"
            "        #[allow(invalid_value)]\n"
            "        let _ = unsafe {\n"
            "            core::mem::MaybeUninit::<core::convert::Infallible>::uninit().assume_init()\n"
            "        };\n"
            "    }"
        ),
        "assert_inhabited requires a proved inhabited concrete type",
    ),
    "maybe_uninit_count": (
        in_helper(
            "    let initialized = core::mem::MaybeUninit::new(lane as i32);\n"
            "    let valid = unsafe { initialized.assume_init() };"
        ),
        "not proved block-uniform",
    ),
    "maybe_uninit_pointer": (
        in_helper(
            "    let initialized = unsafe { core::mem::MaybeUninit::new(input.add(lane)) };\n"
            "    let input = unsafe { initialized.assume_init() };"
        ),
        "not proved block-uniform",
    ),
    "early_return": (in_helper("    if lane & 1 == 0 { return; }"), "lane-varying"),
    "cycle": (
        in_helper(
            "    let mut remaining = lane;\n    while remaining > 0 { remaining -= 1; }"
        ),
        "lane-varying cycle",
    ),
    "default": (
        {
            "api": original["api"].replace(
                "(input, valid, -7_i32),",
                "(input, valid, cuda_device::thread::threadIdx_x() as i32),",
            )
        },
        "not proved block-uniform",
    ),
    # The same helper Instance first receives uniform inputs, then varying inputs.
    "call_context": (
        {
            "root": original["root"].replace(
                "roundtrip::<4, true>(input, output, post, valid, mode);",
                "roundtrip::<4, true>(input, output, post, valid, mode);\n            coop_marker_fixture_helper::roundtrip::<4, true>(input, output, post, valid + cuda_device::thread::threadIdx_x() as i32, mode);",
                1,
            )
        },
        "not proved block-uniform",
    ),
    "mixed_raw": (
        in_helper(
            "    unsafe { coop_marker_fixture_api::load::<ITEMS_PER_THREAD, TRANSPOSE>(input, post_store, valid, true); }"
        ),
        "raw and checked library calls",
    ),
    "missing_shape": (
        {"root": original["root"].replace("block = (128, 1, 1), ", "")},
        "requires #[launch_contract(block",
    ),
    "two_dimensional_shape": (
        {
            "root": original["root"].replace(
                "block = (128, 1, 1)", "block = (64, 2, 1)", 1
            )
        },
        "exact block shape [B,1,1]",
    ),
    "shared_shape": (
        {
            "root": "block = (64, 1, 1)".join(
                original["root"].rsplit("block = (128, 1, 1)", 1)
            )
        },
        None,
    ),
    "fixed_export": (
        {
            "root": original["root"].replace(
                "coop_marker_fixture_helper::roundtrip::<1, false>(input, output, post, valid, mode);",
                "exported_helper(input, output, post, valid, mode);",
                1,
            )
            + """
#[unsafe(no_mangle)]
unsafe fn exported_helper(input: *const i32, output: *mut i32, post: *mut i32, valid: i32, mode: u32) {
    unsafe { coop_marker_fixture_helper::roundtrip::<1, false>(input, output, post, valid, mode); }
}
"""
        },
        "fixed external symbol",
    ),
    "mode": (in_helper("    let mode = mode + (lane & 1) as u32;"), "lane-varying"),
    "borrow_mutation": (
        {
            "helper": original["helper"].replace(
                anchor,
                "    let mut changed = valid;\n    vary(&mut changed);\n    let valid = changed;\n"
                + anchor,
            )
            + "\n#[inline(never)]\nfn vary(value: &mut i32) { *value += cuda_device::thread::threadIdx_x() as i32; }\n"
        },
        "not proved block-uniform",
    ),
    "aggregate_reference": (
        in_helper(
            "    let mut changed = valid;\n"
            "    let holder = (&mut changed,);\n"
            "    *holder.0 = lane as i32;\n"
            "    let valid = changed;"
        ),
        "not proved block-uniform",
    ),
    "projected_reborrow": (
        in_helper(
            "    let mut pair = (lane as i32, valid);\n"
            "    let mut pointer = &mut pair;\n"
            "    if mode & 8 != 0 { pointer = &mut pair; }\n"
            "    let other = &mut pointer.1;\n"
            "    *other = 0;\n"
            "    let valid = pair.0;"
        ),
        "not proved block-uniform",
    ),
    "reconvergent_loop": (
        {
            "helper": original["helper"].replace(
                "payload[i] += CONTEXT_INCREMENT;",
                "if lane & 1 == 0 { payload[i] += CONTEXT_INCREMENT; }"
                " else { payload[i] += 2; }",
            )
        },
        None,
    ),
    "exhaustive_match": (
        {
            "helper": in_helper(
                "    let choice = if mode & 8 == 0 { Some(1) } else { None };\n"
                "    let adjustment = match choice { Some(value) => value, None => 0 };"
            )["helper"].replace(
                "payload[i] += CONTEXT_INCREMENT;",
                "payload[i] += CONTEXT_INCREMENT + adjustment;",
            )
        },
        None,
    ),
    "destructor": (
        {
            "helper": original["helper"].replace(
                anchor, "    let _guard = Guard;\n" + anchor
            )
            + "\nstruct Guard;\nimpl Drop for Guard { fn drop(&mut self) { cuda_device::thread::sync_threads(); } }\n"
        },
        "destructor",
    ),
    "recursion": (
        {
            "helper": original["helper"].replace(
                anchor, "    recurse(mode);\n" + anchor
            )
            + "\n#[inline(never)]\nfn recurse(n: u32) { if n > 0 { recurse(n - 1); } }\n"
        },
        "recursive helper",
    ),
    "unknown_call": (
        {
            "helper": original["helper"].replace(
                anchor, "    unsafe { external_effect(); }\n" + anchor
            )
            + '\nunsafe extern "C" { fn external_effect(); }\n'
        },
        "without MIR",
    ),
    "panic": (
        in_helper('    if lane & 1 == 0 { panic!("lane exit"); }'),
        "lane-varying",
    ),
    "unchecked_bounds": ({}, "unchecked indexing"),
}
probe_api = original["api"].replace("fn $name()", "pub fn $name()")
for name in ["native_mutation", "nested_reference", "nested_uniform", "atomic_control"]:
    source_name = "nested_reference" if name == "nested_uniform" else name
    probe = (
        Path(__file__).parent / "oxide-fixture/compile_cases" / f"{source_name}.rs"
    ).read_text()
    if name == "nested_uniform":
        probe = probe.replace("cuda_device::thread::threadIdx_x() as i32", "128_i32")
    cases[name] = (
        {"root": probe, "api": probe_api},
        None if name == "nested_uniform" else "not proved block-uniform",
    )
cases["reborrow_uniform"] = (
    {
        "helper": cases["projected_reborrow"][0]["helper"].replace(
            "let valid = pair.0;", "let valid = pair.1;"
        )
    },
    None,
)
cases["maybe_uninit_dead"] = (
    in_helper(
        "    match ITEMS_PER_THREAD {\n"
        "        0 => {\n"
        "            #[allow(invalid_value)]\n"
        "            let _ = unsafe {\n"
        "                core::mem::MaybeUninit::<core::convert::Infallible>::uninit().assume_init()\n"
        "            };\n"
        "        }\n"
        "        _ => {}\n"
        "    }"
    ),
    None,
)
cases.update(
    {
        "enum_uniform": (
            {
                "helper": in_helper(
                    "    let choice = enum_choice(mode & 1 == 0, valid);\n"
                    "    let moved = choice;\n"
                    "    let valid = match moved { Some(count) => count, None => valid };"
                )["helper"]
                + """
#[inline(never)]
fn enum_choice(present: bool, count: i32) -> Option<i32> {
    if present { Some(count) } else { None }
}
"""
            },
            None,
        ),
        "enum_uniform_tag": (
            in_helper(
                "    let choice = Some(lane);\n"
                "    match choice { Some(_) => cuda_device::thread::sync_threads(), None => {} }"
            ),
            None,
        ),
        "enum_presence": (
            in_helper(
                "    let choice = if lane & 1 == 0 { Some(valid) } else { None };\n"
                "    match choice { Some(_) => cuda_device::thread::sync_threads(), None => {} }"
            ),
            "lane-varying",
        ),
        "enum_fieldless": (
            in_helper(
                "    enum Choice { Present, Absent }\n"
                "    let choice = if lane & 1 == 0 { Choice::Present } else { Choice::Absent };\n"
                "    match choice { Choice::Present => cuda_device::thread::sync_threads(), Choice::Absent => {} }"
            ),
            "lane-varying",
        ),
        "enum_count": (
            in_helper(
                "    let choice = Some(lane as i32);\n"
                "    let valid = match choice { Some(count) => count, None => valid };"
            ),
            "not proved block-uniform",
        ),
        "enum_pointer": (
            in_helper(
                "    let choice = Some(unsafe { input.add(lane) });\n"
                "    let input = match choice { Some(pointer) => pointer, None => input };"
            ),
            "not proved block-uniform",
        ),
        "enum_default": (
            {
                "api": original["api"].replace(
                    "(input, valid, -7_i32),",
                    "(input, valid, match Some(cuda_device::thread::threadIdx_x() as i32) { Some(value) => value, None => -7_i32 }),",
                )
            },
            "not proved block-uniform",
        ),
        "enum_mutation": (
            {
                "helper": in_helper(
                    "    let mut choice = Some(valid);\n"
                    "    replace_choice(&mut choice, lane as i32);\n"
                    "    let valid = match choice { Some(count) => count, None => valid };"
                )["helper"]
                + """
#[inline(never)]
fn replace_choice(choice: &mut Option<i32>, count: i32) {
    *choice = Some(count);
}
"""
            },
            "not proved block-uniform",
        ),
        "enum_projection": (
            in_helper(
                "    let choice = Some(lane as u32);\n"
                "    let raw = &choice as *const Option<u32> as *const (u32, u32);\n"
                "    let valid = unsafe { (*raw).1 } as i32;"
            ),
            "not proved block-uniform",
        ),
    }
)
unknown = set(args.case) - cases.keys()
if unknown:
    parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
results = {}
try:
    for name, (changes, diagnostic) in cases.items():
        if args.case and name not in args.case:
            continue
        for key, path in paths.items():
            path.write_text(changes.get(key, original[key]))
        helper_lib.write_text(
            helper_lib_original
            + (
                '\n#[cfg(feature = "checked")]\npub use checked::shape_probe;\n'
                if name == "distinct_shapes"
                else ""
            )
        )
        if name in [
            "native_mutation",
            "nested_reference",
            "nested_uniform",
            "atomic_control",
        ]:
            helper_lib.write_text(
                helper_lib_original
                + "\npub use coop_marker_fixture_api::{direct_load_full, direct_load_filled};\n"
            )
        case_env = env.copy()
        if name == "unchecked_bounds":
            case_env["CUDA_OXIDE_UNCHECKED_INDEXING"] = "1"
        if name in [
            "aggregate_reference",
            "projected_reborrow",
            "reborrow_uniform",
            "exhaustive_match",
        ]:
            # Preserve reference projections and exhaustive-match branches
            # before the checker sees them.
            case_env["RUSTFLAGS"] = (
                case_env.get("RUSTFLAGS", "") + " -Zmir-opt-level=0"
            ).strip()
        log_path = output / f"reject-{name}.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=fixture / "consumer",
                env=case_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        text = log_path.read_text()
        if diagnostic is None:
            assert result.returncode == 0, (
                f"{name}: expected successful compilation; see {log_path}"
            )
        else:
            assert result.returncode != 0, f"{name} unexpectedly compiled"
            assert diagnostic in text, (
                f"{name}: expected {diagnostic!r}; see {log_path}"
            )
        results[name] = {"returncode": result.returncode, "diagnostic": diagnostic}
        print(f"PASS {name}", flush=True)
finally:
    helper_lib.write_text(helper_lib_original)
    for key, path in paths.items():
        path.write_text(original[key])
    (output / "rejections.json").write_text(json.dumps(results, indent=2) + "\n")
