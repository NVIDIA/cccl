# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile storage/layout/epoch contrasts in an isolated --storage fixture."""

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
source = output / "fixture/consumer/src/storage.rs"
original = source.read_text()
command = record["argv"].copy()
if "run" in command:
    command[command.index("run")] = "build"
env = os.environ.copy()
env.update(
    CUDA_OXIDE_BACKEND=record["backend"],
    CUDA_HOME=record["cuda_root"],
    CUDA_PATH=record["cuda_root"],
)
prefix = """// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
use coop_marker_fixture_helper::storage::{load,store,reload};
use cuda_device::{CollectiveTiles,DisjointSlice,library,thread,kernel,launch_contract};
#[kernel]
#[launch_contract(domain=1,block=(128,1,1),dynamic_shared=inferred)]
pub fn probe(input:&[i32],mut output:DisjointSlice<i32,CollectiveTiles<4>>,valid:usize) {
    let offset=thread::blockIdx_x() as usize*512;
    let mut values=[0_i32;1];
    unsafe {
"""
suffix = "\n}}\npub fn run()->Result<(),Box<dyn std::error::Error>> {Ok(())}\n"
new = "let mut scratch=library::storage_new_v1::<0,0,false,false>();\n"
load = "load::<1,true,_>(input,offset,valid,&mut values,&mut scratch);\n"
store = "store::<1,true,_>(&mut output,offset,valid,&mut values,&mut scratch);\n"
reserve = "let view=library::storage_reserve_v1::<i32,_,128,0>(&mut scratch);\n"
release = "library::storage_release_v1(view);\n"
sync = "thread::sync_threads();\n"
cases = {
    "manual_reuse": (new + load + load, "explicit storage reuse requires"),
    "manual_sync": (new + load + sync + load, None),
    "automatic": (new.replace("false,false", "true,false") + load + load, None),
    "exclusive_sites": (new.replace("false,false", "false,true") + load + load, None),
    "exclusive_loop": (
        new.replace("false,false", "false,true")
        + "let mut i=0;while i<2 {"
        + load
        + "i+=1;}",
        "explicit storage reuse requires",
    ),
    "managed_loop": (
        "let mut i=0;while i<2 {load::<1,true,_>(input,offset,valid,&mut values,());i+=1;}",
        None,
    ),
    "constructor_loop": (
        "let mut i=0;while i<2 {" + new + load + "i+=1;}",
        "explicit storage reuse requires",
    ),
    "live_reservation": (
        new + reserve + sync + load + release,
        "overlaps a live reservation",
    ),
    "release_without_sync": (
        new + reserve + release + load,
        "explicit storage reuse requires",
    ),
    "sync_before_release": (
        new + reserve + sync + release + load,
        "explicit storage reuse requires",
    ),
    "release_then_sync": (new + reserve + release + sync + load, None),
    "forgotten_view": (
        new + reserve + "core::mem::forget(view);" + sync + load,
        "overlaps a live reservation",
    ),
    "exclusive_views": (
        new.replace("false,false", "false,true")
        + reserve
        + "let other=library::storage_reserve_v1::<u64,_,3,32>(&mut scratch);"
        + release
        + "library::storage_release_v1(other);",
        None,
    ),
    "reserve_auto": (
        new.replace("false,false", "true,false") + reserve + release,
        "reservations cannot use auto_sync",
    ),
    "zero_extent": (
        new + reserve.replace("128,0", "0,0") + release,
        "extent must be positive",
    ),
    "invalid_alignment": (
        new.replace("0,0,false", "0,3,false") + load,
        "alignment must be zero or a power of two",
    ),
    "unsupported_scalar": (
        new + reserve.replace("::<i32,", "::<bool,") + release,
        "unsupported library scalar",
    ),
    "complete_capacity_under": (
        new.replace("0,0,false,false", "528,0,false,true") + load + load,
        "smaller than complete plan",
    ),
    "complete_capacity_exact": (
        new.replace("0,0,false,false", "1024,0,false,true") + load + load,
        None,
    ),
    "complete_capacity_extra": (
        new.replace("0,0,false,false", "2048,64,false,true") + load + load,
        None,
    ),
    "ambiguous_origin": (
        new
        + "let mut other=library::storage_new_v1::<0,0,false,false>();let selected=if valid==0 {&mut scratch}else{&mut other};load::<1,true,_>(input,offset,valid,&mut values,selected);",
        "no unambiguous descriptor origin",
    ),
    "direct_output_epoch": (
        "store::<1,false,_>(&mut output,offset,valid,&mut values,());reload::<1,false,_>(&mut output,offset,valid,&mut values,());",
        "conflicting collective resource access",
    ),
    "managed_output_epoch": (
        "store::<1,true,_>(&mut output,offset,valid,&mut values,());reload::<1,true,_>(&mut output,offset,valid,&mut values,());",
        None,
    ),
    "independent_output_epoch": (
        new
        + "let mut other=library::storage_new_v1::<0,0,false,false>();"
        + store
        + "reload::<1,true,_>(&mut output,offset,valid,&mut values,&mut other);",
        "conflicting collective resource access",
    ),
    "explicit_output_sync": (
        new
        + store
        + sync
        + "reload::<1,true,_>(&mut output,offset,valid,&mut values,&mut scratch);",
        None,
    ),
    "native_fixed_prefix": (
        "let _prefix=cuda_device::DynamicSharedArray::<[u64;8],32>::get_fixed();"
        + new
        + load
        + sync
        + store,
        None,
    ),
    "native_unconstrained": (
        "let _prefix=cuda_device::DynamicSharedArray::<u64,32>::get();" + new + load,
        "unconstrained native dynamic shared view",
    ),
}
cases.update(
    {
        "configure_before_use": (
            new
            + "let mut scratch: library::StorageDescriptor<0,32,false,false>=library::storage_reconfigure_v1(scratch);"
            + load,
            None,
        ),
        "configure_after_use": (
            new
            + load
            + sync
            + "let mut scratch: library::StorageDescriptor<0,32,false,false>=library::storage_reconfigure_v1(scratch);"
            + load,
            "storage policy cannot change after use",
        ),
        "configure_chain": (
            new
            + "let scratch: library::StorageDescriptor<1024,0,false,false>=library::storage_reconfigure_v1(scratch);let mut scratch: library::StorageDescriptor<1024,32,false,false>=library::storage_reconfigure_v1(scratch);"
            + load,
            None,
        ),
    }
)
wrapper = """
struct Wrapper { raw: library::StorageDescriptor }
#[inline(never)]
fn use_raw(raw: &mut library::StorageDescriptor) {
    unsafe {
        let view = library::storage_reserve_v1::<i32, _, 128, 0>(raw);
        library::storage_release_v1(view);
    }
}
#[inline(never)]
fn use_wrapper(wrapper: &mut Wrapper) { use_raw(&mut wrapper.raw); }
let mut wrapper = Wrapper { raw: library::storage_new_v1::<0, 0, false, false>() };
"""
cases.update(
    {
        "wrapper_raw_borrow": (wrapper + "use_raw(&mut wrapper.raw);", None),
        "wrapper_local_field": (
            wrapper
            + "let view=library::storage_reserve_v1::<i32,_,128,0>(&mut wrapper.raw);"
            + release,
            None,
        ),
        "wrapper_helper": (wrapper + "use_wrapper(&mut wrapper);", None),
        "wrapper_helper_reuse": (
            wrapper + "use_wrapper(&mut wrapper);use_wrapper(&mut wrapper);",
            "explicit storage reuse requires",
        ),
        "wrapper_helper_reuse_sync": (
            wrapper
            + "use_wrapper(&mut wrapper);"
            + sync
            + "use_wrapper(&mut wrapper);",
            None,
        ),
        "wrapper_returned_field": (
            wrapper
            + """
#[inline(never)]
fn raw(wrapper: &mut Wrapper) -> &mut library::StorageDescriptor { &mut wrapper.raw }
use_raw(raw(&mut wrapper));
""",
            None,
        ),
        "wrapper_nested": (
            wrapper
            + """
struct Outer { inner: Wrapper }
#[inline(never)]
fn use_outer(outer: &mut Outer) { use_wrapper(&mut outer.inner); }
let mut outer = Outer { inner: wrapper };
use_outer(&mut outer);
""",
            None,
        ),
        "wrapper_ambiguous": (
            wrapper
            + "let mut other=Wrapper { raw: library::storage_new_v1::<0,0,false,false>() };"
            + "use_wrapper(if valid==0 {&mut wrapper} else {&mut other});",
            "no unambiguous descriptor origin",
        ),
        "wrapper_callee_ambiguous": (
            wrapper
            + """
#[inline(never)]
fn pick<'a>(a: &'a mut Wrapper, b: &'a mut Wrapper, count: usize)
    -> &'a mut library::StorageDescriptor
{
    if count == 0 { &mut a.raw } else { &mut b.raw }
}
let mut other=Wrapper { raw: library::storage_new_v1::<0,0,false,false>() };
use_raw(pick(&mut wrapper, &mut other, valid));
""",
            "no unambiguous descriptor origin",
        ),
        "wrapper_replacement": (
            wrapper
            + """
#[inline(never)]
fn replace(wrapper: &mut Wrapper, other: Wrapper) { *wrapper = other; }
let other=Wrapper { raw: library::storage_new_v1::<0,0,false,false>() };
replace(&mut wrapper, other);
use_wrapper(&mut wrapper);
""",
            "mutation through a reference to a resource descriptor",
        ),
        "wrapper_enum_replacement": (
            wrapper
            + """
enum Storage { Present(Wrapper), Absent }
#[inline(never)]
fn replace(storage: &mut Storage) { *storage = Storage::Absent; }
let mut storage = Storage::Present(wrapper);
replace(&mut storage);
""",
            "mutation through a reference to a resource descriptor",
        ),
        "control_scalar_invalidation": (
            new
            + """
struct Scalar { value: usize }
#[inline(never)]
fn change(value: &mut Scalar) { value.value=thread::threadIdx_x() as usize; }
let mut control=Scalar { value: valid };
change(&mut control);
"""
            + load.replace("offset,valid", "offset,control.value"),
            "lane-varying control flow",
        ),
    }
)
for capacity in [55, 56]:
    cases[f"padded_capacity_{capacity}"] = (
        f"let mut scratch=library::storage_new_v1::<{capacity},0,false,true>();"
        "let first=library::storage_reserve_v1::<i32,_,3,0>(&mut scratch);"
        "let second=library::storage_reserve_v1::<u64,_,3,32>(&mut scratch);"
        "library::storage_release_v1(first);library::storage_release_v1(second);",
        "smaller than complete plan" if capacity == 55 else None,
    )
for dtype in ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64"]:
    cases[f"reserve_{dtype}"] = (
        new + reserve.replace("::<i32,_,128,0>", f"::<{dtype},_,3,0>") + release,
        None,
    )
unknown = set(args.case) - cases.keys()
if unknown:
    parser.error(f"unknown cases: {sorted(unknown)}")
results = {}
try:
    for name, (body, diagnostic) in cases.items():
        if args.case and name not in args.case:
            continue
        source.write_text(prefix + body + suffix)
        log_path = output / f"storage-{name}.log"
        case_env = env.copy()
        case_env["CARGO_TARGET_DIR"] = str(output / "storage-targets" / name)
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=record["cwd"],
                env=case_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        text = log_path.read_text()
        passed = (
            result.returncode == 0
            if diagnostic is None
            else result.returncode != 0 and diagnostic in text
        )
        results[name] = {
            "returncode": result.returncode,
            "expected": diagnostic,
            "passed": passed,
        }
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            print(text[-5000:], flush=True)
finally:
    source.write_text(original)
    (output / "storage-compile-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
raise SystemExit(0 if all(item["passed"] for item in results.values()) else 1)
