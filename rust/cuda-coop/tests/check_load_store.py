# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile public Load/Store participation and resource-epoch contrasts.

Use the same Cargo config/backend as the qualified public example. The config
is optional for an installed compatible Oxide revision; local source integration
can supply explicit Cargo patches without changing checked-in manifests.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cargo-oxide", required=True, type=Path)
parser.add_argument("--backend", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cargo-config", type=Path)
parser.add_argument(
    "--cargo-lock", type=Path, help="Compatible lockfile for local source patches"
)
parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
parser.add_argument("--arch", default="sm_120")
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
(output / "src").mkdir()
package = Path(__file__).resolve().parents[1]
manifest = (package / "tests/load_store/Cargo.toml").read_text()
manifest = manifest.replace('path = "../.."', f"path = {json.dumps(str(package))}")
(output / "Cargo.toml").write_text(manifest)
shutil.copyfile(
    args.cargo_lock or package / "tests/load_store/Cargo.lock", output / "Cargo.lock"
)
source = output / "src/main.rs"
prefix = """// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
use cuda_coop as coop;
use cuda_device::{DisjointSlice,CollectiveTiles,kernel,launch_contract,thread};
#[kernel]
#[launch_contract(domain=1,block=(128,1,1),dynamic_shared=inferred)]
pub fn probe(input:&[i32],mut output:DisjointSlice<i32,CollectiveTiles<4>>,valid:usize) {
 let block=coop::this_block();
 let offset=thread::blockIdx_x() as usize*512;
 let mut values=coop::ThreadData::<_,4>::new();
"""
load = "coop::load(block,input,&mut values,coop::LoadOptions::default().offset(offset).valid_items(valid).oob_default(0));\n"
store = "coop::store(block,&mut output,&values,coop::StoreOptions::default().offset(offset).valid_items(valid));\n"
reload = "coop::load(block,&mut output,&mut values,coop::LoadOptions::default().offset(offset).valid_items(valid));\n"
cases = {
    "varying_presence": (
        "let options=if thread::threadIdx_x()&1==0 { coop::LoadOptions::default().valid_items(valid) } else { coop::LoadOptions::default() };coop::load(block,input,&mut values,options);",
        "lane-varying",
    ),
    "varying_count": (
        load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved block-uniform",
    ),
    "varying_default": (
        load.replace(".oob_default(0)", ".oob_default(thread::threadIdx_x() as i32)"),
        "not proved block-uniform",
    ),
    "varying_offset": (
        load.replace(
            ".offset(offset)", ".offset(offset+thread::threadIdx_x() as usize)"
        ),
        "not proved block-uniform",
    ),
    "varying_participation": (
        "if thread::threadIdx_x()&1==0 {" + load + "}",
        "lane-varying",
    ),
    "early_return": ("if thread::threadIdx_x()!=0 { return; }" + load, "lane-varying"),
    "missing_completion": (
        load + store + reload,
        "requires a barrier covering both accesses",
    ),
    "write_after_read": (
        reload
        + "coop::store(block,&mut output,1_i32,coop::StoreOptions::default().offset(offset));",
        "requires a barrier covering both accesses",
    ),
    "loop_without_completion": (
        load + "let mut iteration=0;while iteration<2 {" + store + "iteration+=1;}",
        "requires a barrier covering both accesses",
    ),
    "loop_with_completion": (
        load
        + "let mut iteration=0;while iteration<2 {"
        + store
        + "thread::sync_threads();iteration+=1;}",
        None,
    ),
    "explicit_completion": (load + store + "thread::sync_threads();" + reload, None),
    "shared_payload": (
        load + "let shared=&values;" + store.replace("&values", "shared"),
        None,
    ),
    "reconverged_selection": (
        "let mut other=coop::ThreadData::<_,4>::new();values.write(0,7);other.write(0,9);"
        "let selected=if thread::threadIdx_x()&1==0 {values} else {other};"
        "coop::store(block,&mut output,&selected,coop::StoreOptions::default().offset(offset).valid_items(1));",
        None,
    ),
}
transpose_load = load.replace(
    ".oob_default(0)", ".oob_default(0).algorithm(coop::Transpose)"
)
transpose_store = store.replace("&values", "&mut values").replace(
    ".valid_items(valid)", ".valid_items(valid).algorithm(coop::Transpose)"
)
cases.update(
    shared_transpose=(
        load + transpose_store.replace("&mut values", "&values"),
        "StoreValue<i32, Transpose>",
    ),
    transpose_managed_reuse=(
        transpose_load + transpose_store + transpose_store,
        None,
    ),
    transpose_no_trailing_completion=(
        transpose_load + transpose_store + reload,
        "requires a barrier covering both accesses",
    ),
    transpose_varying_count=(
        transpose_load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved block-uniform",
    ),
    transpose_varying_participation=(
        "if thread::threadIdx_x()&1==0 {" + transpose_load + "}",
        "lane-varying",
    ),
    transpose_loop=(
        transpose_load
        + "let mut iteration=0;while iteration<2 {"
        + transpose_store
        + "iteration+=1;}",
        None,
    ),
    generic_algorithm=(
        "values.write(0,0_i32);fn forward<A:coop::Algorithm>(block:coop::ThreadGroup<coop::Block>,input:&[i32],output:&mut DisjointSlice<i32,CollectiveTiles<4>>,algorithm:A) {"
        "let mut values=coop::ThreadData::<_,4>::new();"
        "coop::load(block,input,&mut values,coop::LoadOptions::default().algorithm(algorithm));"
        "coop::store(block,output,&mut values,coop::StoreOptions::default().algorithm(algorithm));}"
        "forward(block,input,&mut output,coop::Transpose);",
        None,
    ),
)

# Public descriptors preserve the existing static-site and completion rules.
new_storage = "let mut scratch=coop::TempStorage::new();"
sync = "thread::sync_threads();"
explicit_load = transpose_load.replace(
    ".algorithm(coop::Transpose)",
    ".algorithm(coop::Transpose).temp_storage(&mut scratch)",
)
explicit_store = transpose_store.replace(
    ".algorithm(coop::Transpose)",
    ".algorithm(coop::Transpose).temp_storage(&mut scratch)",
)
reserve = "let view=scratch.reserve::<i32,3>();"
release = "view.release();"
cases.update(
    storage_manual=(new_storage + explicit_load + sync + explicit_store, None),
    storage_manual_missing_barrier=(
        new_storage + explicit_load + explicit_store,
        "requires a full-block barrier",
    ),
    storage_auto_loop=(
        new_storage.replace("new()", "new().auto_sync()")
        + "let mut iteration=0;while iteration<2 {"
        + explicit_load
        + explicit_store
        + "iteration+=1;}",
        None,
    ),
    storage_direct_auto_no_completion=(
        new_storage.replace("new()", "new().auto_sync()")
        + load
        + store.replace(
            ".valid_items(valid)", ".valid_items(valid).temp_storage(&mut scratch)"
        )
        + reload,
        "requires a barrier covering both accesses",
    ),
    storage_exclusive_sites=(
        new_storage.replace("new()", "new().exclusive()")
        + explicit_load
        + explicit_load
        + explicit_store,
        None,
    ),
    storage_exclusive_loop=(
        new_storage.replace("new()", "new().exclusive()")
        + "let mut iteration=0;while iteration<2 {"
        + explicit_load
        + "iteration+=1;}",
        "requires a full-block barrier",
    ),
    storage_reserve_auto=(
        new_storage.replace("new()", "new().auto_sync()") + reserve + release,
        "no method named `reserve`",
    ),
    storage_reserve_live=(
        new_storage + reserve + sync + explicit_load + release,
        "overlaps a live reservation",
    ),
    storage_release_no_barrier=(
        new_storage + reserve + release + explicit_load,
        "requires a full-block barrier",
    ),
    storage_barrier_before_release=(
        new_storage + reserve + sync + release + explicit_load,
        "requires a full-block barrier",
    ),
    storage_release_then_barrier=(
        new_storage + reserve + release + sync + explicit_load,
        None,
    ),
    storage_reserve_exclusive=(
        new_storage.replace("new()", "new().exclusive()")
        + reserve
        + "let other=scratch.reserve_aligned::<u64,3,32>();"
        + release
        + "other.release();",
        None,
    ),
    storage_extent_zero=(
        new_storage + reserve.replace("i32,3", "i32,0") + release,
        "assertion failed: N > 0",
    ),
    storage_numeric_only=(
        new_storage + reserve.replace("i32,3", "bool,3") + release,
        "Numeric",
    ),
    storage_alignment_invalid=(
        "let _scratch=coop::TempStorage::new().with_alignment::<3>();",
        "assertion failed",
    ),
    storage_capacity_zero=(
        "let _scratch=coop::TempStorage::new().with_capacity::<0>();",
        "assertion failed",
    ),
    storage_options_consumed=(
        new_storage
        + "let options=coop::LoadOptions::default().temp_storage(&mut scratch);coop::load(block,input,&mut values,options);coop::load(block,input,&mut values,options);",
        "use of moved value: `options`",
    ),
    storage_policy_after_direct=(
        new_storage
        + load.replace(".oob_default(0)", ".oob_default(0).temp_storage(&mut scratch)")
        + "let _changed=scratch.exclusive();",
        "storage policy cannot change after use",
    ),
    storage_descriptor_mutation=(
        "#[inline(never)] fn replace(old:&mut coop::TempStorage,new:coop::TempStorage){*old=new;}"
        + new_storage
        + "let other=coop::TempStorage::new();replace(&mut scratch,other);",
        "mutation through a reference to a resource descriptor",
    ),
    storage_reservation_mutation=(
        "#[inline(never)] fn replace(old:&mut coop::StorageReservation<i32,3>,new:coop::StorageReservation<i32,3>){*old=new;}"
        + new_storage
        + "let mut view=scratch.reserve::<i32,3>();let mut other=coop::TempStorage::new();let replacement=other.reserve::<i32,3>();replace(&mut view,replacement);view.release();",
        "mutation through a reference to a resource descriptor",
    ),
)
for capacity in (55, 56):
    cases[f"storage_complete_capacity_{capacity}"] = (
        f"let mut scratch=coop::TempStorage::new().with_capacity::<{capacity}>().exclusive();"
        + reserve
        + "let other=scratch.reserve_aligned::<u64,3,32>();"
        + release
        + "other.release();",
        "smaller than complete plan" if capacity == 55 else None,
    )

# Storage-only cases still infer the payload type from a typed producer.
for name, (body, diagnostic) in list(cases.items()):
    if name.startswith("storage_"):
        cases[name] = ("values.write(0, 0_i32);" + body, diagnostic)

# Physical Warp operations use the same public calls and canonical carrier.
# Query precision is granted only to the genuine compiler query, not to
# arithmetic that happens to divide the lane rank by 32.
warp_prefix = (
    prefix.replace("block=(128,1,1)", "block=(64,1,1)").replace("*512", "*256")
    + "let warp=coop::this_warp();let q=cuda_device::library::warp_group_index_v1::<32>() as usize;\n"
)
warp_load = load.replace("(block,", "(warp,")
warp_store = store.replace("(block,", "(warp,")
warp_reload = reload.replace("(block,", "(warp,")
warp_helper = """#[inline(never)] fn put(group:coop::ThreadGroup<coop::Warp>,
 output:&mut DisjointSlice<i32,CollectiveTiles<4>>,offset:usize) {
 coop::store(group,output,7_i32,coop::StoreOptions::default().offset(offset)); }
"""
warp_loop = "let mut i=0;while i<q+1 {" + warp_load + "i+=1;}"
cases.update(
    warp_sibling_participation=(
        warp_load + "if q==0 {" + warp_store + "}thread::sync_threads();" + reload,
        None,
    ),
    warp_lane_participation=(
        "if thread::threadIdx_x()&1==0 {" + warp_load + "}",
        "lane-varying",
    ),
    warp_sibling_count=(
        warp_load.replace(".valid_items(valid)", ".valid_items(q+1)"),
        None,
    ),
    warp_sibling_presence=(
        "let options=if q==0 { coop::LoadOptions::default().offset(offset).valid_items(valid) } else { coop::LoadOptions::default().offset(offset) };"
        "coop::load(warp,input,&mut values,options);",
        None,
    ),
    warp_lane_count=(
        warp_load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved Warp-uniform",
    ),
    warp_arithmetic_origin=(
        warp_load.replace(
            ".offset(offset)", ".offset((thread::threadIdx_x() as usize/32)*128)"
        ),
        "not proved Warp-uniform",
    ),
    warp_block_control=(
        load.replace(".offset(offset)", ".offset(q*128)"),
        "not proved block-uniform",
    ),
    warp_read_after_write=(
        warp_load + warp_store + warp_reload,
        "requires a barrier covering both accesses",
    ),
    warp_read_after_completion=(
        warp_load + warp_store + "thread::sync_threads();" + warp_reload,
        None,
    ),
    warp_write_after_read=(
        warp_reload + warp_store,
        "requires a barrier covering both accesses",
    ),
    warp_to_block=(
        warp_load + warp_store + reload,
        "requires a barrier covering both accesses",
    ),
    warp_from_block=(
        load + store + warp_reload,
        "requires a barrier covering both accesses",
    ),
    warp_from_block_completed=(
        load + store + "thread::sync_threads();" + warp_reload,
        None,
    ),
    warp_helper_reuse=(
        "values.write(0,0_i32);"
        + warp_helper
        + "put(warp,&mut output,offset);put(warp,&mut output,offset);",
        "requires a barrier covering both accesses",
    ),
    warp_helper_completed=(
        "values.write(0,0_i32);"
        + warp_helper
        + "put(warp,&mut output,offset);thread::sync_threads();put(warp,&mut output,offset);",
        None,
    ),
    warp_sibling_cycle=(warp_loop + warp_load, None),
    warp_cycle_then_block=(warp_loop + load, "lane-varying"),
    warp_lane_cycle=(
        warp_loop.replace("i<q+1", "i<thread::threadIdx_x() as usize+1") + warp_load,
        "lane-varying",
    ),
    warp_helper_cycle=(
        "#[inline(never)] fn cycles(input:&[i32]) {let q=cuda_device::library::warp_group_index_v1::<32>() as usize;let mut values=coop::ThreadData::<_,4>::new();let warp=coop::this_warp();let valid=1;let offset=0;"
        + warp_loop
        + "}cycles(input);"
        + load,
        "lane-varying",
    ),
    warp_partial_geometry=(warp_load, "complete physical warps"),
    warp_transpose=(
        warp_load.replace(
            ".oob_default(0)", ".oob_default(0).algorithm(coop::Transpose)"
        ),
        "expected `Direct`, found `Transpose`",
    ),
    warp_explicit_storage=(
        new_storage
        + warp_load.replace(
            ".oob_default(0)", ".oob_default(0).temp_storage(&mut scratch)"
        ),
        "expected `LoadOptions<i32>`",
    ),
)

striped_load = load.replace(
    ".oob_default(0)", ".oob_default(0).algorithm(coop::Striped)"
)
striped_store = store.replace(
    ".valid_items(valid)", ".valid_items(valid).algorithm(coop::Striped)"
)
cases.update(
    striped_shared=(striped_load + striped_store, None),
    striped_varying_count=(
        striped_load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved block-uniform",
    ),
    striped_missing_completion=(
        striped_load + striped_store + reload,
        "requires a barrier covering both accesses",
    ),
)


vectorize_load = load.replace(
    ".oob_default(0)", ".oob_default(0).algorithm(coop::Vectorize)"
)
vectorize_store = store.replace(
    ".valid_items(valid)", ".valid_items(valid).algorithm(coop::Vectorize)"
)
cases.update(
    vectorize_shared=(vectorize_load + vectorize_store, None),
    vectorize_varying_offset=(
        vectorize_load.replace(
            ".offset(offset)", ".offset(offset+thread::threadIdx_x() as usize)"
        ),
        "not proved block-uniform",
    ),
    vectorize_missing_completion=(
        vectorize_load + vectorize_store + reload,
        "requires a barrier covering both accesses",
    ),
)

# Both algorithms participate as complete blocks and mutate the payload.
for name, selector in [
    ("warp_transpose", "WarpTranspose"),
    ("warp_timesliced", "WarpTransposeTimesliced"),
]:
    warp_load = transpose_load.replace("coop::Transpose", f"coop::{selector}")
    warp_store = transpose_store.replace("coop::Transpose", f"coop::{selector}")
    cases[name + "_shared"] = (
        load + warp_store.replace("&mut values", "&values"),
        f"StoreValue<i32, {selector}>",
    )
    cases[name + "_managed_reuse"] = (warp_load + warp_store + warp_store, None)
    cases[name + "_scalar"] = (
        warp_load
        + f"coop::store(block,&mut output,-17_i32,coop::StoreOptions::default().offset(offset).algorithm(coop::{selector}));",
        None,
    )
    cases[name + "_partial_block"] = (warp_load, "divisible by 32")
    cases[name + "_warp_only_participation"] = (
        "if thread::threadIdx_x()<32 {" + warp_load + "}",
        "lane-varying",
    )

# Logical group precision and completion use canonical public descriptors.
logical_prefix = warp_prefix.replace(
    "let warp=coop::this_warp();", "let warp=coop::this_warp().group_by::<8>();"
).replace(
    "cuda_device::library::warp_group_index_v1::<32>() as usize",
    "warp.rank_at::<coop::Warp>() as usize",
)
logical_load = load.replace("(block,", "(warp,")
logical_store = store.replace("(block,", "(warp,")
logical_reload = reload.replace("(block,", "(warp,")
logical_sync = "use cuda_device::cooperative_groups::ThreadGroup as _;cuda_device::cooperative_groups::this_thread_block().tiled_partition::<8>().sync();"
wide_sync = logical_sync.replace("::<8>()", "::<32>()")
narrow_sync = logical_sync.replace("::<8>()", "::<4>()")
logical_loop = "let mut i=0;while i<q+1 {" + logical_load + "i+=1;}"
cases.update(
    logical_matching_completion=(
        logical_load + logical_store + logical_sync + logical_reload,
        None,
    ),
    logical_containing_completion=(
        logical_load + logical_store + wide_sync + logical_reload,
        None,
    ),
    logical_narrow_completion=(
        logical_load + logical_store + narrow_sync + logical_reload,
        "requires a barrier covering both accesses",
    ),
    logical_write_after_read=(logical_reload + logical_sync + logical_store, None),
    logical_terminal_divergent_sync=(
        logical_load + "if thread::threadIdx_x()&1==0 {" + logical_sync + "}",
        "control narrower than its width",
    ),
    logical_terminal_wide_sync=(
        logical_load + "if q==0 {" + wide_sync + "}",
        "control narrower than its width",
    ),
    logical_sibling_sync=(
        "if q==0 {"
        + logical_load
        + logical_store
        + logical_sync
        + logical_reload
        + "}",
        None,
    ),
    logical_raw_sync_no_credit=(
        logical_load
        + logical_store
        + "cuda_device::warp::sync_mask(u32::MAX);"
        + logical_reload,
        "intrinsic is outside",
    ),
    logical_to_physical=(
        logical_load
        + logical_store
        + logical_sync
        + logical_reload.replace("(warp,", "(coop::this_warp(),"),
        "requires a barrier covering both accesses",
    ),
    logical_to_physical_completed=(
        logical_load
        + logical_store
        + wide_sync
        + logical_reload.replace("(warp,", "(coop::this_warp(),"),
        None,
    ),
    logical_read_kinds_retained=(
        reload + logical_reload + logical_sync + logical_store,
        "requires a barrier covering both accesses",
    ),
    logical_block_dirty_retained=(
        new_storage + explicit_load + logical_sync + explicit_store,
        "requires a full-block barrier",
    ),
    logical_managed_block_reuse=(transpose_load + logical_sync + transpose_load, None),
    logical_completion_join=(
        logical_load
        + logical_store
        + "if valid==1 {"
        + logical_sync
        + "}"
        + logical_reload,
        "requires a barrier covering both accesses",
    ),
    logical_loop_completion=(
        logical_load
        + "let mut i=0;while i<2 {"
        + logical_store
        + logical_sync
        + "i+=1;}"
        + logical_reload,
        None,
    ),
    logical_cycle_to_wider=(logical_loop + load, "lane-varying"),
    logical_sibling_loop=(logical_loop + logical_load, None),
    logical_helper_cycle=(
        "#[inline(never)] fn cycle(input:&[i32]) {let warp=coop::this_warp().group_by::<8>();let q=warp.rank_at::<coop::Warp>() as usize;let valid=1;let offset=0;let mut values=coop::ThreadData::<_,4>::new();"
        + logical_loop
        + "}cycle(input);"
        + load,
        "lane-varying",
    ),
    logical_lane_control=(
        logical_load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved Warp-uniform",
    ),
    logical_public_rank_control=(
        logical_load.replace(".valid_items(valid)", ".valid_items(q+1)"),
        None,
    ),
    logical_group6=(
        logical_load.replace(
            "(warp,", "(coop::this_warp().group_by_non_exhaustive::<6>(),"
        ),
        "LoadStoreGroup",
    ),
    logical_width1_origin=(
        "let chosen=if thread::threadIdx_x()&1==0 {&mut output}else{&mut other};"
        + logical_store.replace(
            "(warp,&mut output", "(coop::this_warp().group_by::<1>(),chosen"
        ),
        "not proved Warp-uniform",
    ),
    logical_structural_width6=(
        'values.write(0,0_i32);let mut local=[0_i32;4];let flags=[1_u8;4];fn registration()->&\'static str {"unused"} unsafe {cuda_device::library::checked_warp_call_v1::<i32,_,_,_,_,6,4,true>(registration,&mut output,(core::ptr::null_mut::<i32>(),),(local.as_mut_ptr(),flags.as_ptr()));}',
        "power-of-two width in 1..=32",
    ),
)
for width in (1, 2, 4, 8, 16, 32):
    for exhaustive in (True, False):
        constructor = "group_by" if exhaustive else "group_by_non_exhaustive"
        cases[f"logical_width{width}_{exhaustive}"] = (
            logical_load.replace(
                "(warp,", f"(coop::this_warp().{constructor}::<{width}>(),"
            ),
            None,
        )


command = [
    str(args.cargo_oxide.resolve()),
    "oxide",
    "build",
    "--arch",
    args.arch,
    "--materialize-cubin",
    "--",
    "--locked",
]
if args.cargo_config:
    command += ["--config", str(args.cargo_config.resolve())]
env = os.environ.copy()
env.update(
    CUDA_OXIDE_BACKEND=str(args.backend.resolve()),
    CUDA_HOME=str(args.cuda_root.resolve()),
    CUDA_PATH=str(args.cuda_root.resolve()),
)
results = []
for name, (body, diagnostic) in cases.items():
    block_warp = name.startswith(("warp_transpose_", "warp_timesliced_"))
    case_prefix = warp_prefix if name.startswith("warp_") and not block_warp else prefix
    if name.startswith("logical_"):
        case_prefix = logical_prefix
    if name == "logical_width1_origin":
        case_prefix = case_prefix.replace(
            "valid:usize)",
            "valid:usize,mut other:DisjointSlice<i32,CollectiveTiles<4>>)",
        )
        case_prefix = case_prefix.replace("pub fn probe(", "pub fn probe<'a>(").replace(
            "DisjointSlice<i32,", "DisjointSlice<'a,i32,"
        )
    if name == "warp_partial_geometry":
        case_prefix = case_prefix.replace("block=(64,1,1)", "block=(48,1,1)")
    elif name.endswith("_partial_block"):
        case_prefix = case_prefix.replace("block=(128,1,1)", "block=(33,1,1)")
    text = case_prefix + body + "\n}\nfn main() {}\n"
    source.write_text(text)
    (output / f"{name}.rs").write_text(text)
    run = subprocess.run(
        command,
        cwd=output,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output / f"{name}.log").write_text(run.stdout)
    passed = (
        run.returncode == 0
        if diagnostic is None
        else run.returncode != 0 and diagnostic in run.stdout
    )
    results.append(
        {"case": name, "expected": diagnostic, "exit": run.returncode, "pass": passed}
    )
    print(name, "PASS" if passed else "FAIL", flush=True)
(output / "results.json").write_text(
    json.dumps(
        {"command": command, "backend": env["CUDA_OXIDE_BACKEND"], "cases": results},
        indent=2,
    )
    + "\n"
)
raise SystemExit(0 if all(row["pass"] for row in results) else 1)
