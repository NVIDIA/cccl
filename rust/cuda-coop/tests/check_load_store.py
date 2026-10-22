# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile public collective participation, result and resource-epoch contrasts.

Use the qualified installed Oxide CLI and its matching automatic backend, or
supply an explicit --backend for a labelled local compiler candidate. The optional
Cargo config supports local source integration without changing manifests.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cargo-oxide", required=True, type=Path)
parser.add_argument(
    "--backend",
    type=Path,
    help="Explicit candidate backend; omitted uses the installed CLI cache",
)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cargo-config", type=Path)
parser.add_argument(
    "--cargo-lock", type=Path, help="Compatible lockfile for local source patches"
)
parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
parser.add_argument("--arch", default="sm_120")
parser.add_argument(
    "--case", action="append", help="Run only this named case; repeat to select several"
)
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
    warp_cycle_then_block=(warp_loop + load, None),
    warp_lane_cycle=(
        warp_loop.replace("i<q+1", "i<thread::threadIdx_x() as usize+1") + warp_load,
        "lane-varying",
    ),
    warp_helper_cycle=(
        "#[inline(never)] fn cycles(input:&[i32]) {let q=cuda_device::library::warp_group_index_v1::<32>() as usize;let mut values=coop::ThreadData::<_,4>::new();let warp=coop::this_warp();let valid=1;let offset=0;"
        + warp_loop
        + "}cycles(input);"
        + load,
        None,
    ),
    warp_partial_geometry=(warp_load, "complete physical warps"),
    warp_transpose=(
        warp_load.replace(
            ".oob_default(0)", ".oob_default(0).algorithm(coop::Transpose)"
        ),
        None,
    ),
    warp_explicit_storage=(
        new_storage
        + warp_load.replace(
            ".oob_default(0)", ".oob_default(0).temp_storage(&mut scratch)"
        ),
        "LoadStoreGroup<Direct, &mut TempStorage>",
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
    logical_cycle_to_wider=(logical_loop + load, None),
    logical_sibling_loop=(logical_loop + logical_load, None),
    logical_helper_cycle=(
        "#[inline(never)] fn cycle(input:&[i32]) {let warp=coop::this_warp().group_by::<8>();let q=warp.rank_at::<coop::Warp>() as usize;let valid=1;let offset=0;let mut values=coop::ThreadData::<_,4>::new();"
        + logical_loop
        + "}cycle(input);"
        + load,
        None,
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
# Finite sibling cycles reconverge structurally; value precision and memory
# completion remain independent requirements for a subsequent wider call.
cycle = logical_loop.replace("i<q+1", "i<q")
block_sync = "thread::sync_threads();"
write_cycle = (
    logical_load + "let mut i=0;while i<q+1 {" + logical_store + logical_sync + "i+=1;}"
)
cases.update(
    logical_cycle_wider_sequence=(
        cycle + logical_load.replace("(warp,", "(coop::this_warp(),") + load,
        None,
    ),
    logical_cycle_fresh_counter=(
        cycle + "let mut j=0;while j<valid {" + load + "j+=1;}",
        None,
    ),
    logical_cycle_tainted_count=(
        cycle + load.replace(".valid_items(valid)", ".valid_items(q+1)"),
        "not proved block-uniform",
    ),
    logical_cycle_carried_count=(
        cycle + load.replace(".valid_items(valid)", ".valid_items(i)"),
        "not proved block-uniform",
    ),
    logical_cycle_active_barrier=(
        logical_load + "let mut i=0;while i<q+1 {" + block_sync + "i+=1;}",
        "block barrier is reached through lane-varying",
    ),
    logical_cycle_continue=(
        "let mut i=0;while i<2 {i+=1;if q==0 {continue;}" + load + "}",
        "lane-varying",
    ),
    logical_cycle_nested_region=(
        "let mut i=0;while i<q+1 {let mut j=0;while j<2 {" + load + "j+=1;}i+=1;}",
        "lane-varying",
    ),
    logical_cycle_early_return=(
        cycle + "if q==0 {return;}" + load,
        "lane-varying",
    ),
    logical_cycle_write_no_completion=(
        write_cycle + reload,
        "requires a barrier covering both accesses",
    ),
    logical_cycle_write_completed=(write_cycle + block_sync + reload, None),
    logical_cycle_outer_break=(
        "let mut k=0;'outer: while k<2 {let mut i=0;while i<q {if valid==0 {break 'outer;}"
        + logical_load
        + "i+=1;}"
        + load
        + "k+=1;}",
        "lane-varying",
    ),
    logical_cycle_inner_break=(
        "let mut k=0;while k<2 {let mut i=0;while i<q {if valid==0 {break;}"
        + logical_load
        + "i+=1;}"
        + load
        + "k+=1;}",
        None,
    ),
    logical_cycle_lane_scalar=(
        load
        + "let limit=thread::threadIdx_x() as usize%8;let mut i=0;let mut acc=1_i32;"
        "while i<limit {acc=acc.wrapping_mul(3).wrapping_add(i as i32);i+=1;}"
        "values.write(0,acc);" + store,
        None,
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


# Replicated implicit Warp pools use actual provider extents and scoped completion.
logical_transpose_load = logical_load.replace(
    ".oob_default(0)", ".oob_default(0).algorithm(coop::Transpose)"
)
logical_transpose_store = logical_store.replace("&values", "&mut values").replace(
    ".valid_items(valid)", ".valid_items(valid).algorithm(coop::Transpose)"
)
logical_transpose_reload = logical_reload.replace(
    ".valid_items(valid)", ".valid_items(valid).algorithm(coop::Transpose)"
)
warp16 = "coop::this_warp().group_by::<16>()"
logical_fixed_view = "let prefix=unsafe {cuda_device::DynamicSharedArray::<[u64;8],32>::get_fixed()};let _=prefix;"
cases.update(
    logical_transpose_managed=(
        logical_transpose_load
        + logical_transpose_store
        + logical_transpose_store
        + logical_transpose_reload,
        None,
    ),
    logical_transpose_shared=(
        logical_transpose_load
        + logical_transpose_store.replace("&mut values", "&values"),
        "StoreValue<i32, Transpose>",
    ),
    logical_transpose_sibling_count=(
        logical_transpose_load.replace(".valid_items(valid)", ".valid_items(q)"),
        None,
    ),
    logical_transpose_lane_count=(
        logical_transpose_load.replace(
            ".valid_items(valid)", ".valid_items(thread::threadIdx_x() as usize)"
        ),
        "not proved Warp-uniform",
    ),
    logical_transpose_block_pool=(
        transpose_load
        + logical_transpose_load
        + logical_transpose_load
        + transpose_load,
        None,
    ),
    logical_transpose_containing_sync=(
        logical_transpose_load
        + logical_transpose_load.replace("(warp,", f"({warp16},")
        + logical_sync.replace("::<8>()", "::<16>()")
        + logical_transpose_load,
        None,
    ),
    logical_transpose_wider_clean=(
        logical_transpose_load
        + logical_transpose_store
        + logical_transpose_reload.replace("(warp,", f"({warp16},"),
        "requires a barrier covering both accesses",
    ),
    logical_transpose_wider_completed=(
        logical_transpose_load
        + logical_transpose_store
        + logical_sync.replace("::<8>()", "::<16>()")
        + logical_transpose_reload.replace("(warp,", f"({warp16},"),
        None,
    ),
    logical_transpose_block_completion=(
        logical_transpose_load + logical_transpose_store + reload,
        "requires a barrier covering both accesses",
    ),
    logical_transpose_live_reservation=(
        new_storage
        + reserve
        + logical_transpose_load
        + logical_sync
        + explicit_load
        + release,
        "overlaps a live reservation",
    ),
    logical_transpose_explicit_dirty=(
        new_storage
        + explicit_load
        + logical_transpose_load
        + logical_transpose_load
        + explicit_load,
        "requires a full-block barrier",
    ),
    logical_transpose_loop=(
        "let mut i=0;while i<2 {"
        + logical_transpose_load
        + logical_transpose_store
        + "i+=1;}"
        + logical_transpose_reload,
        None,
    ),
    logical_shared_getter_lookalike=(
        "struct DynamicSharedArray;impl DynamicSharedArray {#[inline(never)] fn get_fixed()->*mut u8 {if thread::threadIdx_x()&1==0 {thread::sync_threads();}core::ptr::null_mut()}}let _=DynamicSharedArray::get_fixed();"
        + logical_transpose_load,
        "block barrier is reached through lane-varying control flow",
    ),
    logical_transpose_fixed_prefix=(logical_fixed_view + logical_transpose_load, None),
    logical_transpose_raw_view=(
        logical_fixed_view.replace("get_fixed()", "get_raw()") + logical_transpose_load,
        "planned storage cannot include an unconstrained native dynamic shared view",
    ),
    logical_zero_raw_view=(
        logical_fixed_view.replace("get_fixed()", "get_raw()") + logical_load,
        "planned storage cannot include an unconstrained native dynamic shared view",
    ),
    block_zero_raw_numeric=(
        "let raw=unsafe {cuda_device::DynamicSharedArray::<[i32;128],16>::get_raw()};unsafe {raw.add(thread::threadIdx_x() as usize).write(1);}"
        + load,
        "planned storage cannot include an unconstrained native dynamic shared view",
    ),
    logical_zero_no_completion=(
        logical_load + logical_store + logical_reload,
        "requires a barrier covering both accesses",
    ),
)


# Scratch-free Warp choices preserve shared payloads without completion credit.
for selector in ("Striped", "Vectorize"):
    selected_load = logical_load.replace(
        ".oob_default(0)", f".oob_default(0).algorithm(coop::{selector})"
    )
    selected_store = logical_store.replace(
        ".valid_items(valid)", f".valid_items(valid).algorithm(coop::{selector})"
    )
    key = "logical_" + selector.lower()
    cases[key + "_shared"] = (selected_load + selected_store, None)
    cases[key + "_non_exhaustive"] = (
        selected_load.replace(
            "(warp,", "(coop::this_warp().group_by_non_exhaustive::<8>(),"
        )
        + selected_store.replace(
            "(warp,", "(coop::this_warp().group_by_non_exhaustive::<8>(),"
        ),
        None,
    )
    cases[key + "_physical"] = (
        selected_load.replace("(warp,", "(coop::this_warp(),")
        + selected_store.replace("(warp,", "(coop::this_warp(),"),
        None,
    )
    cases[key + "_missing_completion"] = (
        selected_load + selected_store + logical_reload,
        "requires a barrier covering both accesses",
    )
    cases[key + "_explicit_storage"] = (
        new_storage
        + selected_load.replace(
            ".oob_default(0)", ".oob_default(0).temp_storage(&mut scratch)"
        ),
        f"LoadStoreGroup<{selector}, &mut TempStorage>",
    )


# Scan returns fresh lane-private output while retaining the existing group and
# storage proofs. These cases exercise the public API, not private markers.
scan = "values=coop::scan(block,&values,Default::default());"
scan_manual = "values=coop::scan(block,&values,coop::ScanOptions::default().temp_storage(&mut scratch));"
scan_warp = "let warp=coop::this_warp().group_by::<8>();"
cases.update(
    scan_defaults=(
        load
        + scan
        + "values=coop::exclusive_sum(block,&values,Default::default());"
        + "values=coop::inclusive_sum(block,&values,Default::default());"
        + "values=coop::exclusive_scan(block,&values,Default::default());"
        + "values=coop::inclusive_scan(block,&values,Default::default());"
        + store,
        None,
    ),
    scan_generic_seed=(
        load + "fn seeded<T:coop::Numeric,Op:coop::ScanOperator<T>,const N:usize>("
        "group:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N>,seed:T,op:Op)"
        "->coop::ThreadData<T,N> where ():coop::ScanMode<T,Op,T> {"
        "coop::exclusive_scan(group,input,coop::ScanOptions::default().scan_op(op).initial_value(seed))}"
        "values=seeded(block,&values,valid as i32,coop::Min);" + store,
        None,
    ),
    scan_varying_seed=(
        load
        + "values=coop::exclusive_scan(block,&values,coop::ScanOptions::default().initial_value(thread::threadIdx_x() as i32));",
        "not proved block-uniform",
    ),
    scan_varying_participation=(
        load + "if thread::threadIdx_x()&1==0 {" + scan + "}",
        "lane-varying",
    ),
    scan_warp_scalar=(
        load + scan_warp + "let seed=warp.rank_at::<coop::Warp>() as i32;"
        "let result=coop::exclusive_scan(warp,thread::threadIdx_x() as i32,coop::ScanOptions::default().initial_value(seed));"
        "values.write(0,result);",
        None,
    ),
    scan_warp_payload=(
        load + scan_warp + "let _result=coop::scan(warp,&values,Default::default());",
        "ScanValue",
    ),
    scan_warp_algorithm=(
        load
        + scan_warp
        + "let _result=coop::scan(warp,1_i32,coop::ScanOptions::default().algorithm(coop::Raking));",
        "expected `()`, found `Raking`",
    ),
    scan_warp_storage=(
        load
        + scan_warp
        + new_storage
        + "let _result=coop::scan(warp,1_i32,coop::ScanOptions::default().temp_storage(&mut scratch));",
        "expected struct `ScanOptions<i32, _, _, _, _, ()>`",
    ),
    scan_missing_seed=(
        load
        + "let _result=coop::scan(block,&values,coop::ScanOptions::default().scan_op(coop::Min));",
        "ScanMode",
    ),
    scan_inclusive_seed=(
        load
        + "let _result=coop::scan(block,&values,coop::ScanOptions::default().mode(coop::Inclusive).initial_value(0));",
        "ScanMode",
    ),
    scan_seed_dtype=(
        load
        + "let _result=coop::exclusive_scan(block,&values,coop::ScanOptions::default().initial_value(0_i16));",
        "mismatched types",
    ),
    scan_float_bitwise=(
        load
        + "let _result=coop::inclusive_scan(block,1_f32,coop::ScanOptions::default().scan_op(coop::BitAnd));",
        "ScanOperator<f32>",
    ),
    scan_storage_reuse=(
        load + new_storage + scan_manual + sync + scan_manual + store,
        None,
    ),
    scan_storage_missing_barrier=(
        load + new_storage + scan_manual + scan_manual,
        "requires a full-block barrier",
    ),
    scan_no_global_completion=(
        load + store + scan + reload,
        "requires a barrier covering both accesses",
    ),
    scan_storage_auto_reuse=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + scan_manual
        + scan_manual
        + store,
        None,
    ),
)

# A cold hint preserves participation and outstanding memory-access facts.
cases.update(
    cold_path_before_collective=("core::hint::cold_path();" + load + store, None),
    cold_path_divergent_collective=(
        "if thread::threadIdx_x()&1==0 {core::hint::cold_path();" + load + "}",
        "lane-varying",
    ),
    cold_path_no_completion=(
        load + store + "core::hint::cold_path();" + reload,
        "requires a barrier covering both accesses",
    ),
    cold_path_user_function=(
        "fn cold_path() {thread::sync_threads();}"
        "if thread::threadIdx_x()&1==0 {cold_path();}" + load,
        "block barrier is reached through lane-varying control flow",
    ),
)

# Reduce returns a guarded scalar only at the group root. That local result
# does not establish collective participation or complete global accesses.
reduce_call = "let _result=coop::sum(block,&values,Default::default());"
reduce_manual = "let _result=coop::sum(block,&values,coop::ReduceOptions::default().temp_storage(&mut scratch));"
reduce_prefix = "let group=coop::this_warp().group_by_non_exhaustive::<17>();"
cases.update(
    reduce_defaults=(
        load
        + reduce_call
        + "let _result:Option<i32>=coop::reduce(block,&mut values,Default::default());"
        + "let _result=coop::sum(block,1_i32,coop::ReduceOptions::default().valid_items(valid));"
        + "let _result=coop::sum(coop::this_warp(),&values,Default::default());"
        + "let _result=coop::sum(coop::this_warp().group_by::<8>(),&values,Default::default());",
        None,
    ),
    reduce_generic_operator=(
        load + "fn aggregate<T:coop::Numeric,Op:coop::ReduceOperator<T>,const N:usize>("
        "group:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N>,op:Op)"
        "->Option<T> {coop::reduce(group,input,coop::ReduceOptions::default().binary_op(op))}"
        "let _result=aggregate(block,&values,coop::Min);",
        None,
    ),
    reduce_payload_count=(
        load
        + "let _result=coop::sum(block,&values,coop::ReduceOptions::default().valid_items(valid));",
        "expected struct `ReduceOptions<i32, _, _, (), _>`",
    ),
    reduce_one_item_payload_count=(
        "values.write(0,0_i32);let mut one=coop::ThreadData::<_,1>::new();one.write(0,1_i32);"
        "let _result=coop::sum(block,&one,coop::ReduceOptions::default().valid_items(valid));",
        "expected struct `ReduceOptions<i32, _, _, (), _>`",
    ),
    reduce_float_bitwise=(
        "values.write(0,0_i32);let _result=coop::reduce(block,1_f32,coop::ReduceOptions::default().binary_op(coop::BitAnd));",
        "ReduceOperator<f32>",
    ),
    reduce_warp_algorithm=(
        "values.write(0,0_i32);let _result=coop::sum(coop::this_warp(),1_i32,coop::ReduceOptions::default().algorithm(coop::Raking));",
        "expected `()`, found `Raking`",
    ),
    reduce_warp_storage=(
        "values.write(0,0_i32);"
        + new_storage
        + "let _result=coop::sum(coop::this_warp(),1_i32,coop::ReduceOptions::default().temp_storage(&mut scratch));",
        "expected struct `ReduceOptions<i32, _, _, _, ()>`",
    ),
    reduce_thread_group=(
        "values.write(0,0_i32);let _result=coop::sum(coop::this_thread(),1_i32,Default::default());",
        "ReduceGroup",
    ),
    reduce_varying_count=(
        "values.write(0,0_i32);let count=thread::threadIdx_x() as usize+1;"
        "let _result=coop::sum(block,1_i32,coop::ReduceOptions::default().valid_items(count));",
        "not proved block-uniform",
    ),
    reduce_varying_participation=(
        load + "if thread::threadIdx_x()&1==0 {" + reduce_call + "}",
        "lane-varying",
    ),
    reduce_warp_uniform_count=(
        "values.write(0,0_i32);let warp=coop::this_warp();let count=warp.rank_at::<coop::Block>() as usize+1;"
        "let _result=coop::sum(warp,1_i32,coop::ReduceOptions::default().valid_items(count));",
        None,
    ),
    reduce_warp_varying_count=(
        "values.write(0,0_i32);let count=(thread::threadIdx_x()&31) as usize+1;"
        "let _result=coop::sum(coop::this_warp(),1_i32,coop::ReduceOptions::default().valid_items(count));",
        "not proved Warp-uniform",
    ),
    reduce_prefix_member=(
        load + reduce_prefix + "if group.is_member()!=0 {"
        "let _result=coop::sum(group,&values,Default::default());}",
        None,
    ),
    reduce_prefix_missing_guard=(
        load
        + reduce_prefix
        + "let _result=coop::sum(group,&values,Default::default());",
        "lane-varying",
    ),
    reduce_prefix_arithmetic_guard=(
        load + reduce_prefix + "if thread::threadIdx_x()%32<17 {"
        "let _result=coop::sum(group,&values,Default::default());}",
        "lane-varying",
    ),
    reduce_prefix_varying_count=(
        "values.write(0,0_i32);" + reduce_prefix + "if group.is_member()!=0 {"
        "let count=(thread::threadIdx_x()&31) as usize+1;"
        "let _result=coop::sum(group,1_i32,coop::ReduceOptions::default().valid_items(count));}",
        "not proved Warp-uniform",
    ),
    reduce_result_is_not_participation=(
        load + "if let Some(total)=coop::sum(block,&values,Default::default()) {"
        "let _result=coop::sum(block,total,Default::default());}",
        "lane-varying",
    ),
    reduce_storage_auto_reuse=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + reduce_manual
        + reduce_manual,
        None,
    ),
    reduce_storage_reuse=(
        load + new_storage + reduce_manual + sync + reduce_manual,
        None,
    ),
    reduce_storage_missing_barrier=(
        load + new_storage + reduce_manual + reduce_manual,
        "requires a full-block barrier",
    ),
    reduce_no_global_completion=(
        load + store + reduce_call + reload,
        "requires a barrier covering both accesses",
    ),
)

# Batched reductions return a calculated partial payload. Result validity is
# lane-local and does not replace participation or global-access completion.
batched_group = "let group=coop::this_warp().group_by::<8>();"
batched_call = "let _result=coop::reduce_batched(group,&values,Default::default());"
cases.update(
    batched_generic_defaults=(
        load + "fn forward<T:coop::Numeric,G:coop::ReduceBatchedGroup,const B:usize>("
        "group:coop::ThreadGroup<G>,input:&coop::ThreadData<T,B>)"
        "->coop::ThreadData<T,{coop::batched_extent(B,G::WIDTH)}> "
        "where [();coop::batched_extent(B,G::WIDTH)]: {"
        "coop::reduce_batched(group,input,Default::default())}"
        "fn selected<T:coop::Numeric,G:coop::ReduceBatchedGroup,Op:coop::ReduceOperator<T>,const B:usize>("
        "group:coop::ThreadGroup<G>,input:&coop::ThreadData<T,B>,op:Op)"
        "->coop::ThreadData<T,{coop::batched_extent(B,G::WIDTH)}> "
        "where [();coop::batched_extent(B,G::WIDTH)]: {"
        "coop::reduce_batched(group,input,coop::ReduceBatchedOptions::default().binary_op(op).output_layout(coop::Blocked))}"
        "let _result=forward(coop::this_warp().group_by::<8>(),&mut values);"
        "let _result=selected(coop::this_warp(),&values,coop::Min);",
        None,
    ),
    batched_scalar=(
        load
        + "let _result=coop::reduce_batched(coop::this_warp(),&1_i32,Default::default());",
        "mismatched types",
    ),
    batched_block=(
        load + "let _result=coop::reduce_batched(block,&values,Default::default());",
        "ReduceBatchedGroup",
    ),
    batched_prefix17=(
        load + "let group=coop::this_warp().group_by_non_exhaustive::<17>();"
        "if group.is_member()!=0 {" + batched_call + "}",
        "ReduceBatchedGroup",
    ),
    batched_lane_divergence=(
        load + batched_group + "if thread::threadIdx_x()&1==0 {" + batched_call + "}",
        "lane-varying",
    ),
    batched_read_epoch=(
        reload + batched_group + batched_call + store,
        "requires a barrier covering both accesses",
    ),
    batched_read_completed=(
        reload + batched_group + batched_call + sync + store,
        None,
    ),
    batched_result_is_not_participation=(
        load
        + batched_group
        + "let result=coop::reduce_batched(group,&values,Default::default());"
        "if result.get(0).is_some() {" + batched_call + "}",
        "lane-varying",
    ),
    batched_count=(
        load + "let _result=coop::reduce_batched(coop::this_warp(),&values,"
        "coop::ReduceBatchedOptions::default().valid_items(valid));",
        "no method named `valid_items`",
    ),
    batched_storage=(
        load
        + new_storage
        + "let _result=coop::reduce_batched(coop::this_warp(),&values,"
        "coop::ReduceBatchedOptions::default().temp_storage(&mut scratch));",
        "no method named `temp_storage`",
    ),
    batched_algorithm=(
        load + "let _result=coop::reduce_batched(coop::this_warp(),&values,"
        "coop::ReduceBatchedOptions::default().algorithm(coop::Raking));",
        "no method named `algorithm`",
    ),
    batched_float_bitwise=(
        "values.write(0,0_i32);let mut floats=coop::ThreadData::<_,4>::new();"
        "floats.write(0,1_f32);let _result=coop::reduce_batched(coop::this_warp(),&floats,"
        "coop::ReduceBatchedOptions::default().binary_op(coop::BitAnd));",
        "ReduceOperator<f32>",
    ),
)

# Exchange preserves extent/alignment and input ownership. Fresh output does not
# establish participation or complete an unrelated global-memory access.
exchange = "values=coop::exchange(block,&values,Default::default());"
exchange_warp = "let warp=coop::this_warp().group_by_non_exhaustive::<8>();"
cases.update(
    exchange_generic_defaults=(
        load
        + "fn forward<T:coop::Numeric,G:coop::ExchangeGroup,const N:usize,const A:usize>("
        "group:coop::ThreadGroup<G>,input:&coop::ThreadData<T,N,A>)->coop::ThreadData<T,N,A> "
        "where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::exchange(group,input,Default::default())}"
        "fn selected<T:coop::Numeric,G:coop::ExchangeGroup,M:coop::ExchangeMode,const N:usize,const A:usize>("
        "group:coop::ThreadGroup<G>,input:&coop::ThreadData<T,N,A>,options:coop::ExchangeOptions<M>)"
        "->coop::ThreadData<T,N,A> where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::exchange(group,input,options)}"
        "values=forward(block,&values);"
        "values=selected(block,&mut values,coop::ExchangeOptions::default());"
        "let mut one=coop::ThreadData::<_,1>::new();one.write(0,7_i32);"
        "let _one=forward(block,&mut one);"
        "let mut aligned=coop::ThreadData::<_,4,16>::new();"
        "aligned.write(0,0_i32);aligned.write(1,1);aligned.write(2,2);aligned.write(3,3);"
        "let aligned_result:coop::ThreadData<_,4,16>=selected(block,&mut aligned,"
        "coop::ExchangeOptions::default().mode(coop::BlockedToStriped));"
        + store.replace("&values", "&aligned_result"),
        None,
    ),
    exchange_warp_siblings=(
        load
        + "values=coop::exchange(coop::this_warp(),&values,Default::default());"
        + exchange_warp
        + "if warp.rank_at::<coop::Warp>()&1==0 {"
        "values=coop::exchange(warp,&values,Default::default());"
        "values=coop::exchange(warp,&values,coop::ExchangeOptions::default().mode(coop::BlockedToStriped));}"
        + store,
        None,
    ),
    exchange_block_partial_block=(load + exchange + store, None),
    exchange_block_partition=(
        load
        + "let _result=coop::exchange(block.group_by::<2>(),&values,Default::default());",
        "ExchangeGroup",
    ),
    exchange_prefix17=(
        load + "let group=coop::this_warp().group_by_non_exhaustive::<17>();"
        "if group.is_member()!=0 {let _result=coop::exchange(group,&values,Default::default());}",
        "ExchangeGroup",
    ),
    exchange_bad_mode=(
        load + "struct StripedToBlocked;"
        "let _result=coop::exchange(block,&values,coop::ExchangeOptions::default().mode(StripedToBlocked));",
        "ExchangeMode",
    ),
    exchange_block_lane_divergence=(
        load + "if thread::threadIdx_x()&1==0 {" + exchange + "}",
        "lane-varying",
    ),
    exchange_warp_lane_divergence=(
        load
        + exchange_warp
        + "if thread::threadIdx_x()&1==0 {let _result=coop::exchange(warp,&values,Default::default());}",
        "lane-varying",
    ),
    exchange_warp_partial_block=(
        load
        + "let _result=coop::exchange(coop::this_warp(),&values,Default::default());",
        "complete physical warps",
    ),
    exchange_result_control=(
        load + "let result=coop::exchange(block,&values,Default::default());"
        "if result.get(0)==Some(0) {" + exchange + "}",
        "lane-varying",
    ),
    exchange_read_epoch=(
        reload + exchange + store,
        "requires a barrier covering both accesses",
    ),
    exchange_read_completed=(reload + exchange + sync + store, None),
)

# Shuffle preserves its input but returns one invalid edge. Neither result
# values nor validity prove participation or unrelated global-memory completion.
shuffle = "let result=coop::shuffle(block,&values,Default::default());"
cases.update(
    shuffle_generic_defaults=(
        load + "fn forward<T:coop::Numeric,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>)->coop::ThreadData<T,N,A> "
        "where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::shuffle(block,input,Default::default())}"
        "fn selected<T:coop::Numeric,M:coop::ShuffleMode,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>,options:coop::ShuffleOptions<M>)"
        "->coop::ThreadData<T,N,A> where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::shuffle(block,input,options)}"
        "let default_result=forward(block,&values);"
        "let explicit_result=selected(block,&mut values,coop::ShuffleOptions::default().distance::<1>().mode(coop::Down));"
        "let mut one=coop::ThreadData::<_,1>::new();one.write(0,7_i32);"
        "let one_result=forward(block,&mut one);"
        "let mut aligned=coop::ThreadData::<_,4,16>::new();"
        "aligned.write(0,0_i32);aligned.write(1,1);aligned.write(2,2);aligned.write(3,3);"
        "let mut aligned_result:coop::ThreadData<_,4,16>=selected(block,&mut aligned,"
        "coop::ShuffleOptions::default().mode(coop::Up).distance::<1>());"
        "aligned_result.write(0,default_result.get(0).unwrap_or(0));"
        "aligned_result.write(1,explicit_result.get(0).unwrap_or(0));"
        "aligned_result.write(2,one_result.get(0).unwrap_or(0));"
        + store.replace("&values", "&aligned_result"),
        None,
    ),
    shuffle_block_partial_block=(
        load
        + shuffle
        + "let mut repaired=result;if repaired.get(3).is_none(){repaired.write(3,0);}"
        + store.replace("&values", "&repaired"),
        None,
    ),
    shuffle_distance_zero=(
        load
        + "let result=coop::shuffle(block,&values,coop::ShuffleOptions::default().distance::<0>());"
        + store,
        "Shuffle supports only distance one",
    ),
    shuffle_distance_two=(
        load
        + "let result=coop::shuffle(block,&values,coop::ShuffleOptions::default().distance::<2>());"
        + store,
        "Shuffle supports only distance one",
    ),
    shuffle_warp_group=(
        load
        + "let result=coop::shuffle(coop::this_warp(),&values,Default::default());",
        "mismatched types",
    ),
    shuffle_block_partition=(
        load
        + "let result=coop::shuffle(block.group_by::<2>(),&values,Default::default());",
        "mismatched types",
    ),
    shuffle_scalar=(
        load + "let result=coop::shuffle(block,&3_i32,Default::default());",
        "mismatched types",
    ),
    shuffle_bad_mode=(
        load + "struct Up;"
        "let result=coop::shuffle(block,&values,coop::ShuffleOptions::default().mode(Up));",
        "ShuffleMode",
    ),
    shuffle_lane_divergence=(
        load + "if thread::threadIdx_x()&1==0 {" + shuffle + "}",
        "lane-varying",
    ),
    shuffle_result_control=(
        load
        + "let first=coop::shuffle(block,&values,coop::ShuffleOptions::default().mode(coop::Up));"
        "if first.get(0).is_some() {" + shuffle + "}",
        "lane-varying",
    ),
    shuffle_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + shuffle
        + store,
        "requires a barrier covering both accesses",
    ),
    shuffle_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + shuffle
        + sync
        + store,
        None,
    ),
)

# Adjacent Difference keeps counted suffixes initialized and requires uniform
# neighbor/count controls. Explicit scratch retains its existing completion rules.
adjacent = "let result=coop::adjacent_difference(block,&values,Default::default());"
adjacent_manual = "values=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().temp_storage(&mut scratch));"
cases.update(
    adjacent_generic_defaults=(
        load + "fn forward<T:coop::Numeric,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>)->coop::ThreadData<T,N,A> "
        "where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::adjacent_difference(block,input,Default::default())}"
        "let default_result=forward(block,&values);"
        "let mut one=coop::ThreadData::<_,1>::new();one.write(0,7_i32);"
        "let one_result=forward(block,&mut one);"
        "let mut aligned=coop::ThreadData::<_,4,16>::new();"
        "aligned.write(0,0_i32);aligned.write(1,1);aligned.write(2,2);aligned.write(3,3);"
        "let mut aligned_result:coop::ThreadData<_,4,16>=forward(block,&mut aligned);"
        "aligned_result.write(0,default_result[0]^one_result[0]);"
        + store.replace("&values", "&aligned_result"),
        None,
    ),
    adjacent_recipes_partial_block=(
        load + "let a=coop::adjacent_difference(block,&values,Default::default());"
        "let b=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().tile_predecessor_item(7));"
        "let c=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().valid_items(valid));"
        "let d=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().tile_predecessor_item(7).valid_items(0));"
        "let e=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().direction(coop::Right));"
        "let f=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().tile_successor_item(7).direction(coop::Right));"
        "let g=coop::adjacent_difference(block,&values,coop::AdjacentDifferenceOptions::default().valid_items(valid).direction(coop::Right));"
        "values.write(0,a[0]^b[0]^c[0]^d[0]^e[0]^f[0]^g[0]);" + store,
        None,
    ),
    adjacent_right_count_successor=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().direction(coop::Right).valid_items(512).tile_successor_item(0));",
        "AdjacentDifferenceDirection",
    ),
    adjacent_predecessor_then_right=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().tile_predecessor_item(0).direction(coop::Right));",
        "AdjacentDifferenceDirection",
    ),
    adjacent_left_successor=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().tile_successor_item(0));",
        "AdjacentDifferenceDirection",
    ),
    adjacent_neighbor_type=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().tile_predecessor_item(0_i64));",
        "mismatched types",
    ),
    adjacent_scalar=(
        load + "let result=coop::adjacent_difference(block,&3_i32,Default::default());",
        "mismatched types",
    ),
    adjacent_warp=(
        load
        + "let result=coop::adjacent_difference(coop::this_warp(),&values,Default::default());",
        "mismatched types",
    ),
    adjacent_partition=(
        load
        + "let result=coop::adjacent_difference(block.group_by::<2>(),&values,Default::default());",
        "mismatched types",
    ),
    adjacent_varying_count=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().valid_items(thread::threadIdx_x() as usize));",
        "not proved block-uniform",
    ),
    adjacent_varying_predecessor=(
        load + "let result=coop::adjacent_difference(block,&values,"
        "coop::AdjacentDifferenceOptions::default().tile_predecessor_item(thread::threadIdx_x() as i32));",
        "not proved block-uniform",
    ),
    adjacent_lane_divergence=(
        load + "if thread::threadIdx_x()&1==0 {" + adjacent + "}",
        "lane-varying",
    ),
    adjacent_result_control=(
        load + adjacent + "if result[0]==0 {" + adjacent + "}",
        "lane-varying",
    ),
    adjacent_storage_missing_barrier=(
        load + new_storage + adjacent_manual + adjacent_manual,
        "requires a full-block barrier",
    ),
    adjacent_storage_completed=(
        load + new_storage + adjacent_manual + sync + adjacent_manual + store,
        None,
    ),
    adjacent_storage_auto=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + adjacent_manual
        + adjacent_manual
        + store,
        None,
    ),
    adjacent_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + adjacent
        + store,
        "requires a barrier covering both accesses",
    ),
    adjacent_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + adjacent
        + sync
        + store,
        None,
    ),
)

# Discontinuity preserves input and supplies one or two independent i32 results.
# Scratch completion and unrelated global-memory completion remain separate.
discontinuity = "let result=coop::discontinuity(block,&values,Default::default());"
discontinuity_pair = "let (_heads,_tails)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails));"
discontinuity_manual = "let (heads,_tails)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails).temp_storage(&mut scratch));values=heads;"
cases.update(
    discontinuity_generic_defaults=(
        load + "fn forward<T:coop::Numeric,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>)->coop::ThreadData<i32,N,A> "
        "where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::discontinuity(block,input,Default::default())}"
        "fn project<T:coop::Numeric,M,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>,mode:M)->M::Output "
        "where coop::Alignment<A>:coop::SupportedAlignment,M:coop::DiscontinuityMode<T,(),(),N,A> {"
        "coop::discontinuity(block,input,coop::DiscontinuityOptions::default().mode(mode))}"
        "fn pair<T:coop::Numeric,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>)->(coop::ThreadData<i32,N,A>,coop::ThreadData<i32,N,A>) "
        "where coop::Alignment<A>:coop::SupportedAlignment {project(block,input,coop::HeadsAndTails)}"
        "let default_result=forward(block,&values);"
        "let mut one=coop::ThreadData::<_,1>::new();one.write(0,7_u8);"
        "let one_result=forward(block,&mut one);"
        "let mut aligned=coop::ThreadData::<_,4,16>::new();"
        "aligned.write(0,0_f64);aligned.write(1,1.0);aligned.write(2,1.0);aligned.write(3,2.0);"
        "let (mut heads,tails):(coop::ThreadData<_,4,16>,coop::ThreadData<_,4,16>)=pair(block,&mut aligned);"
        "heads.write(0,default_result[0]^one_result[0]^tails[0]);"
        + store.replace("&values", "&heads"),
        None,
    ),
    discontinuity_recipes_partial_block=(
        load + "let a=coop::discontinuity(block,&values,Default::default());"
        "let b=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().tile_predecessor_item(7));"
        "let c=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().mode(coop::Tails));"
        "let d=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().tile_successor_item(7).mode(coop::Tails));"
        "let (eh,et)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails));"
        "let (fh,ft)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails).tile_predecessor_item(7));"
        "let (gh,gt)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().tile_successor_item(7).mode(coop::HeadsAndTails));"
        "let (hh,ht)=coop::discontinuity(block,&values,coop::DiscontinuityOptions::default().tile_successor_item(7).tile_predecessor_item(7).mode(coop::HeadsAndTails));"
        "values.write(0,a[0]^b[0]^c[0]^d[0]^eh[0]^et[0]^fh[0]^ft[0]^gh[0]^gt[0]^hh[0]^ht[0]);"
        + store,
        None,
    ),
    discontinuity_heads_successor=(
        load + "let result=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().tile_successor_item(0));",
        "DiscontinuityMode",
    ),
    discontinuity_predecessor_then_tails=(
        load + "let result=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().tile_predecessor_item(0).mode(coop::Tails));",
        "DiscontinuityMode",
    ),
    discontinuity_neighbor_type=(
        load + "let result=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().tile_predecessor_item(0_i64));",
        "mismatched types",
    ),
    discontinuity_no_count=(
        load + "let result=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().valid_items(valid));",
        "no method named `valid_items`",
    ),
    discontinuity_scalar=(
        load + "let result=coop::discontinuity(block,&3_i32,Default::default());",
        "mismatched types",
    ),
    discontinuity_warp=(
        load
        + "let result=coop::discontinuity(coop::this_warp(),&values,Default::default());",
        "mismatched types",
    ),
    discontinuity_partition=(
        load
        + "let result=coop::discontinuity(block.group_by::<2>(),&values,Default::default());",
        "mismatched types",
    ),
    discontinuity_varying_predecessor=(
        load + "let result=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().tile_predecessor_item(thread::threadIdx_x() as i32));",
        "not proved block-uniform",
    ),
    discontinuity_varying_successor_pair=(
        load + "let (_heads,_tails)=coop::discontinuity(block,&values,"
        "coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails).tile_successor_item(thread::threadIdx_x() as i32));",
        "not proved block-uniform",
    ),
    discontinuity_lane_divergence=(
        load + "if thread::threadIdx_x()&1==0 {" + discontinuity + "}",
        "lane-varying",
    ),
    discontinuity_result_control=(
        load + discontinuity + "if result[0]==0 {" + discontinuity + "}",
        "lane-varying",
    ),
    discontinuity_storage_missing_barrier=(
        load + new_storage + discontinuity_manual + discontinuity_manual,
        "requires a full-block barrier",
    ),
    discontinuity_storage_completed=(
        load + new_storage + discontinuity_manual + sync + discontinuity_manual + store,
        None,
    ),
    discontinuity_storage_auto=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + discontinuity_manual
        + discontinuity_manual
        + store,
        None,
    ),
    discontinuity_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + discontinuity_pair
        + store,
        "requires a barrier covering both accesses",
    ),
    discontinuity_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + discontinuity_pair
        + sync
        + store,
        None,
    ),
)

# Histogram has independent input T/N/ALIGN and counter C/K, with default Atomic.
# Internal counter readiness does not complete unrelated global-memory accesses.
histogram = "let result=coop::histogram(block,&values,coop::HistogramOptions::<129>::default().counter_type::<i64>().bins_per_thread::<2>());"
histogram_manual = histogram.replace(
    ".bins_per_thread::<2>()", ".bins_per_thread::<2>().temp_storage(&mut scratch)"
)
cases.update(
    histogram_generic_defaults=(
        load + "fn forward<T:coop::HistogramSample,C:coop::HistogramCounter,"
        "const N:usize,const A:usize,const BINS:usize,const K:usize>("
        "block:coop::ThreadGroup<coop::Block>,input:&coop::ThreadData<T,N,A>,"
        "options:coop::HistogramOptions<BINS,C,K>)->coop::ThreadData<C,K> "
        "where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::histogram(block,input,options)}"
        "let mut aligned=coop::ThreadData::<_,4,16>::new();"
        "aligned.write(0,values[0]);aligned.write(1,values[1]);"
        "aligned.write(2,values[2]);aligned.write(3,values[3]);"
        "let result:coop::ThreadData<i64,2>=forward(block,&mut aligned,"
        "coop::HistogramOptions::<129>::default().counter_type::<i64>().bins_per_thread::<2>());"
        "values.write(0,result[0] as i32);" + store,
        None,
    ),
    histogram_storage_completed=(
        load + new_storage + histogram_manual + sync + histogram_manual + store,
        None,
    ),
    histogram_storage_auto=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + histogram_manual
        + histogram_manual
        + store,
        None,
    ),
    histogram_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + histogram
        + sync
        + store,
        None,
    ),
    histogram_insufficient_capacity=(
        load
        + "let result=coop::histogram(block,&values,coop::HistogramOptions::<129>::default());",
        "Histogram output capacity is smaller than its bin count",
    ),
    histogram_multidimensional=(
        load + histogram,
        "Histogram requires a 1D Block with 1..=1024 threads",
    ),
    histogram_warp=(
        load + histogram.replace("histogram(block,", "histogram(coop::this_warp(),"),
        "mismatched types",
    ),
    histogram_zero_bins=(
        load + histogram.replace("::<129>", "::<0>"),
        "Histogram requires positive bins <= i32::MAX",
    ),
    histogram_zero_output=(
        load + histogram.replace("bins_per_thread::<2>", "bins_per_thread::<0>"),
        "ThreadData requires at least one item",
    ),
    histogram_bins_encoder_bound=(
        load + histogram.replace("::<129>", "::<2147483648>"),
        "Histogram requires positive bins <= i32::MAX",
    ),
    histogram_lane_divergence=(
        load + "if thread::threadIdx_x()&1==0 {" + histogram + "}",
        "lane-varying",
    ),
    histogram_result_control=(
        load + histogram + "if result[0]==0 {" + histogram + "}",
        "lane-varying",
    ),
    histogram_storage_missing_barrier=(
        load + new_storage + histogram_manual + histogram_manual,
        "requires a full-block barrier",
    ),
    histogram_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + histogram
        + store,
        "requires a barrier covering both accesses",
    ),
)

# Radix Sort keys retain the input, scalar uniformity and storage policy.
radix_sort = "let result=coop::radix_sort_keys(block,&values,Default::default());"
radix_manual = radix_sort.replace(
    "Default::default()", "coop::RadixSortOptions::default().temp_storage(&mut scratch)"
)
cases.update(
    radix_sort_generic_bounds=(
        load + "fn forward<T:coop::RadixKey,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,keys:&coop::ThreadData<T,N,A>,end:i64)"
        "->coop::ThreadData<T,N,A> where coop::Alignment<A>:coop::SupportedAlignment {"
        "coop::radix_sort_keys(block,keys,coop::RadixSortOptions::default()"
        ".begin_bit_const::<8>().end_bit(end).order(coop::Descending))}"
        "let result=forward(block,&mut values,valid as i64);"
        + store.replace("&values", "&result"),
        None,
    ),
    radix_sort_storage_completed=(
        load + new_storage + radix_manual + sync + radix_manual + store,
        None,
    ),
    radix_sort_storage_auto=(
        load
        + new_storage.replace("new()", "new().auto_sync()")
        + radix_manual
        + radix_manual
        + store,
        None,
    ),
    radix_sort_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + radix_sort
        + sync
        + store,
        None,
    ),
    radix_sort_varying_begin=(
        load
        + radix_sort.replace(
            "Default::default()",
            "coop::RadixSortOptions::default().begin_bit(thread::threadIdx_x() as i64)",
        ),
        "not proved block-uniform",
    ),
    radix_sort_varying_end=(
        load
        + radix_sort.replace(
            "Default::default()",
            "coop::RadixSortOptions::default().end_bit(thread::threadIdx_x() as i64 + 1)",
        ),
        "not proved block-uniform",
    ),
    radix_sort_result_control=(
        load + radix_sort + "if result[0]==0 {" + radix_sort + "}",
        "lane-varying",
    ),
    radix_sort_storage_missing_barrier=(
        load + new_storage + radix_manual + radix_manual,
        "requires a full-block barrier",
    ),
    radix_sort_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + radix_sort
        + store,
        "requires a barrier covering both accesses",
    ),
    radix_sort_tile_capacity=(
        "values.write(0,0_i32);let keys=coop::ThreadData::<i32,512>::new();"
        "let result=coop::radix_sort_keys(block,&keys,Default::default());",
        "at most 65535 keys",
    ),
    radix_sort_warp=(
        load + radix_sort.replace("keys(block,", "keys(coop::this_warp(),"),
        "mismatched types",
    ),
)

# Paired Radix Sort keeps independent value types and both returned payloads.
pair_values = (
    "let mut associated=coop::ThreadData::<f64,4,32>::new();"
    "for item in 0..4 { associated.write(item,item as f64+0.25); }"
)
pair_sort = "let (result,pair_result)=coop::radix_sort_pairs(block,&values,&associated,Default::default());"
pair_manual = pair_sort.replace(
    "Default::default()", "coop::RadixSortOptions::default().temp_storage(&mut scratch)"
)
cases.update(
    radix_pairs_generic_defaults=(
        load
        + pair_values
        + "let mut keys=coop::ThreadData::<i32,4,16>::new();"
        + load.replace("&mut values", "&mut keys")
        + "fn forward<K:coop::RadixKey,V:coop::RadixValue,const N:usize,const KA:usize,const VA:usize>("
        "block:coop::ThreadGroup<coop::Block>,keys:&coop::ThreadData<K,N,KA>,"
        "values:&coop::ThreadData<V,N,VA>,end:i64)"
        "->(coop::ThreadData<K,N,KA>,coop::ThreadData<V,N,VA>) "
        "where coop::Alignment<KA>:coop::SupportedAlignment,coop::Alignment<VA>:coop::SupportedAlignment {"
        "let _=coop::radix_sort_pairs(block,keys,values,Default::default());"
        "coop::radix_sort_pairs(block,keys,values,coop::RadixSortOptions::default()"
        ".begin_bit_const::<8>().end_bit(end).order(coop::Descending))}"
        "let (result,pair_result)=forward(block,&mut keys,&mut associated,valid as i64);"
        + store.replace("&values", "&result")
        + store.replace("&mut output", "&mut value_output").replace(
            "&values", "&pair_result"
        ),
        None,
    ),
    radix_pairs_storage_completed=(
        load + pair_values + new_storage + pair_manual + sync + pair_manual + store,
        None,
    ),
    radix_pairs_storage_auto=(
        load
        + pair_values
        + new_storage.replace("new()", "new().auto_sync()")
        + pair_manual
        + pair_manual
        + store,
        None,
    ),
    radix_pairs_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + pair_values
        + pair_sort
        + sync
        + store,
        None,
    ),
    radix_pairs_unequal_extent=(
        load
        + pair_values.replace("f64,4,32", "f64,3,32").replace("0..4", "0..3")
        + pair_sort,
        "mismatched types",
    ),
    radix_pairs_empty_static=(
        load
        + pair_values
        + pair_sort.replace(
            "Default::default()",
            "coop::RadixSortOptions::default().begin_bit_const::<8>().end_bit_const::<8>()",
        ),
        "error[E0080]: evaluation panicked: Radix Sort requires begin < end",
    ),
    radix_pairs_varying_end=(
        load
        + pair_values
        + pair_sort.replace(
            "Default::default()",
            "coop::RadixSortOptions::default().end_bit(thread::threadIdx_x() as i64+1)",
        ),
        "not proved block-uniform",
    ),
    radix_pairs_value_control=(
        load + pair_values + pair_sort + "if pair_result[0]==0.0 {" + pair_sort + "}",
        "lane-varying",
    ),
    radix_pairs_storage_missing_barrier=(
        load + pair_values + new_storage + pair_manual + pair_manual,
        "requires a full-block barrier",
    ),
    radix_pairs_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + pair_values
        + pair_sort
        + store,
        "requires a barrier covering both accesses",
    ),
    radix_pairs_tile_capacity=(
        "values.write(0,0_i32);let keys=coop::ThreadData::<i32,512>::new();"
        "let associated=coop::ThreadData::<f64,512>::new();"
        "let _=coop::radix_sort_pairs(block,&keys,&associated,Default::default());",
        "at most 65535 keys",
    ),
)

# Rank returns natural-aligned i32 positions while preserving its key payload.
radix_rank = "let result=coop::radix_rank_keys(block,&values,Default::default());"
cases.update(
    radix_rank_generic_defaults=(
        load + "fn forward<T:coop::RadixKey,const N:usize,const A:usize>("
        "block:coop::ThreadGroup<coop::Block>,keys:&coop::ThreadData<T,N,A>,begin:i64)"
        "->coop::ThreadData<i32,N> where coop::Alignment<A>:coop::SupportedAlignment {"
        "let _:coop::ThreadData<i32,N>=coop::radix_rank_keys(block,keys,Default::default());"
        "coop::radix_rank_keys(block,keys,coop::RadixRankOptions::default()"
        ".radix_bits::<4>().begin_bit(begin).order(coop::Descending))}"
        "let mut keys=coop::ThreadData::<i32,4,32>::new();"
        + load.replace("&mut values", "&mut keys")
        + "let result=forward(block,&keys,valid as i64);"
        + store.replace("&values", "&result"),
        None,
    ),
    radix_rank_implicit_reuse=(
        load
        + radix_rank.replace("result", "first")
        + radix_rank
        + "values.write(0,first[0]+result[0]);"
        + store,
        None,
    ),
    radix_rank_read_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + radix_rank
        + sync
        + store,
        None,
    ),
    radix_rank_varying_begin=(
        load
        + radix_rank.replace(
            "Default::default()",
            "coop::RadixRankOptions::default().radix_bits::<4>()"
            ".begin_bit(thread::threadIdx_x() as i64)",
        ),
        "not proved block-uniform",
    ),
    radix_rank_result_control=(
        load + radix_rank + "if result[0]==0 {" + radix_rank + "}",
        "lane-varying",
    ),
    radix_rank_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + radix_rank
        + store,
        "requires a barrier covering both accesses",
    ),
    radix_rank_tile_capacity=(
        "values.write(0,0_i32);let keys=coop::ThreadData::<i32,512>::new();"
        "let _=coop::radix_rank_keys(block,&keys,Default::default());",
        "at most 65535 keys",
    ),
)

# Merge Sort uses complete groups, typed partial controls and existing scratch epochs.
merge_sort = "let result=coop::merge_sort_keys(block,&values,Default::default());"
merge_sort_manual = merge_sort.replace(
    "Default::default()", "coop::MergeSortOptions::default().temp_storage(&mut scratch)"
)
cases.update(
    merge_sort_generic=(
        "fn sorted<T:coop::Numeric,G:coop::MergeSortGroup<S>,O:coop::MergeSortOrder,"
        "P:coop::MergeSortPrefix<T>,S:coop::TempStorageLike,const N:usize,const A:usize>"
        "(g:coop::ThreadGroup<G>,v:&coop::ThreadData<T,N,A>,o:coop::MergeSortOptions<T,O,P,S>)"
        "->coop::ThreadData<T,N,A> where coop::Alignment<A>:coop::SupportedAlignment"
        "{coop::merge_sort_keys(g,v,o)}"
        + load
        + "let _=sorted(block,&values,Default::default());"
        "let group=coop::this_warp().group_by::<8>();"
        "let _=sorted(group,&values,coop::MergeSortOptions::default()"
        ".order(coop::Descending).valid_items(8,i32::MIN));"
        "let _=sorted(group,&values,coop::MergeSortOptions::default()"
        ".valid_items(8,i32::MIN).order(coop::Descending));",
        None,
    ),
    merge_sort_storage_completed=(
        load
        + new_storage
        + merge_sort_manual
        + sync
        + merge_sort_manual
        + sync
        + store,
        None,
    ),
    merge_sort_auto_epoch_completed=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + sync
        + new_storage.replace("new()", "new().auto_sync()")
        + merge_sort_manual
        + merge_sort_manual
        + sync
        + store,
        None,
    ),
    merge_sort_nonpower_partial_block=(load + merge_sort, "power-of-two Block"),
    merge_sort_warp_partial_geometry=(
        load + merge_sort.replace("(block,", "(coop::this_warp(),"),
        "complete physical warps",
    ),
    merge_sort_varying_count=(
        load
        + merge_sort.replace(
            "Default::default()",
            "coop::MergeSortOptions::default().valid_items(thread::threadIdx_x() as i64,i32::MAX)",
        ),
        "not proved block-uniform",
    ),
    merge_sort_varying_sentinel=(
        load
        + merge_sort.replace(
            "Default::default()",
            "coop::MergeSortOptions::default().valid_items(1,thread::threadIdx_x() as i32)",
        ),
        "not proved block-uniform",
    ),
    merge_sort_result_control=(
        load + merge_sort + "if result[0]==0 {" + merge_sort + "}",
        "lane-varying",
    ),
    merge_sort_storage_missing_barrier=(
        load + new_storage + merge_sort_manual + merge_sort_manual,
        "requires a full-block barrier",
    ),
    merge_sort_read_epoch=(
        reload.replace(".valid_items(valid)", ".valid_items(valid).oob_default(0)")
        + merge_sort
        + store,
        "requires a barrier covering both accesses",
    ),
)

# Merge Sort Pairs keeps independent value types, extents and result owners.
merge_pairs = "let (result,pair_result)=coop::merge_sort_pairs(block,&values,&associated,Default::default());"
merge_pairs_manual = merge_pairs.replace(
    "Default::default()", "coop::MergeSortOptions::default().temp_storage(&mut scratch)"
)
cases.update(
    merge_pairs_generic=(
        "fn sorted<K:coop::Numeric,V:coop::MergeSortValue,G:coop::MergeSortGroup<S>,"
        "O:coop::MergeSortOrder,P:coop::MergeSortPrefix<K>,S:coop::TempStorageLike,"
        "const N:usize,const KA:usize,const VA:usize>(g:coop::ThreadGroup<G>,"
        "keys:&coop::ThreadData<K,N,KA>,values:&coop::ThreadData<V,N,VA>,"
        "o:coop::MergeSortOptions<K,O,P,S>)"
        "->(coop::ThreadData<K,N,KA>,coop::ThreadData<V,N,VA>) "
        "where coop::Alignment<KA>:coop::SupportedAlignment,coop::Alignment<VA>:coop::SupportedAlignment"
        "{coop::merge_sort_pairs(g,keys,values,o)}"
        + load
        + pair_values
        + "let _=sorted(block,&values,&associated,Default::default());"
        "let (result,pair_result)=sorted(block,&values,&associated,coop::MergeSortOptions::default()"
        ".valid_items(valid as i64,i32::MIN).order(coop::Descending));"
        "let _:coop::ThreadData<f64,4,32>=pair_result;"
        + store.replace("&values", "&result"),
        None,
    ),
    merge_pairs_storage_completed=(
        load
        + pair_values
        + new_storage
        + merge_pairs_manual.replace("result,pair_result", "first,first_values")
        + sync
        + merge_pairs_manual
        + sync
        + "values.write(0,first[0]+result[0]+first_values[0] as i32+pair_result[0] as i32);"
        + "let mut scratch=coop::TempStorage::new().auto_sync();"
        + merge_pairs_manual
        + merge_pairs_manual
        + sync
        + store,
        None,
    ),
    merge_pairs_unequal_extent=(
        load
        + pair_values.replace("f64,4,32", "f64,3,32").replace("0..4", "0..3")
        + merge_pairs,
        "mismatched types",
    ),
    merge_pairs_value_sentinel=(
        load
        + pair_values
        + merge_pairs.replace(
            "Default::default()",
            "coop::MergeSortOptions::default().valid_items(1,0_f64)",
        ),
        "mismatched types",
    ),
    merge_pairs_value_result_type=(
        load
        + pair_values
        + merge_pairs
        + "let _:coop::ThreadData<i32,4,32>=pair_result;",
        "mismatched types",
    ),
    merge_pairs_value_control=(
        load
        + pair_values
        + merge_pairs
        + "if pair_result[0]==0.0 {"
        + merge_pairs
        + "}",
        "lane-varying",
    ),
    merge_pairs_storage_missing_barrier=(
        load + pair_values + new_storage + merge_pairs_manual + merge_pairs_manual,
        "requires a full-block barrier",
    ),
)

if args.case:
    unknown = set(args.case).difference(cases)
    if unknown:
        parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
    cases = {name: case for name, case in cases.items() if name in args.case}


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
    CUDA_HOME=str(args.cuda_root.resolve()),
    CUDA_PATH=str(args.cuda_root.resolve()),
)
if args.backend:
    env["CUDA_OXIDE_BACKEND"] = str(args.backend.resolve())
else:
    env.pop("CUDA_OXIDE_BACKEND", None)
results = []
for name, (body, diagnostic) in cases.items():
    block_warp = name.startswith(("warp_transpose_", "warp_timesliced_"))
    case_prefix = warp_prefix if name.startswith("warp_") and not block_warp else prefix
    if name.startswith("logical_"):
        case_prefix = logical_prefix
    if name.startswith("batched_"):
        case_prefix = case_prefix.replace(
            "use cuda_coop as coop;",
            "#![feature(generic_const_exprs)]\n#![allow(incomplete_features)]\nuse cuda_coop as coop;",
        )
    if name == "radix_pairs_generic_defaults":
        case_prefix = case_prefix.replace(
            "valid:usize)",
            "valid:usize,mut value_output:DisjointSlice<f64,CollectiveTiles<4>>)",
        )
    if name == "logical_width1_origin":
        case_prefix = case_prefix.replace(
            "valid:usize)",
            "valid:usize,mut other:DisjointSlice<i32,CollectiveTiles<4>>)",
        )
        case_prefix = case_prefix.replace("pub fn probe(", "pub fn probe<'a>(").replace(
            "DisjointSlice<i32,", "DisjointSlice<'a,i32,"
        )
    if name == "block_zero_raw_numeric":
        case_prefix = case_prefix.replace(
            "dynamic_shared=inferred", "dynamic_shared=512"
        )
    if name == "histogram_multidimensional":
        case_prefix = case_prefix.replace(
            "domain=1,block=(128,1,1)", "domain=3,block=(4,4,8)"
        )
    if name == "merge_sort_warp_partial_geometry":
        case_prefix = case_prefix.replace("block=(128,1,1)", "block=(48,1,1)")
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
        {
            "command": command,
            "backend": env.get("CUDA_OXIDE_BACKEND"),
            "cases": results,
        },
        indent=2,
    )
    + "\n"
)
raise SystemExit(0 if all(row["pass"] for row in results) else 1)
