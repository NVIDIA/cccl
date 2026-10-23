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
        "requires a full-block barrier",
    ),
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
    text = prefix + body + "\n}\nfn main() {}\n"
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
