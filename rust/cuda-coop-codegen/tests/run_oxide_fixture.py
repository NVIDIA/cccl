# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run coordinated source integration with an explicitly supplied Oxide checkout."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oxide-root", required=True, type=Path)
parser.add_argument("--cargo-oxide", required=True, type=Path)
parser.add_argument("--backend", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
parser.add_argument("--arch", default="sm_120")
parser.add_argument(
    "--checked", action="store_true", help="exercise the checked full-block marker"
)
parser.add_argument(
    "--root-context",
    action="store_true",
    help="exercise shared helpers at 64 and 128 threads",
)
parser.add_argument(
    "--carrier", action="store_true", help="exercise private collective-only carriers"
)
parser.add_argument(
    "--validity",
    action="store_true",
    help="exercise private typed Direct payload validity",
)
args = parser.parse_args()
source = Path(__file__).resolve().parent
checkout = source.parents[2]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
fixture = output / "fixture"
shutil.copytree(source / "oxide-fixture", fixture)
# Only the generated copy receives cross-checkout paths. Ordinary source files
# and the reviewed manifests remain unchanged.
api_manifest = fixture / "api/Cargo.toml"
api_manifest.write_text(
    api_manifest.read_text().replace(
        'path = "../../.."', f"path = {json.dumps(str(source.parent))}"
    )
)
oxide = args.oxide_root.resolve()
manifest = fixture / "Cargo.toml"
with manifest.open("a") as stream:
    stream.write("\n[patch.crates-io]\n")
    for name in ["cuda-device", "cuda-host", "cuda-macros", "oxide-artifacts"]:
        stream.write(
            f"{name} = {{ path = {json.dumps(str(oxide / 'cuda-oxide/crates' / name))} }}\n"
        )
    for name in ["cuda-core", "cuda-core-derive", "cuda-bindings", "cuda-async"]:
        stream.write(f"{name} = {{ path = {json.dumps(str(oxide / name))} }}\n")
# Reuse exact same-base dependency resolutions; Cargo adds only fixture/provider
# packages. Do not refresh every example's lockfile for this integration test.
shutil.copyfile(
    oxide / "cuda-oxide/crates/rustc-codegen-cuda/examples/vecadd/Cargo.lock",
    fixture / "Cargo.lock",
)
env = os.environ.copy()
env.update(
    CCCL_SOURCE=str(checkout),
    CUDA_HOME=str(args.cuda_root.resolve()),
    CUDA_PATH=str(args.cuda_root.resolve()),
    CUDA_OXIDE_BACKEND=str(args.backend.resolve()),
)
command = [
    str(args.cargo_oxide.resolve()),
    "oxide",
    "run",
    "--arch",
    args.arch,
    "--materialize-cubin",
]
if args.validity:
    command.extend(["--features", "validity", "--bin", "coop-validity-fixture"])
elif args.root_context:
    command.extend(["--features", "root-context"])
elif args.checked:
    command.extend(["--features", "checked"])
consumer = fixture / ("carrier" if args.carrier or args.validity else "consumer")
with (output / "build-run.log").open("w") as log:
    result = subprocess.run(
        command, cwd=consumer, env=env, stdout=log, stderr=subprocess.STDOUT
    )
(output / "command.json").write_text(
    json.dumps(
        {
            "argv": command,
            "cwd": str(consumer),
            "oxide_root": str(oxide),
            "cccl_source": str(checkout),
            "backend": env["CUDA_OXIDE_BACKEND"],
            "cuda_root": env["CUDA_HOME"],
            "returncode": result.returncode,
        },
        indent=2,
    )
    + "\n"
)
raise SystemExit(result.returncode)
