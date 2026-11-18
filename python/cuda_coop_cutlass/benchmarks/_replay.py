# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load qualified cubins while retaining their source compilation provenance."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from _kernels import PreparedKernel, _checked


def load(case, inputs, outputs, stream, blocks, provider, artifacts, bundle):
    """Require matching source metadata before replaying on the current GPU."""
    from cuda.bindings import driver

    bundle = Path(bundle)
    qualification = bundle / "qualification.json"
    receipt = json.loads(qualification.read_text())
    if receipt["errors"]:
        raise ValueError("Source qualification contains errors")
    matches = [
        result
        for result in receipt["results"]
        if (result["case"], result["blocks"], result["launches"])
        == (case.name, blocks, 1)
    ]
    if len(matches) != 1 or not matches[0]["correct"]:
        raise ValueError("Expected one qualified source kernel for this case")
    source = matches[0]
    metadata = source["kernel"].copy()
    device = _checked(driver.cuCtxGetDevice())
    major = _checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    )
    minor = _checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    )
    expected = {
        "case": asdict(case),
        "provider": provider,
        "arch": f"sm_{major}{minor}",
        "grid": [blocks, 1, 1],
        "block": [case.threads, 1, 1],
    }
    if source["provider"] != provider or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Source kernel metadata does not match this launch")
    directory = bundle / (f"d{source['device']}-{case.name}-b{blocks}-l1")
    names = ["input", "input_aux", "output", "output_aux"]
    abi = metadata.get("abi", [])
    if not abi or len(set(abi)) != len(abi) or not set(abi) <= set(names):
        raise ValueError("Source kernel ABI is not a unique pointer subset")
    layout = metadata.get("parameter_layout")
    if layout is None:
        # Older native bundles have the same declared extern-C signature.
        if provider != "cub" or abi != names:
            raise ValueError("Source kernel has no captured parameter layout")
    else:
        if len(layout) != len(abi):
            raise ValueError("Source parameter layout length differs from ABI")
        previous_end = 0
        for index, parameter in enumerate(layout):
            if (
                parameter["index"] != index
                or parameter["binding"] != abi[index]
                or parameter["size"] != 8
                or parameter["offset"] % 8
                or parameter["offset"] < previous_end
            ):
                raise ValueError("Source parameter layout is not a pointer ABI")
            previous_end = parameter["offset"] + parameter["size"]
    source_metadata = directory / "kernel.json"
    if json.loads(source_metadata.read_text()) != metadata:
        raise ValueError("Kernel metadata differs from source qualification")
    source_cubin = directory / "kernel.cubin"
    digest = hashlib.sha256(source_cubin.read_bytes()).hexdigest()
    if digest != metadata["cubin_sha256"]:
        raise ValueError("Source cubin hash mismatch")
    cubin = Path(artifacts) / "kernel.cubin"
    shutil.copyfile(source_cubin, cubin)
    metadata["compilation_environment"] = metadata.get(
        "compilation_environment", receipt["versions"]
    )
    metadata["compilation_process_environment"] = metadata.get(
        "compilation_process_environment", receipt.get("environment", {})
    )
    metadata["replay_of"] = {
        "qualification": str(qualification.resolve()),
        "qualification_sha256": hashlib.sha256(
            qualification.read_bytes()
        ).hexdigest(),
        "cubin": str(source_cubin.resolve()),
        "cubin_sha256": digest,
        "metadata_sha256": hashlib.sha256(
            source_metadata.read_bytes()
        ).hexdigest(),
    }
    metadata["cubin"] = str(cubin.resolve())
    metadata["replay_loader_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (Path(artifacts) / "kernel.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    pointers = {**inputs, **outputs}
    return PreparedKernel(
        cubin,
        metadata["symbol"],
        stream,
        pointers,
        tuple(metadata["grid"]),
        tuple(metadata["block"]),
        metadata["dynamic_shared_bytes"],
        metadata,
    )
