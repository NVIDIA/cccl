# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the cuda-coop-cutlass release files with their package index hashes."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


def published_hashes(url: str) -> dict[str, str]:
    try:
        with urlopen(url, timeout=15) as response:
            files = json.load(response)["urls"]
    except HTTPError as error:
        if error.code != 404:
            raise
        return {}
    return {file["filename"]: file["digests"]["sha256"] for file in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=["testpypi", "pypi"])
    parser.add_argument("dist", type=Path)
    parser.add_argument(
        "--allow-missing", action="store_true", help="Allow files awaiting upload"
    )
    args = parser.parse_args()

    wheels = list(args.dist.glob("cuda_coop_cutlass-*-py3-none-any.whl"))
    if len(wheels) != 1:
        parser.error("expected one cuda-coop-cutlass universal wheel")
    wheel = wheels[0]
    prefix = wheel.name.removesuffix("-py3-none-any.whl")
    version = prefix.removeprefix("cuda_coop_cutlass-")
    sdist = args.dist / f"{prefix}.tar.gz"
    if set(args.dist.iterdir()) != {wheel, sdist}:
        parser.error("expected only the release wheel and its matching sdist")
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (wheel, sdist)
    }
    host = "test.pypi.org" if args.index == "testpypi" else "pypi.org"
    url = f"https://{host}/pypi/cuda-coop-cutlass/{quote(version, safe='')}/json"
    for attempt in range(6):
        published = published_hashes(url)
        for name, digest in published.items():
            if name not in expected:
                raise SystemExit(f"Unexpected file on {host}: {name}")
            if digest != expected[name]:
                raise SystemExit(f"SHA-256 mismatch on {host}: {name}")
        missing = expected.keys() - published.keys()
        if not missing or args.allow_missing:
            print(f"Verified {len(published)} of {len(expected)} files at {url}")
            return
        # A completed upload can take a few seconds to appear in the JSON API.
        if attempt < 5:
            time.sleep(10)
    raise SystemExit(f"Missing from {host}: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
