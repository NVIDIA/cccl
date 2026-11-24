# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verify the cuda-coop-cutlass release wheel and its hash on the package index."""

import argparse
import json
import time
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
    parser.add_argument("version")
    parser.add_argument("sha256", help="SHA-256 of the GPU-tested wheel")
    parser.add_argument(
        "--allow-missing", action="store_true", help="Allow the wheel to await upload"
    )
    args = parser.parse_args()

    wheel = f"cuda_coop_cutlass-{args.version}-py3-none-any.whl"
    host = "test.pypi.org" if args.index == "testpypi" else "pypi.org"
    url = f"https://{host}/pypi/cuda-coop-cutlass/{quote(args.version, safe='')}/json"
    for attempt in range(6):
        published = published_hashes(url)
        for name, remote_digest in published.items():
            if name != wheel:
                raise SystemExit(f"Unexpected file on {host}: {name}")
            if remote_digest != args.sha256:
                raise SystemExit(f"SHA-256 mismatch on {host}: {name}")
        if published:
            print(f"Verified {wheel} at {url}")
            return
        if args.allow_missing:
            print(f"Awaiting upload of {wheel} at {url}")
            return
        # A completed upload can take a few seconds to appear in the JSON API.
        if attempt < 5:
            time.sleep(10)
    raise SystemExit(f"Missing from {host}: {wheel}")


if __name__ == "__main__":
    main()
