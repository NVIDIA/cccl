#!/usr/bin/env python3

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: nvcc_time_trace.py <nvcc> [arguments...]")
    compiler = sys.argv[1]
    args = sys.argv[2:]
    # CUDA 13.3 can lose temporary files when time tracing is combined with
    # parallel device-image compilation. Keep parallelism between CUDA files.
    for index, arg in enumerate(args):
        if arg in ("--threads", "-t"):
            args[index + 1] = "1"
        elif arg.startswith(("--threads=", "-t=")):
            args[index] = "--threads=1"
    os.execvp(compiler, [compiler, *args, "--fdevice-time-trace=-"])


if __name__ == "__main__":
    main()
