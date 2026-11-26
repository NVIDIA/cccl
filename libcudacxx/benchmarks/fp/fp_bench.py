#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build, run and compare the nvbench benchmarks of the fp component on demand.

    fp_bench.py                                   # every benchmark: configure, build, run, table
    fp_bench.py --bench fpmp --family arith trig  # one benchmark, two of its executables
    fp_bench.py --compare-ref origin/main         # the same, and the same benchmarks built against
                                                  # the library of another git ref, then compared
    fp_bench.py --baseline other/fpmp.csv         # compare with the table of an earlier run
    fp_bench.py --baseline-dir baselines --store-baseline --note 'main <sha>'

The benchmarks live in libcudacxx/benchmarks/bench/fp/<bench>/ (one executable per .cu file)
and are built with CCCL's `libcudacxx-benchmark` CMake preset in the preset's own tree,
build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark under the checkout (the tree a manual
`cmake --preset libcudacxx-benchmark` or ci/util/build_and_test_targets.sh uses too; --build-dir
puts it elsewhere). Each benchmark has a table script that joins the nvbench CSV output into one
row per cell and a comparison script that judges a run against a baseline; both live next to
this file. Results go to OUT/<bench>/ (the nvbench csv/json/md/log of each executable, the
joined table, compare.csv and compare.md); OUT is build/fp_bench by default.

`--compare-ref REF` is the regression check: it adds a git worktree of REF under OUT/ref,
overlays the current libcudacxx/benchmarks directory onto it (so both sides run the same
benchmark sources and tooling against their own library headers), builds it in its own
build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark, runs it, and compares the current run with
it. REF needs no benchmarks of its own. The worktree is kept for the next run and refreshed
to REF each time.

Storage matters: a family's compile opens a few thousand headers of the checkout, and on a
network home share that turned a one-minute build into seven; the driver measures the open
and write latency of the checkout and the output directories at start and warns when they are
slow (README, Storage).

The exit status is 1 when a comparison reports a FAIL, when a build or run fails, and 0
otherwise. Timings on a GPU shared with another process read 20 to 40 % low; the comparison
scripts take `--compare-args '--perf-fail 100'` to turn performance moves into warnings.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The benchmarks of the fp component: where the sources are (relative to the CCCL root), a file
# that proves they are the expected ones, the prefix of their ninja targets (<prefix><family>.base),
# and the scripts (next to this file) that join and compare their tables.
BENCHMARKS = {
    "fpmp": {
        "src": "libcudacxx/benchmarks/bench/fp/fpmp",
        "marker": "fpmp_bench.cuh",
        "prefix": "libcudacxx.bench.fp.fpmp.",
        "table": "fp_table.py",
        "table_file": "fpmp.csv",
        "compare": "fp_compare.py",
    },
}
PRESET = "libcudacxx-benchmark"
DEFAULT_NVBENCH_ARGS = "--min-time 0.1 --max-noise 5 --timeout 10"


def log(msg):
    print(msg, flush=True)


def run(cmd, **kw):
    """Run a command, echo it on failure."""
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log("  [FAIL] " + " ".join(str(c) for c in cmd))
    return r.returncode == 0


def git(root, *args):
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def default_nvcc():
    if shutil.which("nvcc"):
        return None  # cmake finds it
    return (
        "/usr/local/cuda/bin/nvcc"
        if os.path.exists("/usr/local/cuda/bin/nvcc")
        else None
    )


def fs_type(path):
    """The filesystem type of the mount holding `path` (Linux, /proc/mounts), '' when unknown."""
    real, best, typ = os.path.realpath(path), "", ""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1]
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(
                    mnt
                ) > len(best):
                    best, typ = mnt, parts[2]
    except OSError:
        pass
    return typ


def probe_reads(root, count=200):
    """Milliseconds per open of a header of the checkout: the cost a compile pays thousands of times."""
    files = sorted(
        glob.glob(
            os.path.join(root, "libcudacxx", "include", "cuda", "std", "__*", "*.h")
        )
    )[:count]
    if not files:
        return 0.0
    t0 = time.perf_counter()
    for f in files:
        with open(f, "rb") as h:
            h.read(64)
    return (time.perf_counter() - t0) * 1e3 / len(files)


def probe_writes(path, count=20):
    """Milliseconds per small file created, written, closed and removed in `path`."""
    t0 = time.perf_counter()
    for i in range(count):
        p = os.path.join(path, f".fp_bench_probe_{os.getpid()}_{i}")
        with open(p, "wb") as h:
            h.write(b"x" * 4096)
        os.remove(p)
    return (time.perf_counter() - t0) * 1e3 / count


def check_storage(root, out, build_dir):
    """Warn when the checkout or an output volume is slow: a network home share turns the one-minute
    build of a family into seven (README, Storage)."""
    r = probe_reads(root)
    # compile time grows with this figure: 1.6 ms gave a one-minute family build, 10 ms seven minutes;
    # a local disk needs 0.1 ms
    if r > 8.0:
        log(
            f"  [WARN] {root} reads slowly ({r:.1f} ms per header open, {fs_type(root) or 'unknown fs'}): a family"
            " builds in minutes instead of one; put a git worktree of this tree on local or scratch storage (README, Storage)"
        )
    elif r > 3.0:
        log(
            f"  [NOTE] {root}: {r:.1f} ms per header open ({fs_type(root) or 'unknown fs'}); compile time grows with"
            " this figure, a local disk needs 0.1 ms (README, Storage)"
        )
    probed = set()
    for d in (
        out,
        build_dir if os.path.isdir(build_dir) else os.path.dirname(build_dir),
    ):
        if not os.path.isdir(d) or os.path.realpath(d) in probed:
            continue
        probed.add(os.path.realpath(d))
        w = probe_writes(d)
        if w > 20.0:  # a local disk writes in 0.1 ms, a fast NFS volume in 5 ms
            log(
                f"  [WARN] {d} writes slowly ({w:.0f} ms per file, {fs_type(d) or 'unknown fs'}):"
                " use --out and --build-dir on local or scratch storage (README, Storage)"
            )


def cccl_root(explicit):
    if explicit:
        return os.path.realpath(explicit)
    return os.path.realpath(git(HERE, "rev-parse", "--show-toplevel"))


def families(root, bench, wanted):
    src = os.path.join(root, BENCHMARKS[bench]["src"])
    if not os.path.exists(os.path.join(src, BENCHMARKS[bench]["marker"])):
        sys.exit(
            f"no {bench} benchmarks at {src}: the tree needs libcudacxx/benchmarks/bench/fp"
        )
    all_fams = sorted(
        os.path.basename(f)[:-3] for f in glob.glob(os.path.join(src, "*.cu"))
    )
    if not wanted:
        return all_fams
    fams = [f for f in all_fams if f in wanted]
    if not fams:
        sys.exit(f"--family {wanted} matches none of: {all_fams}")
    return fams


def preset_build_dir(root):
    """The preset's own tree, build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark, as CMakePresets.json
    and the other CCCL tooling (ci/build_common.sh, ci/sass/sass_diff.sh) place it."""
    infix = os.environ.get("CCCL_BUILD_INFIX", "")
    return os.path.join(root, "build", *([infix] if infix else []), PRESET)


def cache_vars(build):
    """The source directory, the architectures and CPM's directory of an existing tree, from its
    CMakeCache.txt."""
    cache = os.path.join(build, "CMakeCache.txt")
    vals = {}
    if os.path.exists(cache):
        for line in open(cache):
            for key in (
                "CMAKE_HOME_DIRECTORY",
                "CMAKE_CUDA_ARCHITECTURES",
                "CPM_DIRECTORY",
                "CMAKE_COMMAND",
            ):
                if line.startswith(key + ":"):
                    vals[key] = line.split("=", 1)[1].strip()
    return vals


def tree_is_current(have, root, build, cmake="cmake"):
    """Whether an existing tree was configured for exactly this checkout path. CMake and CPM
    compare path strings, not what they resolve to: a tree configured before the checkout moved
    behind a symlink still resolves to it, but CPM's cached directory no longer matches the one
    its module is included from, and the regeneration fails with "Unknown CMake command
    CPMAddPackage". A tree made on another host records that host's cmake path."""
    if not have or not os.path.exists(os.path.join(build, "build.ninja")):
        return False
    if os.path.realpath(have.get("CMAKE_HOME_DIRECTORY", "")) != os.path.realpath(root):
        return False  # another checkout
    # CMake records CPM's directory resolved, whatever path the source was given as
    cmake_dir = os.path.join(os.path.realpath(root), "cmake")
    if have.get("CPM_DIRECTORY", cmake_dir) != cmake_dir:
        return False
    # ninja regenerates with the cmake that configured the tree: a tree made on a host whose
    # cmake is missing here, or of another version, fails there ("cmake: not found")
    mine = shutil.which(cmake) or cmake
    return (
        cmake_version(have.get("CMAKE_COMMAND", mine))
        == cmake_version(mine)
        is not None
    )


def cmake_version(exe):
    """The first line of `exe --version`, None when it cannot run (a cmake of another host).
    Paths are no basis for the comparison: a snap's `cmake` on PATH is a wrapper, the cache
    records the binary behind it."""
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=True
        ).stdout
        return out.split("\n", 1)[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def configure(root, build, log_dir, arch, cmake, cpm_cache, nvcc):
    """Configure the preset into `build` unless it is configured for this tree already; an
    existing tree keeps its architectures unless --arch names others."""
    have = cache_vars(build)
    if tree_is_current(have, root, build, cmake):
        have_arch = have.get("CMAKE_CUDA_ARCHITECTURES", "")
        if not arch or arch == have_arch:
            log(f"  [CMAKE] {build} (configured, arch {have_arch})")
            return True
        log(f"  [CMAKE] {build}: arch {have_arch} -> {arch}")
        cmd = [cmake, "-S", root, "-B", build, f"-DCMAKE_CUDA_ARCHITECTURES={arch}"]
    else:
        if have:
            log(
                f"  [CMAKE] {build} was configured for {have.get('CMAKE_HOME_DIRECTORY')}"
                f" (CPM at {have.get('CPM_DIRECTORY', '?')}, cmake {have.get('CMAKE_COMMAND', '?')}):"
                f" configuring again for {root}"
            )
            shutil.rmtree(build)
        os.makedirs(build, exist_ok=True)
        arch = arch or "native"
        cmd = [
            cmake,
            "--preset",
            PRESET,
            "-B",
            build,
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        ]
        if nvcc:
            cmd.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        if cpm_cache:
            cmd.append(f"-DCPM_SOURCE_CACHE={cpm_cache}")
        log(f"  [CMAKE] {PRESET} -> {build} (arch {arch})")
    env = dict(os.environ, PWD=root)
    with open(os.path.join(log_dir, "configure.log"), "w") as f:
        return run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)


def build(build_dir, log_dir, targets, jobs):
    log(f"  [NINJA] {' '.join(targets)}")
    with open(os.path.join(log_dir, "build.log"), "w") as f:
        return run(
            ["ninja", "-C", build_dir, f"-j{jobs}", *targets],
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def run_bench(bench, build_dir, out, fams, gpu, nvbench_args):
    """Run the executables of one benchmark into out/<bench>; the joined table follows."""
    res = os.path.join(out, bench)
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    ok = True
    for fam in fams:
        exe = os.path.join(
            build_dir, "bin", BENCHMARKS[bench]["prefix"] + fam + ".base"
        )
        t0 = time.time()
        with open(os.path.join(res, fam + ".log"), "w") as f:
            r = subprocess.run(
                [
                    exe,
                    "--devices",
                    "0",
                    "--csv",
                    os.path.join(res, fam + ".csv"),
                    "--json",
                    os.path.join(res, fam + ".json"),
                    "--md",
                    os.path.join(res, fam + ".md"),
                    *nvbench_args.split(),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if r.returncode == 0:
            log(f"  [OK]   {bench} {fam} ({time.time() - t0:.0f}s)")
        else:
            log(f"  [FAIL] {bench} {fam} (see {res}/{fam}.log)")
            ok = False
    table = os.path.join(res, BENCHMARKS[bench]["table_file"])
    ok &= run(
        [
            sys.executable,
            os.path.join(HERE, BENCHMARKS[bench]["table"]),
            "--bin-dir",
            os.path.join(build_dir, "bin"),
            "--csv-dir",
            res,
            "--out",
            table,
            "--md",
            table[:-4] + ".md",
        ]
    )
    if ok:
        log(f"  [TABLE] {table}")
    return ok


def compare(bench, out, baseline_args, compare_args):
    res = os.path.join(out, bench)
    table = os.path.join(res, BENCHMARKS[bench]["table_file"])
    cmd = [
        sys.executable,
        os.path.join(HERE, BENCHMARKS[bench]["compare"]),
        "--results",
        table,
        *baseline_args,
        *compare_args.split(),
    ]
    if "--store" not in baseline_args:
        cmd += [
            "--out-csv",
            os.path.join(res, "compare.csv"),
            "--out-md",
            os.path.join(res, "compare.md"),
        ]
    return run(cmd)


def ref_tree(root, out, ref):
    """A worktree of `ref` under out/ref/src with the current benchmarks directory overlaid."""
    src = os.path.realpath(os.path.join(out, "ref", "src"))
    sha = git(root, "rev-parse", "--verify", ref + "^{commit}")
    if os.path.exists(os.path.join(src, ".git")):
        subprocess.run(
            ["git", "-C", src, "checkout", "-q", "--detach", sha], check=True
        )
    else:
        shutil.rmtree(src, ignore_errors=True)  # a leftover without .git
        git(root, "worktree", "prune")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        subprocess.run(
            ["git", "-C", root, "worktree", "add", "-q", "--detach", src, sha],
            check=True,
        )
    bench_dir = os.path.join("libcudacxx", "benchmarks")
    shutil.rmtree(os.path.join(src, bench_dir), ignore_errors=True)
    shutil.copytree(
        os.path.join(root, bench_dir),
        os.path.join(src, bench_dir),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    log(f"  [REF]  {ref} = {sha[:10]} at {src}, with the current libcudacxx/benchmarks")
    return src


def one_tree(root, out, build_dir, a, benches):
    """Configure, build and run the selected benchmarks of one tree; returns success."""
    os.makedirs(out, exist_ok=True)
    fams = {b: families(root, b, a.family) for b in benches}
    if not a.no_build:
        if not configure(root, build_dir, out, a.arch, a.cmake, a.cpm_cache, a.nvcc):
            return False
        targets = [
            BENCHMARKS[b]["prefix"] + f + ".base" for b in benches for f in fams[b]
        ]
        if not build(build_dir, out, targets, a.jobs):
            return False
    ok = True
    for b in benches:
        ok &= run_bench(b, build_dir, out, fams[b], a.gpu, a.nvbench_args)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bench",
        action="append",
        choices=sorted(BENCHMARKS),
        help="benchmark(s); default all",
    )
    ap.add_argument(
        "--family",
        action="append",
        help="executable(s) of the benchmark; default all its .cu files",
    )
    ap.add_argument(
        "--out",
        help="results, logs and the ref worktree (default <cccl>/build/fp_bench)",
    )
    ap.add_argument(
        "--build-dir",
        help="the CMake tree of this checkout (default build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark)",
    )
    ap.add_argument(
        "--cccl-root", help="the CCCL checkout (default: the one holding this script)"
    )
    ap.add_argument(
        "--arch",
        help="CMAKE_CUDA_ARCHITECTURES: native for a new tree; an existing tree keeps its own unless given",
    )
    ap.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for the runs, a UUID is safest")
    ap.add_argument(
        "--jobs", type=int, default=4, help="parallel compile jobs (default 4)"
    )
    ap.add_argument(
        "--nvbench-args",
        default=DEFAULT_NVBENCH_ARGS,
        help=f"default: '{DEFAULT_NVBENCH_ARGS}'",
    )
    ap.add_argument("--cmake", default="cmake")
    ap.add_argument(
        "--nvcc",
        default=default_nvcc(),
        help="CMAKE_CUDA_COMPILER (default: nvcc on PATH, else /usr/local/cuda)",
    )
    ap.add_argument(
        "--cpm-cache",
        default=os.environ.get("CPM_SOURCE_CACHE"),
        help="CPM source cache for nvbench",
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="run the executables already in the build tree",
    )
    ap.add_argument("--baseline", help="a table of an earlier run to compare with")
    ap.add_argument("--baseline-dir", help="per-GPU baselines, <Device slug>.csv")
    ap.add_argument(
        "--store-baseline", action="store_true", help="store this run in --baseline-dir"
    )
    ap.add_argument(
        "--note",
        default="",
        help="with --store-baseline: what the baseline is a snapshot of",
    )
    ap.add_argument(
        "--compare-ref",
        help="build and run the same benchmarks against this git ref and compare",
    )
    ap.add_argument(
        "--compare-args", default="", help="extra options of the comparison script"
    )
    a = ap.parse_args()
    if a.store_baseline and not a.baseline_dir:
        ap.error("--store-baseline needs --baseline-dir")

    root = cccl_root(a.cccl_root)
    out = os.path.abspath(a.out or os.path.join(root, "build", "fp_bench"))
    benches = a.bench or sorted(BENCHMARKS)
    os.makedirs(out, exist_ok=True)

    build_dir = os.path.abspath(a.build_dir) if a.build_dir else preset_build_dir(root)
    check_storage(root, out, build_dir)
    log(f"== {', '.join(benches)} of {root}")
    ok = one_tree(root, out, build_dir, a, benches)

    if a.compare_ref:
        src = ref_tree(root, out, a.compare_ref)
        ref_out = os.path.join(out, "ref")
        log(f"== {', '.join(benches)} of {a.compare_ref}")
        ok &= one_tree(src, ref_out, preset_build_dir(src), a, benches)
        for b in benches:
            log(f"== {b}: this tree against {a.compare_ref}")
            ref_table = os.path.join(ref_out, b, BENCHMARKS[b]["table_file"])
            ok &= compare(b, out, ["--baseline", ref_table], a.compare_args)
    if a.baseline or a.baseline_dir:
        for b in benches:
            if a.store_baseline:
                args = ["--baseline-dir", a.baseline_dir, "--store"] + (
                    ["--note", a.note] if a.note else []
                )
            elif a.baseline_dir:
                args = ["--baseline-dir", a.baseline_dir]
            else:
                args = ["--baseline", a.baseline]
            log(
                f"== {b}: "
                + (
                    "storing the baseline"
                    if a.store_baseline
                    else "against the baseline"
                )
            )
            ok &= compare(b, out, args, a.compare_args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
