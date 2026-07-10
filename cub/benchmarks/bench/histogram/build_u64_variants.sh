#!/usr/bin/env bash
# Build wide-output / 64-bit-offset histogram bench variants for the large-input
# characterization sweep. Local accumulation stays u32 while final histogram outputs
# are u64, so individual on-chip/block partials remain cheap without overflowing the
# result when more than 2^32 samples land in one bin. TUNE_OffsetT is also 64-bit because
# it is required for >2^31 elements. The variants are written as *.u64
# binaries, relinking against the prebuilt nvbench objects -- same surgical recompile
# recipe as build_hitrate_variants.sh. DO NOT run while a benchmark is on the GPU.
#
# Caller must first write /tmp/hist_u64_cmds.json via:
#   (cd <build>; ninja -t compdb) | <extract even/range .cu compile commands>
set -euo pipefail
BUILD=$(cd "${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}" && pwd)
CMDS=/tmp/hist_u64_cmds.json
LABELS="even range multi_even multi_range"
# Use libcu++ fixed-width types. The cache and spill paths use libcu++ atomics, so
# uint64_t is supported even on platforms where it aliases unsigned long rather than
# unsigned long long.
LEGACY_COUNTER="${TUNE_COUNTER:-}"
LOCAL_COUNTER="${TUNE_LOCAL_COUNTER:-${LEGACY_COUNTER:-cuda::std::uint32_t}}"
GLOBAL_COUNTER="${TUNE_GLOBAL_COUNTER:-${LEGACY_COUNTER:-cuda::std::uint64_t}}"
OFFSET="${TUNE_OFFSET:-cuda::std::int64_t}"
cd "$BUILD"
# Map the label (used for the compile-cmds key + obj name) to the dotted CUB target
# stem the unified sweep expects: multi_even -> multi.even, multi_range -> multi.range.
target_stem() { case "$1" in multi_even) echo "multi.even";; multi_range) echo "multi.range";; *) echo "$1";; esac; }
for L in $LABELS; do
  STEM=$(target_stem "$L")
  OUT="bin/cub.bench.histogram.${STEM}.base.u64"
  OBJ="/tmp/hist_${L}_u64.o"
  echo "  compile+link $L (LocalCounter=$LOCAL_COUNTER GlobalCounter=$GLOBAL_COUNTER OffsetT=$OFFSET) -> $OUT"
  rm -f "$OBJ"
  L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" \
    LOCAL_COUNTER="$LOCAL_COUNTER" GLOBAL_COUNTER="$GLOBAL_COUNTER" OFFSET="$OFFSET" python3 - <<'PY'
import json, os, subprocess, shlex, sys
L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
local_counter=os.environ['LOCAL_COUNTER']; global_counter=os.environ['GLOBAL_COUNTER']; offset=os.environ['OFFSET']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
i=toks.index('-o'); toks[i+1]=obj
toks=[t for t in toks if not t.startswith(('-DTUNE_LocalCounterT=', '-DTUNE_GlobalCounterT=', '-DTUNE_OffsetT='))]
nvcc_i=next(j for j,t in enumerate(toks) if t.endswith('nvcc'))
toks=toks[:nvcc_i+1] + [f'-DTUNE_LocalCounterT={local_counter}', f'-DTUNE_GlobalCounterT={global_counter}', f'-DTUNE_OffsetT={offset}'] + toks[nvcc_i+1:]
r=subprocess.run(toks, cwd=build)
if r.returncode!=0: sys.exit(f'compile {L} u64 failed')
helper=f'{build}/nvbench_helper/CMakeFiles/cccl.nvbench_helper.dir/nvbench_helper/nvbench_helper.cu.o'
mainobj=f'{build}/_deps/nvbench-build/nvbench/CMakeFiles/nvbench.main.dir/main.cu.o'
libs=shlex.split(f'-Xlinker -rpath -Xlinker {build}/lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so '
  f'{build}/lib/libnvbench.so -lstdc++fs /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so '
  f'/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl /usr/lib/x86_64-linux-gnu/librt.a '
  f'/usr/local/cuda/lib64/libcupti.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl')
link=['/usr/local/cuda/bin/nvcc','-ccbin=/usr/bin/c++','-arch=native',helper,mainobj,obj,'-o',out]+libs
r=subprocess.run(link, cwd=build)
if r.returncode!=0: sys.exit(f'link {L} u64 failed')
PY
done
echo "=== u64 variants: ==="
find bin -maxdepth 1 -type f -name '*.u64' -printf '%s %p\n' | sort
