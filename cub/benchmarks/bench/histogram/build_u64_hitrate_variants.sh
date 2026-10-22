#!/usr/bin/env bash
# Build the 4 histogram benches with BOTH the hit-rate instrumentation
# (-DCUB_HISTO_TRACK_HITRATE=1), u32 local counters, and u64 global counters/offsets
# (-DTUNE_LocalCounterT='unsigned int' -DTUNE_GlobalCounterT='unsigned long long'
#  -DTUNE_OffsetT='long long') into
# *.hitrate.u64 binaries. Local cache counts remain 32-bit, while the 64-bit output
# type and large-input extent may still change kernel occupancy and the queried slot
# count. Measure this variant directly rather than reusing rates from the narrow-output
# binary.
#
# Surgical recompile + relink against the prebuilt nvbench objects, same recipe as
# build_hitrate_variants.sh and build_u64_variants.sh. DO NOT run while a benchmark
# is on the GPU. Caller must have written /tmp/hist_u64_cmds.json (the per-label
# base compile commands), as build_u64_variants.sh requires.
set -euo pipefail
BUILD=$(cd "${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}" && pwd)
CMDS=/tmp/hist_u64_cmds.json
LABELS="even range multi_even multi_range"
LEGACY_COUNTER="${TUNE_COUNTER:-}"
LOCAL_COUNTER="${TUNE_LOCAL_COUNTER:-${LEGACY_COUNTER:-unsigned int}}"
GLOBAL_COUNTER="${TUNE_GLOBAL_COUNTER:-${LEGACY_COUNTER:-unsigned long long}}"
OFFSET="${TUNE_OFFSET:-long long}"
cd "$BUILD"
for L in $LABELS; do
  OUT="bin/cub.bench.histogram.${L}.base.hitrate.u64"
  OBJ="/tmp/hist_${L}_hitrate_u64.o"
  echo "  compile+link $L (TRACK_HITRATE=1, LocalCounter=$LOCAL_COUNTER GlobalCounter=$GLOBAL_COUNTER OffsetT=$OFFSET) -> $OUT"
  rm -f "$OBJ"
  L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" \
    LOCAL_COUNTER="$LOCAL_COUNTER" GLOBAL_COUNTER="$GLOBAL_COUNTER" OFFSET="$OFFSET" python3 - <<'PY'
import json, os, subprocess, shlex, sys
L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
local_counter=os.environ['LOCAL_COUNTER']; global_counter=os.environ['GLOBAL_COUNTER']; offset=os.environ['OFFSET']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
i=toks.index('-o'); toks[i+1]=obj
# Inject instrumentation plus the three type defines after nvcc (the command may be ccache-prefixed).
nvcc_i=next(j for j,t in enumerate(toks) if t.endswith('nvcc'))
defines=['-DCUB_HISTO_TRACK_HITRATE=1', f'-DTUNE_LocalCounterT={local_counter}', f'-DTUNE_GlobalCounterT={global_counter}', f'-DTUNE_OffsetT={offset}']
toks=toks[:nvcc_i+1] + defines + toks[nvcc_i+1:]
r=subprocess.run(toks, cwd=build)
if r.returncode!=0: sys.exit(f'compile {L} hitrate.u64 failed')
helper=f'{build}/nvbench_helper/CMakeFiles/cccl.nvbench_helper.dir/nvbench_helper/nvbench_helper.cu.o'
mainobj=f'{build}/_deps/nvbench-build/nvbench/CMakeFiles/nvbench.main.dir/main.cu.o'
libs=shlex.split(f'-Xlinker -rpath -Xlinker {build}/lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so '
  f'{build}/lib/libnvbench.so -lstdc++fs /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so '
  f'/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl /usr/lib/x86_64-linux-gnu/librt.a '
  f'/usr/local/cuda/lib64/libcupti.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl')
link=['/usr/local/cuda/bin/nvcc','-ccbin=/usr/bin/c++','-arch=native',helper,mainobj,obj,'-o',out]+libs
r=subprocess.run(link, cwd=build)
if r.returncode!=0: sys.exit(f'link {L} hitrate.u64 failed')
PY
done
echo "=== u64 hitrate variants: ==="
find bin -maxdepth 1 -type f -name '*.hitrate.u64' -printf '%s %p\n' | sort
