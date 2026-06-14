#!/usr/bin/env bash
# Build the 4 histogram benches with BOTH the hit-rate instrumentation
# (-DCUB_HISTO_TRACK_HITRATE=1) AND the 64-bit counter/offset widths
# (-DTUNE_CounterT='unsigned long long' -DTUNE_OffsetT='long long') into
# *.hitrate.u64 binaries. These let the hit-rate sweep measure cache hit rate at
# the 64-bit counter width -- which differs from the 4-byte rate because the cache
# slot count is byte-budget / (sizeof(int) + replicas*CounterSize()), so a wider
# counter fits FEWER slots and hits less. (The 4-byte rates in the main run's
# hitrate_results.json are therefore NOT reusable for the u64 figures.)
#
# Surgical recompile + relink against the prebuilt nvbench objects, same recipe as
# build_hitrate_variants.sh and build_u64_variants.sh. DO NOT run while a benchmark
# is on the GPU. Caller must have written /tmp/hist_u64_cmds.json (the per-label
# base compile commands), as build_u64_variants.sh requires.
set -euo pipefail
BUILD=$(cd "${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}" && pwd)
CMDS=/tmp/hist_u64_cmds.json
LABELS="even range multi_even multi_range"
COUNTER="${TUNE_COUNTER:-unsigned long long}"
OFFSET="${TUNE_OFFSET:-long long}"
cd "$BUILD"
for L in $LABELS; do
  OUT="bin/cub.bench.histogram.${L}.base.hitrate.u64"
  OBJ="/tmp/hist_${L}_hitrate_u64.o"
  echo "  compile+link $L (TRACK_HITRATE=1, CounterT=$COUNTER OffsetT=$OFFSET) -> $OUT"
  rm -f "$OBJ"
  L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" COUNTER="$COUNTER" OFFSET="$OFFSET" python3 - <<'PY'
import json, os, subprocess, shlex, sys
L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
counter=os.environ['COUNTER']; offset=os.environ['OFFSET']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
i=toks.index('-o'); toks[i+1]=obj
# Inject all three defines right after the nvcc token (command may be ccache-prefixed).
nvcc_i=next(j for j,t in enumerate(toks) if t.endswith('nvcc'))
defines=['-DCUB_HISTO_TRACK_HITRATE=1', f'-DTUNE_CounterT={counter}', f'-DTUNE_OffsetT={offset}']
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
echo "=== u64 hitrate variants: ==="; ls -la bin/*.hitrate.u64 2>/dev/null | awk '{print $5,$9}'
