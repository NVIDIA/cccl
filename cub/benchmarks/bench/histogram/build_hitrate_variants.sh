#!/usr/bin/env bash
# Build the 4 histogram benches with -DCUB_HISTO_TRACK_HITRATE=1 into *.hitrate
# binaries (surgical recompile + relink against the prebuilt nvbench objects).
# The hit-rate sweep (pass 2) uses these; the perf sweep (pass 1) uses the normal
# zero-overhead binaries. DO NOT run while a benchmark is on the GPU.
set -euo pipefail
# Benchmark build dir (override with HIST_BENCH_BUILD). Must already contain the
# prebuilt nvbench objects + the 4 stock histogram bench targets.
BUILD=${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}
CMDS=/tmp/hist_compile_cmds.json
LABELS="even range multi_even multi_range"
cd "$BUILD"
for L in $LABELS; do
  OUT="bin/cub.bench.histogram.${L}.base.hitrate"
  OBJ="/tmp/hist_${L}_hitrate.o"
  echo "  compile+link $L -> $OUT"
  L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" python3 - <<'PY'
import json, os, subprocess, shlex, sys
L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
i=toks.index('-o'); toks[i+1]=obj
toks=[toks[0], '-DCUB_HISTO_TRACK_HITRATE=1'] + toks[1:]
if not os.path.exists(obj):
    r=subprocess.run(toks, cwd=build)
    if r.returncode!=0: sys.exit(f'compile {L} hitrate failed')
helper=f'{build}/nvbench_helper/CMakeFiles/cccl.nvbench_helper.dir/nvbench_helper/nvbench_helper.cu.o'
mainobj=f'{build}/_deps/nvbench-build/nvbench/CMakeFiles/nvbench.main.dir/main.cu.o'
libs=shlex.split(f'-Xlinker -rpath -Xlinker {build}/lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so '
  f'{build}/lib/libnvbench.so -lstdc++fs /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so '
  f'/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl /usr/lib/x86_64-linux-gnu/librt.a '
  f'/usr/local/cuda/lib64/libcupti.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl')
link=['/usr/local/cuda/bin/nvcc','-ccbin=/usr/bin/c++','-arch=native',helper,mainobj,obj,'-o',out]+libs
r=subprocess.run(link, cwd=build)
if r.returncode!=0: sys.exit(f'link {L} hitrate failed')
PY
done
echo "=== hitrate variants: ==="; ls -la bin/*hitrate 2>/dev/null | awk '{print $5,$9}'
