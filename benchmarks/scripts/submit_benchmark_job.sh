#!/bin/bash

# This script schedules a SLURM job via crun on computelab to run all CCCL benchmarks and produce a benchmark database
# TODO: set those accordingly
scratch=/home/scratch."$USER"_sw
node_selector="cpu.arch=x86_64 and gpu.product_name='*B200*'"
container_image="rapidsai/devcontainers:25.12-cpp-gcc14-cuda13.0"
jobtime="4:00:00"
benchmark_preset="cub-benchmark"

batch_script=$scratch/batch.sh
cat << BATCH_SCRIPT > $batch_script
#!/bin/bash

pip install --break-system-packages fpzip pandas scipy

# clone CCCL
host=\$(hostname)
cd $scratch
if [ -d "\$host/cccl" ]; then
    rm -r \$host/cccl
fi
mkdir \$host
cd \$host
git clone --depth 1 git@github.com:NVIDIA/cccl.git
cd cccl

# configure cmake
mkdir build_perf
cd build_perf
cmake .. --preset $benchmark_preset

# run benchmarks
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../benchmarks/scripts/
../benchmarks/scripts/run.py

echo "Benchmark done. Results in $scratch/\$host/cccl/build_perf/cccl_meta_bench.db"
BATCH_SCRIPT
chmod +x $batch_script

# schedule SLURM job
echo "Scheduling script $batch_script"
echo "#################################################################################"
cat $batch_script
echo "#################################################################################"
crun -q "$node_selector" -ex -t $jobtime -img $container_image -b $batch_script
