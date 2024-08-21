#!/usr/bin/env bash

# This script just wraps launching a repo docs build within a container
#
# Additional options, e.g --stage sphinx will be passed on to repo.sh

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

cd $SCRIPT_PATH

## Clean image directory, without this any artifacts will prevent fetching
rm -rf img
mkdir -p img

# Pull cub images
if [ ! -d cubimg ]; then
    git clone -b gh-pages https://github.com/NVlabs/cub.git cubimg
fi

if [ ! -n "$(find cubimg -name 'example_range.png')" ]; then
    wget -q https://raw.githubusercontent.com/NVIDIA/NVTX/release-v3/docs/images/example_range.png -O cubimg/example_range.png
fi

if [ ! -n "$(find img -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O img/logo.png

    # Parse files and collects unique names ending with .png
    imgs=( $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub/cub | uniq) )
    imgs+=( "cub_overview.png" "nested_composition.png" "tile.png" "blocked.png" "striped.png" )

    for img in "${imgs[@]}"
    do
        echo ${img}
        cp cubimg/${img} img/${img}
    done
fi

if ! ./repo.sh docs "$@"; then
    echo "!!! There were errors while generating"
    exit 1
fi
