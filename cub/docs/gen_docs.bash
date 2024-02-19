#!/usr/bin/env bash

## This script just wraps launching a docs build within a container
## Tag is passed on as the first argument ${1}

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

cd $SCRIPT_PATH

## Clean image directory, without this any artifacts will prevent fetching
rm -rf img
mkdir -p img

if [ ! -n "$(find img -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O img/logo.png

    # Parse files and collects unique names ending with .png
    imgs=( $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub | uniq) )
    imgs+=( "cub_overview.png" "nested_composition.png" "tile.png" "blocked.png" "striped.png" )

    for img in "${imgs[@]}"
    do
        echo ${img}
        wget -q https://nvlabs.github.io/cub/${img} -O img/${img} || echo "!!! Failed to fetch $img"
    done
fi

./repo.sh docs || echo "!!! There were errors while generating"
