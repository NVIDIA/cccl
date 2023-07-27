#!/bin/env bash


mkdir -p img

if [ ! -n "$(find img -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O img/logo.png

    # Parse files and collects unique names ending with .png
    imgs=$(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub)
    imgs="${imgs}\ncub_overview.png\nnested_composition.png\ntile.png\nblocked.png\nstriped.png"

    for img in $(echo -e ${imgs} | sort | uniq)
    do 
        echo ${img}
        wget -q https://nvlabs.github.io/cub/${img} -O img/${img}
    done  
fi

./repo.sh docs
