#!/usr/bin/env bash

## Usage:
## ./scrape_docs.bash [path]
## [path] is the starting point for searching for HTML. Ideally this is the siteroot
## script will find all HTML files and record them into a CSV file that can be used for searching docs.

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

path_to_docs=$(realpath $1)

cd $SCRIPT_PATH

thrust=$(
        cd $path_to_docs;
        find ./ -iname "*.html" -printf 'cccl/%P,'
    )

echo "$thrust" > $path_to_docs/pagelist.txt
