#!/usr/bin/env sh

## This script will produce a 'build_docs' folder that contains a jekyll site containing all the Thrust docs
## This is used in CI to produce a site for Thrust under CCCL

set -ex

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

cd $SCRIPT_PATH
mkdir -p build
cp github_pages/Gemfile build/Gemfile
./generate_markdown.bash
