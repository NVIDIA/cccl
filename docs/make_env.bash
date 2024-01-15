#!/usr/bin/env sh

## This script just wraps launching a docs build within a container
## Tag is passed on as the first argument ${1}

set -ex

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
cd $SCRIPT_PATH

TAG=${1}

docker build -f ./Dockerfile -t $TAG .
