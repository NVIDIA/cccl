#!/bin/bash
set -e
cuobjdump=$1; cubin=$2; testfile=$3
$cuobjdump -symbols "$cubin" | FileCheck --allow-empty "$testfile"
