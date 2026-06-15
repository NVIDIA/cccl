#!/usr/bin/env python3
# Merge two histogram-sweep JSONs that cover DISJOINT sample types (e.g. an I32-only run
# and an F64-only run of the same matrix) into one combined JSON. Every cell key is
# "SampleT|Elements|Bins|InputShape", so I32 and F64 keys never collide -- the merge is a
# per-(binary, column) dict union. Refuses to overwrite a key with a different value
# (would mean the inputs are NOT disjoint / are the same sample from two epochs).
#
# Usage: histogram_merge_samples.py OUT.json IN_A.json IN_B.json [IN_C.json ...]
import json
import sys

out_path = sys.argv[1]
inputs = sys.argv[2:]
if len(inputs) < 2:
    sys.exit("need >=2 input JSONs to merge")

merged = {}
collisions = 0
for path in inputs:
    d = json.load(open(path))
    for binary, cols in d.items():
        mb = merged.setdefault(binary, {})
        for col, cells in cols.items():
            mc = mb.setdefault(col, {})
            for k, v in cells.items():
                if k in mc and mc[k] != v:
                    collisions += 1
                    if collisions <= 5:
                        sys.stderr.write(f"  COLLISION {binary}/{col}/{k}: {mc[k]} vs {v}\n")
                mc[k] = v

if collisions:
    sys.exit(f"REFUSING: {collisions} key collisions -- inputs are not disjoint samples")

json.dump(merged, open(out_path, "w"), indent=0)
# report sample coverage per binary
for binary in merged:
    samples = set()
    for col, cells in merged[binary].items():
        for k in cells:
            samples.add(k.split("|", 1)[0])
    ncols = len(merged[binary])
    print(f"  {binary}: {ncols} columns, samples={sorted(samples)}")
print(f"wrote {out_path}")
