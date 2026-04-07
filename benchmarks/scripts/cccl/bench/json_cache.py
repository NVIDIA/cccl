import json
import os
import subprocess


def get_alg_base(algname: str):
    return os.path.join(".", "bin", algname + ".base")


class JsonCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bench_cache = {}
            cls._instance.device_cache = {}
        return cls._instance

    def get_bench(self, algname):
        if algname not in self.bench_cache:
            result = subprocess.check_output(
                [get_alg_base(algname), "--jsonlist-benches"]
            )
            self.bench_cache[algname] = json.loads(result)
        return self.bench_cache[algname]

    def get_device(self, algname):
        if algname not in self.device_cache:
            result = subprocess.check_output(
                [get_alg_base(algname), "--jsonlist-devices"]
            )
            devices = json.loads(result)["devices"]

            if len(devices) != 1:
                raise Exception(
                    "NVBench doesn't work well with multiple GPUs, use `CUDA_VISIBLE_DEVICES`"
                )

            self.device_cache[algname] = devices[0]

        return self.device_cache[algname]


def json_benches(algname):
    return JsonCache().get_bench(algname)


def device_json(algname):
    return JsonCache().get_device(algname)
