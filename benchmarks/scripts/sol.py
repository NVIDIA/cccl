#!/usr/bin/env python3

import argparse
import os
import re

import cccl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def is_finite(x):
    if isinstance(x, float):
        return x != np.inf and x != -np.inf
    return True


def filter_by_problem_size(df):
    min_elements_pow2 = 28
    if "Elements{io}[pow2]" in df.columns:
        df["Elements{io}[pow2]"] = df["Elements{io}[pow2]"].astype(int)
        df = df[df["Elements{io}[pow2]"] >= min_elements_pow2]
    return df


def filter_by_offset_type(df):
    if "OffsetT{ct}" in df.columns:
        df = df[(df["OffsetT{ct}"] == "I32") | (df["OffsetT{ct}"] == "U32")]
    return df


def filter_by_type(df):
    if "T{ct}" in df:
        # df = df[df['T{ct}'].str.contains('64')]
        df = df[~df["T{ct}"].str.contains("C")]
    elif "KeyT{ct}" in df:
        # df = df[df['KeyT{ct}'].str.contains('64')]
        df = df[~df["KeyT{ct}"].str.contains("C")]
    return df


def alg_dfs(files, alg_regex):
    pattern = re.compile(alg_regex)
    result = {}
    for file in files:
        storage = cccl.bench.SQLiteStorage(file)
        for algname in storage.algnames():
            if pattern.match(algname):
                for subbench in storage.subbenches(algname):
                    df = storage.alg_to_df(algname, subbench)
                    df = df.map(lambda x: x if is_finite(x) else np.nan)
                    df = df.dropna(subset=["center"], how="all")
                    df = filter_by_type(
                        filter_by_offset_type(filter_by_problem_size(df))
                    )
                    df = df.filter(items=["ctk", "cccl", "gpu", "variant", "bw"])
                    df["variant"] = df["variant"].astype(str)
                    df["bw"] = df["bw"] * 100
                    fused_algname = algname.replace("bench.", "") + "." + subbench
                    if fused_algname in result:
                        result[fused_algname] = pd.concat([result[fused_algname], df])
                    else:
                        result[fused_algname] = df
                    print(fused_algname)
    return result


def alg_bws(dfs, verbose):
    medians = None
    for algname in dfs:
        df = dfs[algname]
        df["alg"] = algname
        if df is None:
            medians = df
        else:
            medians = pd.concat([medians, df])
    # print more information if it's not unique across all runs or when requested (verbose)
    medians["hue"] = ""
    if verbose or medians["cccl"].unique().size > 1:
        medians["hue"] = medians["hue"] + "CCCL " + medians["cccl"].astype(str) + " "
    gpuname = (
        medians["gpu"]
        if verbose
        else medians["gpu"].astype(str).map(lambda x: x[: x.find("(") - 1])
    )
    medians["hue"] = medians["hue"] + gpuname + " "
    if medians["variant"].unique().size > 1:
        variant = (
            medians["variant"]
            .astype(str)
            .map(lambda x: (" " + x if x != "base" else ""))
        )
        medians["hue"] = medians["hue"] + variant + " "
    if verbose or medians["ctk"].unique().size > 1:
        medians["hue"] = medians["hue"] + "CTK " + medians["ctk"].astype(str)
    return medians.drop(columns=["ctk", "cccl", "gpu", "variant"])


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def plot_sol(medians, box):
    if box:
        ax = sns.boxenplot(data=medians, x="alg", y="bw", hue="hue")
    else:
        ax = sns.barplot(
            data=medians,
            x="alg",
            y="bw",
            hue="hue",
            errorbar=lambda x: (x.min(), x.max()),
        )
        ax.bar_label(ax.containers[0], fmt="%.1f")
        for container in ax.containers[1:]:
            labels = [
                f"{c:.1f}\n({(c / f) * 100:.0f}%)"
                for f, c in zip(ax.containers[0].datavalues, container.datavalues)
            ]
            ax.bar_label(container, labels=labels)

    ax.legend(title=None)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Bandwidth (%SOL)")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=30, rotation_mode="anchor", ha="right"
    )
    ax.set_ylim([0, 100])
    plt.show()


def print_speedup(medians):
    m = medians.groupby(["alg", "hue"], sort=False).mean()
    m["speedup"] = m["bw"] / m.groupby(["alg"])["bw"].transform("first")
    print("# Speedups:")
    print()
    print(m.drop(columns="bw").sort_values(by="speedup", ascending=False).to_markdown())


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        "files", type=file_exists, nargs="+", help="At least one file is required."
    )
    parser.add_argument("--box", action="store_true", help="Plot box instead of bar.")
    parser.add_argument("-v", action="store_true", help="Verbose legend.")
    parser.add_argument(
        "-R", type=str, default=".*", help="Regex for benchmarks selection."
    )
    return parser.parse_args()


def sol():
    args = parse_args()
    medians = alg_bws(alg_dfs(args.files, args.R), args.v)
    print_speedup(medians)
    plot_sol(medians, args.box)


if __name__ == "__main__":
    sol()
