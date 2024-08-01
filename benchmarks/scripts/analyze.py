#!/usr/bin/env python3

import os
import re
import json
import cccl
import math
import argparse
import itertools
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import hdquantiles

pd.options.display.max_colwidth = 100

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = itertools.cycle(default_colors)
color_map = {}

precision = 0.01
sensitivity = 0.5


def get_bench_columns():
    return ['variant', 'elapsed', 'center', 'samples', 'bw']


def get_extended_bench_columns():
    return get_bench_columns() + ['speedup', 'base_samples']


def compute_speedup(df):
    bench_columns = get_bench_columns()
    workload_columns = [col for col in df.columns if col not in bench_columns]
    base_df = df[df['variant'] == 'base'].drop(columns=['variant']).rename(
        columns={'center': 'base_center', 'samples': 'base_samples'})
    base_df.drop(columns=['elapsed', 'bw'], inplace=True)

    merged_df = df.merge(
        base_df, on=[col for col in df.columns if col in workload_columns])
    merged_df['speedup'] = merged_df['base_center'] / merged_df['center']
    merged_df = merged_df.drop(columns=['base_center'])
    return merged_df


def get_ct_axes(df):
    ct_axes = []
    for col in df.columns:
        if '{ct}' in col:
            ct_axes.append(col)

    return ct_axes


def get_rt_axes(df):
    rt_axes = []
    excluded_columns = get_ct_axes(df) + get_extended_bench_columns()

    for col in df.columns:
        if col not in excluded_columns:
            rt_axes.append(col)

    return rt_axes


def ct_space(df):
    ct_axes = get_ct_axes(df)

    unique_ct_combinations = []
    for _, row in df[ct_axes].drop_duplicates().iterrows():
        unique_ct_combinations.append({})
        for col in ct_axes:
            unique_ct_combinations[-1][col] = row[col]

    return unique_ct_combinations


def extract_case(df, ct_point):
    tuning_df_loc = None

    for ct_axis in ct_point:
        if tuning_df_loc is None:
            tuning_df_loc = (df[ct_axis] == ct_point[ct_axis])
        else:
            tuning_df_loc = tuning_df_loc & (df[ct_axis] == ct_point[ct_axis])

    tuning_df = df.loc[tuning_df_loc].copy()
    for ct_axis in ct_point:
        tuning_df.drop(columns=[ct_axis], inplace=True)

    return tuning_df


def extract_rt_axes_values(df):
    rt_axes = get_rt_axes(df)
    rt_axes_values = {}

    for rt_axis in rt_axes:
        rt_axes_values[rt_axis] = list(df[rt_axis].unique())

    return rt_axes_values


def extract_rt_space(df):
    rt_axes = get_rt_axes(df)
    rt_axes_values = []
    for rt_axis in rt_axes:
        values = df[rt_axis].unique()
        rt_axes_values.append(["{}={}".format(rt_axis, v) for v in values])
    return list(itertools.product(*rt_axes_values))


def filter_variants(df, group):
    rt_axes = get_rt_axes(df)
    unique_combinations = set(
        df[rt_axes].drop_duplicates().itertuples(index=False))
    group_combinations = set(
        group[rt_axes].drop_duplicates().itertuples(index=False))
    has_all_combinations = group_combinations == unique_combinations
    return has_all_combinations


def extract_complete_variants(df):
    return df.groupby('variant').filter(functools.partial(filter_variants, df))


def compute_workload_score(rt_axes_values, rt_axes_ids, weights, row):
    rt_workload = []
    for rt_axis in rt_axes_values:
        rt_workload.append("{}={}".format(rt_axis, row[rt_axis]))

    weight = cccl.bench.get_workload_weight(rt_workload, rt_axes_values, rt_axes_ids, weights)
    return row['speedup'] * weight


def compute_variant_score(rt_axes_values, rt_axes_ids, weight_matrix, group):
    workload_score_closure = functools.partial(compute_workload_score, rt_axes_values, rt_axes_ids, weight_matrix)
    score_sum = group.apply(workload_score_closure, axis=1).sum()
    return score_sum


def extract_scores(dfs):
    rt_axes_values = {}
    for subbench in dfs:
        rt_axes_values[subbench] = extract_rt_axes_values(dfs[subbench])

    rt_axes_ids = cccl.bench.compute_axes_ids(rt_axes_values)
    weights = cccl.bench.compute_weight_matrices(rt_axes_values, rt_axes_ids)

    score_dfs = []
    for subbench in dfs:
        score_closure = functools.partial(
            compute_variant_score, rt_axes_values[subbench], rt_axes_ids[subbench], weights[subbench])
        grouped = dfs[subbench].groupby('variant')
        scores = grouped.apply(score_closure).reset_index()
        scores.columns = ['variant', 'score']
        stat = grouped.agg(mins=('speedup', 'min'),
                           means=('speedup', 'mean'),
                           maxs=('speedup', 'max'))
        scores = pd.merge(scores, stat, on='variant')
        score_dfs.append(scores)
    score_df = pd.concat(score_dfs)
    result = score_df.groupby('variant').agg(
        {'score': 'sum', 'mins': 'min', 'means': 'mean', 'maxs': 'max'}).reset_index()
    return result.sort_values(by=['score'], ascending=False)


def distributions_are_different(alpha, row):
    ref_samples = row['base_samples']
    cmp_samples = row['samples']

    # H0: the distributions are not different
    # H1: the distribution are different
    _, p = mannwhitneyu(ref_samples, cmp_samples)

    # Reject H0
    return p < alpha


def remove_matching_distributions(alpha, df):
    closure = functools.partial(distributions_are_different, alpha)
    return df[df.apply(closure, axis=1)]


def get_filenames_map(arr):
    if not arr:
        return []

    prefix = arr[0]
    for string in arr:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break

    return {string: string[len(prefix):] for string in arr}


def is_finite(x):
    if isinstance(x, float):
        return x != np.inf and x != -np.inf
    return True


def iterate_case_dfs(args, callable):
    storages = {}
    algnames = set()
    filenames_map = get_filenames_map(args.files)
    for file in args.files:
        storage = cccl.bench.StorageBase(file)
        algnames.update(storage.algnames())
        storages[filenames_map[file]] = storage

    pattern = re.compile(args.R)

    exact_values = {}
    if args.args:
        for value in args.args:
            name, val = value.split('=')
            exact_values[name] = val

    for algname in algnames:
        if not pattern.match(algname):
            continue

        case_dfs = {}
        for subbench in storage.subbenches(algname):
            for file in storages:
                storage = storages[file]
                df = storage.alg_to_df(algname, subbench)

                df = df.map(lambda x: x if is_finite(x) else np.nan)
                df = df.dropna(subset=['center'], how='all')

                for _, row in df[['ctk', 'cccl']].drop_duplicates().iterrows():
                    ctk_version = row['ctk']
                    cccl_version = row['cccl']
                    ctk_cub_df = df[(df['ctk'] == ctk_version) &
                                    (df['cccl'] == cccl_version)]

                    for gpu in ctk_cub_df['gpu'].unique():
                        target_df = ctk_cub_df[ctk_cub_df['gpu'] == gpu]
                        target_df = target_df.drop(columns=['ctk', 'cccl', 'gpu'])
                        target_df = compute_speedup(target_df)

                        for key in exact_values:
                            if key in target_df.columns:
                                target_df = target_df[target_df[key] == exact_values[key]]

                        for ct_point in ct_space(target_df):
                            point_str = ", ".join(["{}={}".format(k, ct_point[k]) for k in ct_point])
                            case_df = extract_complete_variants(extract_case(target_df, ct_point))
                            case_df['variant'] = case_df['variant'].astype(str) + " ({})".format(file)
                            if point_str not in case_dfs:
                                case_dfs[point_str] = {}
                            if subbench not in case_dfs[point_str]:
                                case_dfs[point_str][subbench] = case_df
                            else:
                                case_dfs[point_str][subbench] = pd.concat([case_dfs[point_str][subbench], case_df])

        for point_str in case_dfs:
            callable(algname, point_str, case_dfs[point_str])


def case_top(alpha, N, algname, ct_point_name, case_dfs):
    print("{}[{}]:".format(algname, ct_point_name))

    if alpha < 1.0:
        case_df = remove_matching_distributions(alpha, case_df)

    for subbench in case_dfs:
        case_dfs[subbench] = extract_complete_variants(case_dfs[subbench])
    with pd.option_context('display.max_rows', None):
        print(extract_scores(case_dfs).head(N))


def top(args):
    iterate_case_dfs(args, functools.partial(case_top, args.alpha, args.top))


def case_coverage(algname, ct_point_name, case_dfs):
    num_variants = cccl.bench.Config().variant_space_size(algname)
    min_coverage = 100.0
    for subbench in case_dfs:
        num_covered_variants = len(case_dfs[subbench]['variant'].unique())
        coverage = (num_covered_variants / num_variants) * 100
        min_coverage = min(min_coverage, coverage)
    case_str = "{}[{}]".format(algname, ct_point_name)
    print("{} coverage: {} / {} ({:.4f}%)".format(
          case_str, num_covered_variants, num_variants, min_coverage))


def coverage(args):
    iterate_case_dfs(args, case_coverage)


def parallel_coordinates_plot(df, title):
    # Parallel coordinates plot adaptation of https://stackoverflow.com/a/69411450
    import matplotlib.cm as cm
    from matplotlib.path import Path
    import matplotlib.patches as patches

    # Variables (the first variable must be categoric):
    my_vars = df.columns.tolist()
    df_plot = df[my_vars]
    df_plot = df_plot.dropna()
    df_plot = df_plot.reset_index(drop=True)

    # Convert to numeric matrix:
    ym = []
    dics_vars = []
    for v, var in enumerate(my_vars):
        if df_plot[var].dtype.kind not in ["i", "u", "f"]:
            dic_var = dict([(val, c)
                           for c, val in enumerate(df_plot[var].unique())])
            dics_vars += [dic_var]
            ym += [[dic_var[i] for i in df_plot[var].tolist()]]
        else:
            ym += [df_plot[var].tolist()]
    ym = np.array(ym).T

    # Padding:
    ymins = ym.min(axis=0)
    ymaxs = ym.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys*0.05
    ymaxs += dys*0.05

    dys = ymaxs - ymins

    # Adjust to the main axis:
    zs = np.zeros_like(ym)
    zs[:, 0] = ym[:, 0]
    zs[:, 1:] = (ym[:, 1:] - ymins[1:])/dys[1:]*dys[0] + ymins[0]

    # Plot:
    fig, host_ax = plt.subplots(figsize=(20, 10), tight_layout=True)

    # Make the axes:
    axes = [host_ax] + [host_ax.twinx() for i in range(ym.shape[1] - 1)]
    dic_count = 0
    for i, ax in enumerate(axes):
        ax.set_ylim(
            bottom=ymins[i],
            top=ymaxs[i]
        )
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.ticklabel_format(style='plain')
        if ax != host_ax:
            ax.spines.left.set_visible(False)
            ax.yaxis.set_ticks_position("right")
            ax.spines.right.set_position(("axes", i/(ym.shape[1] - 1)))
        if df_plot.iloc[:, i].dtype.kind not in ["i", "u", "f"]:
            dic_var_i = dics_vars[dic_count]
            ax.set_yticks(range(len(dic_var_i)))
            if i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels([key_val for key_val in dics_vars[dic_count].keys()])
            dic_count += 1
    host_ax.set_xlim(left=0, right=ym.shape[1] - 1)
    host_ax.set_xticks(range(ym.shape[1]))
    host_ax.set_xticklabels(my_vars, fontsize=14)
    host_ax.tick_params(axis="x", which="major", pad=7)

    # Color map:
    colormap = cm.get_cmap('turbo')

    # Normalize speedups:
    df["speedup_normalized"] = (
        df["speedup"] - df["speedup"].min()) / (df["speedup"].max() - df["speedup"].min())

    # Make the curves:
    host_ax.spines.right.set_visible(False)
    host_ax.xaxis.tick_top()
    for j in range(ym.shape[0]):
        verts = list(zip([x for x in np.linspace(0, len(ym) - 1, len(ym)*3 - 2,
                                                 endpoint=True)],
                     np.repeat(zs[j, :], 3)[1: -1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        color_first_cat_var = colormap(df.loc[j, "speedup_normalized"])
        patch = patches.PathPatch(
            path, facecolor="none", lw=2, alpha=0.05, edgecolor=color_first_cat_var)
        host_ax.add_patch(patch)

    host_ax.set_title(title)
    plt.show()



def case_coverage_plot(algname, ct_point_name, case_dfs):
    data_list = []

    for subbench in case_dfs:
        for _, row_description in case_dfs[subbench].iterrows():
            variant = row_description['variant']
            speedup = row_description['speedup']

            if variant.startswith('base'):
                continue

            varname, _ = variant.split(' ')
            params = varname.split('.')
            data_dict = {'variant': variant}

            for param in params:
                print(variant)
                name, val = param.split('_')
                data_dict[name] = int(val)

            data_dict['speedup'] = speedup
            # data_dict['variant'] = variant
            data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    parallel_coordinates_plot(df, "{} ({})".format(algname, ct_point_name))


def coverage_plot(args):
    iterate_case_dfs(args, case_coverage_plot)


def case_pair_plot(algname, ct_point_name, case_dfs):
    import seaborn as sns
    data_list = []

    for subbench in case_dfs:
        for _, row_description in case_dfs[subbench].iterrows():
            variant = row_description['variant']
            speedup = row_description['speedup']

            if variant.startswith('base'):
                continue

            varname, _ = variant.split(' ')
            params = varname.split('.')
            data_dict = {}

            for param in params:
                print(variant)
                name, val = param.split('_')
                data_dict[name] = int(val)

            data_dict['speedup'] = speedup
            data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    sns.pairplot(df, hue='speedup')
    plt.title("{} ({})".format(algname, ct_point_name))
    plt.show()


def pair_plot(args):
    iterate_case_dfs(args, case_pair_plot)


def qrde_hd(samples):
    """
    Computes quantile-respectful density estimation based on the Harrell-Davis
    quantile estimator. The implementation is based on the following post:
    https://aakinshin.net/posts/qrde-hd by Andrey Akinshin
    """
    min_sample, max_sample = min(samples), max(samples)
    num_quantiles = math.ceil(1.0 / precision)
    quantiles = np.linspace(precision, 1 - precision, num_quantiles - 1)
    hd_quantiles = [min_sample] + list(hdquantiles(samples, quantiles)) + [max_sample]
    width = [hd_quantiles[idx + 1] - hd_quantiles[idx] for idx in range(num_quantiles)]
    p = 1.0 / precision
    height = [1.0 / (p * w) for w in width]
    return width, height


def extract_peaks(pdf):
    peaks = []
    for i in range(1, len(pdf) - 1):
        if pdf[i - 1] < pdf[i] > pdf[i + 1]:
            peaks.append(i)
    return peaks


def extract_modes(samples):
    """
    Extract modes from the given samples based on the lowland algorithm:
    https://aakinshin.net/posts/lowland-multimodality-detection/ by Andrey Akinshin
    Implementation is based on the https://github.com/AndreyAkinshin/perfolizer
    LowlandModalityDetector class.
    """
    mode_ids = []

    widths, heights = qrde_hd(samples)
    peak_ids = extract_peaks(heights)
    bin_area = 1.0 / len(heights)

    x = min(samples)
    peak_xs = []
    peak_ys = []
    bin_lower = [x]
    for idx in range(len(heights)):
        if idx in peak_ids:
            peak_ys.append(heights[idx])
            peak_xs.append(x + widths[idx] / 2)
        x += widths[idx]
        bin_lower.append(x)

    def lowland_between(mode_candidate, left_peak, right_peak):
        left, right = left_peak, right_peak
        min_height = min(heights[left_peak], heights[right_peak])
        while left < right and heights[left] > min_height:
            left += 1
        while left < right and heights[right] > min_height:
            right -= 1

        width = bin_lower[right + 1] - bin_lower[left]
        total_area = width * min_height
        total_bin_area = (right - left + 1) * bin_area

        if total_bin_area / total_area < sensitivity:
            mode_ids.append(mode_candidate)
            return True
        return False

    previousPeaks = [peak_ids[0]]
    for i in range(1, len(peak_ids)):
        currentPeak = peak_ids[i]
        while previousPeaks and heights[previousPeaks[-1]] < heights[currentPeak]:
            if lowland_between(previousPeaks[0], previousPeaks[-1], currentPeak):
                previousPeaks = []
            else:
                previousPeaks.pop()

        if previousPeaks and heights[previousPeaks[-1]] > heights[currentPeak]:
            if lowland_between(previousPeaks[0], previousPeaks[-1], currentPeak):
                previousPeaks = []

        previousPeaks.append(currentPeak)

    mode_ids.append(previousPeaks[0])
    return mode_ids


def hd_displot(samples, label, ax):
    if label not in color_map:
        color_map[label] = next(color_cycle)
    color = color_map[label]
    widths, heights = qrde_hd(samples)
    mode_ids = extract_modes(samples)

    min_sample, max_sample = min(samples), max(samples)

    xs = [min_sample]
    ys = [0]

    peak_xs = []
    peak_ys = []

    x = min(samples)
    for idx in range(len(widths)):
        xs.append(x + widths[idx] / 2)
        ys.append(heights[idx])
        if idx in mode_ids:
            peak_ys.append(heights[idx])
            peak_xs.append(x + widths[idx] / 2)
        x += widths[idx]

    xs = xs + [max_sample]
    ys = ys + [0]

    ax.fill_between(xs, ys, 0, alpha=0.4, color=color)

    quartiles_of_interest = [0.25, 0.5, 0.75]

    for quartile in quartiles_of_interest:
        bin = int(quartile / precision) + 1
        ax.plot([xs[bin], xs[bin]], [0, ys[bin]], color=color)

    ax.plot(xs, ys, label=label, color=color)
    ax.plot(peak_xs, peak_ys, 'o', color=color)
    ax.legend()


def displot(data, ax):
    for variant in data:
        hd_displot(data[variant], variant, ax)


def variant_ratio(data, variant, ax):
    if variant not in color_map:
        color_map[variant] = next(color_cycle)
    color = color_map[variant]

    variant_samples = data[variant]
    base_samples = data['base']

    variant_widths, variant_heights = qrde_hd(variant_samples)
    base_widths, base_heights = qrde_hd(base_samples)

    quantiles = []
    ratios = []

    base_x = min(base_samples)
    variant_x = min(variant_samples)

    for i in range(1, len(variant_heights) - 1):
        base_x += base_widths[i] / 2
        variant_x += variant_widths[i] / 2
        quantiles.append(i * precision)
        ratios.append(base_x / variant_x)

    ax.plot(quantiles, ratios, label=variant, color=color)
    ax.axhline(1, color='red', alpha=0.7)
    ax.legend()
    ax.tick_params(axis='both', direction='in', pad=-22)


def ratio(data, ax):
    for variant in data:
        if variant != 'base':
            variant_ratio(data, variant, ax)


def case_variants(pattern, mode, algname, ct_point_name, case_dfs):
    for subbench in case_dfs:
        case_df = case_dfs[subbench]
        title = "{}[{}]:".format(algname + '/' + subbench, ct_point_name)
        df = case_df[case_df['variant'].str.contains(pattern, regex=True)].reset_index(drop=True)
        rt_axes = get_rt_axes(df)
        rt_axes_values = extract_rt_axes_values(df)

        vertical_axis_name = rt_axes[0]
        if 'Elements{io}[pow2]' in rt_axes:
            vertical_axis_name = 'Elements{io}[pow2]'
        horizontal_axes = rt_axes
        horizontal_axes.remove(vertical_axis_name)
        vertical_axis_values = rt_axes_values[vertical_axis_name]

        vertical_axis_ids = {}
        for idx, val in enumerate(vertical_axis_values):
            vertical_axis_ids[val] = idx

        def extract_horizontal_space(df):
            values = []
            for rt_axis in horizontal_axes:
                values.append(["{}={}".format(rt_axis, v) for v in df[rt_axis].unique()])
            return list(itertools.product(*values))

        if len(horizontal_axes) > 0:
            idx = 0
            horizontal_axis_ids = {}
            for point in extract_horizontal_space(df):
                horizontal_axis_ids[" / ".join(point)] = idx
                idx = idx + 1

        num_rows = len(vertical_axis_ids)
        num_cols = max(1, len(extract_horizontal_space(df)))

        if num_rows == 0:
            return

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, gridspec_kw = {'wspace': 0, 'hspace': 0})


        for _, vertical_row_description in df[[vertical_axis_name]].drop_duplicates().iterrows():
            vertical_val = vertical_row_description[vertical_axis_name]
            vertical_id = vertical_axis_ids[vertical_val]
            vertical_name = "{}={}".format(vertical_axis_name, vertical_val)

            vertical_df = df[df[vertical_axis_name] == vertical_val]

            for _, horizontal_row_description in vertical_df[horizontal_axes].drop_duplicates().iterrows():
                horizontal_df = vertical_df

                for axis in horizontal_axes:
                    horizontal_df = horizontal_df[horizontal_df[axis] == horizontal_row_description[axis]]

                horizontal_id = 0

                if len(horizontal_axes) > 0:
                    horizontal_point = []
                    for rt_axis in horizontal_axes:
                        horizontal_point.append("{}={}".format(rt_axis, horizontal_row_description[rt_axis]))
                    horizontal_name = " / ".join(horizontal_point)
                    horizontal_id = horizontal_axis_ids[horizontal_name]
                    ax=axes[vertical_id, horizontal_id]
                else:
                    ax=axes[vertical_id]
                    ax.set_ylabel(vertical_name)

                data = {}
                for _, variant in horizontal_df[['variant']].drop_duplicates().iterrows():
                    variant_name = variant['variant']
                    if 'base' not in data:
                        data['base'] = horizontal_df[horizontal_df['variant'] == variant_name].iloc[0]['base_samples']
                    data[variant_name] = horizontal_df[horizontal_df['variant'] == variant_name].iloc[0]['samples']

                if mode == 'pdf':
                    # sns.histplot(data=data, ax=ax, kde=True)
                    displot(data, ax)
                else:
                    ratio(data, ax)

                if len(horizontal_axes) > 0:
                    ax=axes[vertical_id, horizontal_id]
                    if vertical_id == (num_rows - 1):
                        ax.set_xlabel(horizontal_name)
                    if horizontal_id == 0:
                        ax.set_ylabel(vertical_name)
                    else:
                        ax.set_ylabel('')

        for ax in axes.flat:
            ax.set_xticklabels([])

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()



def variants(args, mode):
    pattern = re.compile(args.variants_pdf) if mode == 'pdf' else re.compile(args.variants_ratio)
    iterate_case_dfs(args, functools.partial(case_variants, pattern, mode))


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def case_offload(algname, ct_point_name, case_dfs):
    for subbench in case_dfs:
        df = case_dfs[subbench]
        for rt_point in extract_rt_space(df):
            point_df = df
            for rt_kv in rt_point:
                key, value = rt_kv.split('=')
                point_df = point_df[point_df[key] == value]
            point_name = ct_point_name + " " + " ".join(rt_point)
            point_name = point_name.replace(',', '')
            bench_name = "{}.{}-{}".format(algname, subbench, point_name)
            bench_name = bench_name.replace(' ', '___')
            bench_name = "".join(c if c.isalnum() else "_" for c in bench_name)
            with open(bench_name + '.json', 'w') as f:
                obj = json.loads(point_df.to_json(orient='records'))
                json.dump(obj, f, indent=2)


def offload(args):
    iterate_case_dfs(args, case_offload)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        '-R', type=str, default='.*', help="Regex for benchmarks selection.")
    parser.add_argument(
        '--list-benches', action=argparse.BooleanOptionalAction, help="Show available benchmarks.")
    parser.add_argument(
        '--coverage', action=argparse.BooleanOptionalAction, help="Show variant space coverage.")
    parser.add_argument(
        '--coverage-plot', action=argparse.BooleanOptionalAction, help="Plot variant space coverage.")
    parser.add_argument(
        '--pair-plot', action=argparse.BooleanOptionalAction, help="Pair plot.")
    parser.add_argument(
        '--top', default=7, type=int, action='store', nargs='?', help="Show top N variants with highest score.")
    parser.add_argument(
        'files', type=file_exists, nargs='+', help='At least one file is required.')
    parser.add_argument(
        '--alpha', default=1.0, type=float)
    parser.add_argument(
        '--variants-pdf', type=str, help="Show matching variants data.")
    parser.add_argument(
        '--variants-ratio', type=str, help="Show matching variants data.")
    parser.add_argument('-a', '--args', action='append',
                        type=str, help="Parameter in the format `Param=Value`.")
    parser.add_argument(
        '-o', '--offload', action=argparse.BooleanOptionalAction, help="Offload samples")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.list_benches:
        cccl.bench.list_benches()
        return

    if args.coverage:
        coverage(args)
        return

    if args.coverage_plot:
        coverage_plot(args)
        return

    if args.pair_plot:
        pair_plot(args)
        return

    if args.variants_pdf:
        variants(args, 'pdf')
        return

    if args.variants_ratio:
        variants(args, 'ratio')
        return

    if args.offload:
        offload(args)
        return

    top(args)


if __name__ == "__main__":
    main()
