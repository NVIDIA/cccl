#!/usr/bin/env python3
# Copyright (c) 2018 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
r"""Summarize the last ninja build, invoked with ninja's -C syntax.

This script is designed to be automatically run after each ninja build in
order to summarize the build's performance. Making build performance information
more visible should make it easier to notice anomalies and opportunities. To use
this script on Windows just set NINJA_SUMMARIZE_BUILD=1 and run autoninja.bat.

On Linux you can get autoninja to invoke this script using this syntax:

$ NINJA_SUMMARIZE_BUILD=1 autoninja -C out/Default/ chrome

You can also call this script directly using ninja's syntax to specify the
output directory of interest:

> python3 post_build_ninja_summary.py -C out/Default

Typical output looks like this:

>ninja -C out\debug_component base
ninja.exe -C out\debug_component base -j 960 -l 48  -d keeprsp
ninja: Entering directory `out\debug_component'
[1 processes, 1/1 @ 0.3/s : 3.092s ] Regenerating ninja files
Longest build steps:
       0.1 weighted s to build obj/base/base/trace_log.obj (6.7 s elapsed time)
       0.2 weighted s to build nasm.exe, nasm.exe.pdb (0.2 s elapsed time)
       0.3 weighted s to build obj/base/base/win_util.obj (12.4 s elapsed time)
       1.2 weighted s to build base.dll, base.dll.lib (1.2 s elapsed time)
Time by build-step type:
       0.0 s weighted time to generate 6 .lib files (0.3 s elapsed time sum)
       0.1 s weighted time to generate 25 .stamp files (1.2 s elapsed time sum)
       0.2 s weighted time to generate 20 .o files (2.8 s elapsed time sum)
       1.7 s weighted time to generate 4 PEFile (linking) files (2.0 s elapsed
time sum)
      23.9 s weighted time to generate 770 .obj files (974.8 s elapsed time sum)
26.1 s weighted time (982.9 s elapsed time sum, 37.7x parallelism)
839 build steps completed, average of 32.17/s

If no gn clean has been done then results will be for the last non-NULL
invocation of ninja. Ideas for future statistics, and implementations are
appreciated.

The "weighted" time is the elapsed time of each build step divided by the number
of tasks that were running in parallel. This makes it an excellent approximation
of how "important" a slow step was. A link that is entirely or mostly serialized
will have a weighted time that is the same or similar to its elapsed time. A
compile that runs in parallel with 999 other compiles will have a weighted time
that is tiny."""

import argparse
import errno
import fnmatch
import os
import subprocess
import sys

# The number of long build times to report:
long_count = 10
# The number of long times by extension to report
long_ext_count = 10


class Target:
    """Represents a single line read for a .ninja_log file."""
    def __init__(self, start, end):
        """Creates a target object by passing in the start/end times in seconds
        as a float."""
        self.start = start
        self.end = end
        # A list of targets, appended to by the owner of this object.
        self.targets = []
        self.weighted_duration = 0.0

    def Duration(self):
        """Returns the task duration in seconds as a float."""
        return self.end - self.start

    def SetWeightedDuration(self, weighted_duration):
        """Sets the duration, in seconds, passed in as a float."""
        self.weighted_duration = weighted_duration

    def WeightedDuration(self):
        """Returns the task's weighted duration in seconds as a float.

        Weighted_duration takes the elapsed time of the task and divides it
        by how many other tasks were running at the same time. Thus, it
        represents the approximate impact of this task on the total build time,
        with serialized or serializing steps typically ending up with much
        longer weighted durations.
        weighted_duration should always be the same or shorter than duration.
        """
        # Allow for modest floating-point errors
        epsilon = 0.000002
        if (self.weighted_duration > self.Duration() + epsilon):
            print('%s > %s?' % (self.weighted_duration, self.Duration()))
        assert (self.weighted_duration <= self.Duration() + epsilon)
        return self.weighted_duration

    def DescribeTargets(self):
        """Returns a printable string that summarizes the targets."""
        # Some build steps generate dozens of outputs - handle them sanely.
        # The max_length was chosen so that it can fit most of the long
        # single-target names, while minimizing word wrapping.
        result = ', '.join(self.targets)
        max_length = 65
        if len(result) > max_length:
            result = result[:max_length] + '...'
        return result


# Copied with some modifications from ninjatracing
def ReadTargets(log, show_all):
    """Reads all targets from .ninja_log file |log_file|, sorted by duration.

    The result is a list of Target objects."""
    header = log.readline()
    # Handle empty ninja_log gracefully by silently returning an empty list of
    # targets.
    if not header:
        return []
    assert header == '# ninja log v5\n', \
           'unrecognized ninja log version %r' % header
    targets_dict = {}
    last_end_seen = 0.0
    for line in log:
        parts = line.strip().split('\t')
        if len(parts) != 5:
            # If ninja.exe is rudely halted then the .ninja_log file may be
            # corrupt. Silently continue.
            continue
        start, end, _, name, cmdhash = parts  # Ignore restat.
        # Convert from integral milliseconds to float seconds.
        start = int(start) / 1000.0
        end = int(end) / 1000.0
        if not show_all and end < last_end_seen:
            # An earlier time stamp means that this step is the first in a new
            # build, possibly an incremental build. Throw away the previous
            # data so that this new build will be displayed independently.
            # This has to be done by comparing end times because records are
            # written to the .ninja_log file when commands complete, so end
            # times are guaranteed to be in order, but start times are not.
            targets_dict = {}
        target = None
        if cmdhash in targets_dict:
            target = targets_dict[cmdhash]
            if not show_all and (target.start != start or target.end != end):
                # If several builds in a row just run one or two build steps
                # then the end times may not go backwards so the last build may
                # not be detected as such. However in many cases there will be a
                # build step repeated in the two builds and the changed
                # start/stop points for that command, identified by the hash,
                # can be used to detect and reset the target dictionary.
                targets_dict = {}
                target = None
        if not target:
            targets_dict[cmdhash] = target = Target(start, end)
        last_end_seen = end
        target.targets.append(name)
    return list(targets_dict.values())


def GetExtension(target, extra_patterns):
    """Return the file extension that best represents a target.

  For targets that generate multiple outputs it is important to return a
  consistent 'canonical' extension. Ultimately the goal is to group build steps
  by type."""
    for output in target.targets:
        if extra_patterns:
            for fn_pattern in extra_patterns.split(';'):
                if fnmatch.fnmatch(output, '*' + fn_pattern + '*'):
                    return fn_pattern
        # Not a true extension, but a good grouping.
        if output.endswith('type_mappings'):
            extension = 'type_mappings'
            break

        # Capture two extensions if present. For example: file.javac.jar should
        # be distinguished from file.interface.jar.
        root, ext1 = os.path.splitext(output)
        _, ext2 = os.path.splitext(root)
        extension = ext2 + ext1  # Preserve the order in the file name.

        if len(extension) == 0:
            extension = '(no extension found)'

        if ext1 in ['.pdb', '.dll', '.exe']:
            extension = 'PEFile (linking)'
            # Make sure that .dll and .exe are grouped together and that the
            # .dll.lib files don't cause these to be listed as libraries
            break
        if ext1 in ['.so', '.TOC']:
            extension = '.so (linking)'
            # Attempt to identify linking, avoid identifying as '.TOC'
            break
        # Make sure .obj files don't get categorized as mojo files
        if ext1 in ['.obj', '.o']:
            break
        # Jars are the canonical output of java targets.
        if ext1 == '.jar':
            break
        # Normalize all mojo related outputs to 'mojo'.
        if output.count('.mojom') > 0:
            extension = 'mojo'
            break
    return extension


def SummarizeEntries(entries, extra_step_types, elapsed_time_sorting):
    """Print a summary of the passed in list of Target objects."""

    # Create a list that is in order by time stamp and has entries for the
    # beginning and ending of each build step (one time stamp may have multiple
    # entries due to multiple steps starting/stopping at exactly the same time).
    # Iterate through this list, keeping track of which tasks are running at all
    # times. At each time step calculate a running total for weighted time so
    # that when each task ends its own weighted time can easily be calculated.
    task_start_stop_times = []

    earliest = -1
    latest = 0
    total_cpu_time = 0
    for target in entries:
        if earliest < 0 or target.start < earliest:
            earliest = target.start
        if target.end > latest:
            latest = target.end
        total_cpu_time += target.Duration()
        task_start_stop_times.append((target.start, 'start', target))
        task_start_stop_times.append((target.end, 'stop', target))
    length = latest - earliest
    weighted_total = 0.0

    # Sort by the time/type records and ignore |target|
    task_start_stop_times.sort(key=lambda times: times[:2])
    # Now we have all task start/stop times sorted by when they happen. If a
    # task starts and stops on the same time stamp then the start will come
    # first because of the alphabet, which is important for making this work
    # correctly.
    # Track the tasks which are currently running.
    running_tasks = {}
    # Record the time we have processed up to so we know how to calculate time
    # deltas.
    last_time = task_start_stop_times[0][0]
    # Track the accumulated weighted time so that it can efficiently be added
    # to individual tasks.
    last_weighted_time = 0.0
    # Scan all start/stop events.
    for event in task_start_stop_times:
        time, action_name, target = event
        # Accumulate weighted time up to now.
        num_running = len(running_tasks)
        if num_running > 0:
            # Update the total weighted time up to this moment.
            last_weighted_time += (time - last_time) / float(num_running)
        if action_name == 'start':
            # Record the total weighted task time when this task starts.
            running_tasks[target] = last_weighted_time
        if action_name == 'stop':
            # Record the change in the total weighted task time while this task
            # ran.
            weighted_duration = last_weighted_time - running_tasks[target]
            target.SetWeightedDuration(weighted_duration)
            weighted_total += weighted_duration
            del running_tasks[target]
        last_time = time
    assert (len(running_tasks) == 0)

    # Warn if the sum of weighted times is off by more than half a second.
    if abs(length - weighted_total) > 500:
        print('Warning: Possible corrupt ninja log, results may be '
              'untrustworthy. Length = %.3f, weighted total = %.3f' %
              (length, weighted_total))

    # Print the slowest build steps:
    print('    Longest build steps:')
    if elapsed_time_sorting:
        entries.sort(key=lambda x: x.Duration())
    else:
        entries.sort(key=lambda x: x.WeightedDuration())
    for target in entries[-long_count:]:
        print('      %8.1f weighted s to build %s (%.1f s elapsed time)' %
              (target.WeightedDuration(), target.DescribeTargets(),
               target.Duration()))

    # Sum up the time by file extension/type of the output file
    count_by_ext = {}
    time_by_ext = {}
    weighted_time_by_ext = {}
    # Scan through all of the targets to build up per-extension statistics.
    for target in entries:
        extension = GetExtension(target, extra_step_types)
        time_by_ext[extension] = time_by_ext.get(extension,
                                                 0) + target.Duration()
        weighted_time_by_ext[extension] = weighted_time_by_ext.get(
            extension, 0) + target.WeightedDuration()
        count_by_ext[extension] = count_by_ext.get(extension, 0) + 1

    print('    Time by build-step type:')
    # Copy to a list with extension name and total time swapped, to (time, ext)
    if elapsed_time_sorting:
        weighted_time_by_ext_sorted = sorted(
            (y, x) for (x, y) in time_by_ext.items())
    else:
        weighted_time_by_ext_sorted = sorted(
            (y, x) for (x, y) in weighted_time_by_ext.items())
    # Print the slowest build target types:
    for time, extension in weighted_time_by_ext_sorted[-long_ext_count:]:
        print(
            '      %8.1f s weighted time to generate %d %s files '
            '(%1.1f s elapsed time sum)' %
            (time, count_by_ext[extension], extension, time_by_ext[extension]))

    print('    %.1f s weighted time (%.1f s elapsed time sum, %1.1fx '
          'parallelism)' %
          (length, total_cpu_time, total_cpu_time * 1.0 / length))
    print('    %d build steps completed, average of %1.2f/s' %
          (len(entries), len(entries) / (length)))


def main():
    log_file = '.ninja_log'
    metrics_file = 'siso_metrics.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', dest='build_directory', help='Build directory.')
    parser.add_argument(
        '-s',
        '--step-types',
        help='semicolon separated fnmatch patterns for build-step grouping')
    parser.add_argument(
        '-e',
        '--elapsed_time_sorting',
        default=False,
        action='store_true',
        help='Sort output by elapsed time instead of weighted time')
    parser.add_argument('--log-file',
                        help="specific ninja log file to analyze.")
    args, _extra_args = parser.parse_known_args()
    if args.build_directory:
        log_file = os.path.join(args.build_directory, log_file)
        metrics_file = os.path.join(args.build_directory, metrics_file)
    if args.log_file:
        log_file = args.log_file
    if not args.step_types:
        # Offer a convenient way to add extra step types automatically,
        # including when this script is run by autoninja. get() returns None if
        # the variable isn't set.
        args.step_types = os.environ.get('chromium_step_types')
    if args.step_types:
        # Make room for the extra build types.
        global long_ext_count
        long_ext_count += len(args.step_types.split(';'))

    if os.path.exists(metrics_file):
        # Automatically handle summarizing siso builds.
        cmd = ['siso.bat' if 'win32' in sys.platform else 'siso']
        cmd.extend(['metrics', 'summary'])
        if args.build_directory:
            cmd.extend(['-C', args.build_directory])
        if args.step_types:
            cmd.extend(['--step_types', args.step_types])
        if args.elapsed_time_sorting:
            cmd.append('--elapsed_time_sorting')
        subprocess.run(cmd)
    else:
        try:
            with open(log_file, 'r') as log:
                entries = ReadTargets(log, False)
                if entries:
                    SummarizeEntries(entries, args.step_types,
                                     args.elapsed_time_sorting)
        except IOError:
            print('Log file %r not found, no build summary created.' % log_file)
            return errno.ENOENT


if __name__ == '__main__':
    sys.exit(main())
