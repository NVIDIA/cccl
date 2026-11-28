// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Collects the evidence that the AI triage reads about a workflow run's failed jobs.
//
// Inputs (environment variables):
//   RUN_ID      the workflow run to examine
//   PR_NUMBER   the pull request whose diff to collect; may be empty
//
// Outputs:
//   /tmp/ci-logs/run.json       GitHub's record of the run
//   /tmp/ci-logs/jobs.json      the failed jobs, with their annotations and log paths
//   /tmp/ci-logs/job-<ID>.log   each failed job's log
//   /tmp/ci-logs/pr.diff        the pull request's diff
//   has_evidence                step output: whether any log or annotation was collected
//   unlisted_failed_job_count   step output: failed jobs that jobs.json does not list

'use strict';

const fs = require('node:fs');

// This query has GitHub list only the jobs with the given conclusions in the run's latest
// attempt, each with its annotations. A check run's databaseId is the job's ID.
const JOBS = `
  query($run: URI!, $conclusions: [CheckConclusionState!], $cursor: String) {
    resource(url: $run) { ... on WorkflowRun { checkSuite {
      checkRuns(
        first: 100
        after: $cursor
        filterBy: { checkType: LATEST, conclusions: $conclusions }
      ) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          databaseId
          name
          conclusion
          annotations(first: 100) {
            nodes { annotationLevel message title path location { start { line } end { line } } }
          }
        }
      }
    } } }
  }`;

module.exports = async ({ github, context, core }, dir = '/tmp/ci-logs') => {
  const { RUN_ID: runId, PR_NUMBER: pullNumber } = process.env;
  if (!/^[1-9][0-9]*$/.test(runId)) throw new Error('RUN_ID must be a positive integer');
  const { owner, repo } = context.repo;
  const save = (name, text) => fs.writeFileSync(`${dir}/${name}`, text);
  // Empty the directory first, so that it holds only this run's evidence.
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });

  // The triage runs next in the same job, under the same time limit, so stop sending
  // requests after ten minutes.
  const deadline = Date.now() + 600_000;
  // Send a request to GitHub, allowing it two minutes. When a reply breaks off after its
  // headers, Octokit returns an empty body instead of failing, so github-script does not
  // retry it. Make up to three attempts at such a request, unless its two minutes ran out.
  const ask = async request => {
    for (let attempt = 1; attempt <= 3; attempt++) {
      if (Date.now() >= deadline) throw new Error('the collector has used its ten minutes');
      const signal = AbortSignal.timeout(120_000);
      const data = await request({ request: { signal } });
      if (data) return data;
      if (signal.aborted) throw new Error('GitHub took longer than two minutes to reply');
    }
    throw new Error("GitHub's reply was empty or incomplete");
  };
  const rest = (method, parameters) =>
    ask(async options => (await method({ owner, repo, ...parameters, ...options })).data);

  // run.json is GitHub's record of the run, which the triage reads for the commit, the
  // branch and the event. Without it, the job still knows the ID and number of its own
  // run, which is all the report needs. It knows nothing about another run.
  let run = { id: Number(runId), run_number: context.runNumber };
  try {
    const record = await rest(github.rest.actions.getWorkflowRun, { run_id: runId });
    if (!Number.isInteger(record.run_number)) throw new Error("GitHub did not report the run's number");
    run = record;
  } catch (error) {
    if (context.runId !== Number(runId)) throw error;
    core.warning(`Unable to collect the run's details; continuing with its ID and number. (${error.message})`);
  }
  save('run.json', JSON.stringify(run, null, 2));

  if (pullNumber) {
    try {
      save('pr.diff', await rest(github.rest.pulls.get, {
        pull_number: pullNumber, mediaType: { format: 'diff' },
      }));
    } catch (error) {
      core.warning(`Unable to collect the pull request diff; continuing without it. (${error.message})`);
    }
  }

  const jobs = new Map();
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${runId}`;
  // Add to jobs the run's jobs that have one of these conclusions and that `wanted`
  // accepts. If a page cannot be read, warn and keep the jobs from the pages before it.
  // Return GitHub's count of jobs with these conclusions, or null if no page was read.
  const list = async (conclusions, description, wanted = () => true) => {
    let total = null;
    try {
      // Read at most twenty pages, which hold 2,000 jobs. The limit also stops a listing
      // whose cursor never advances.
      for (let page = 1, cursor = null; page <= 20; page++) {
        const { resource } = await ask(options => github.graphql(JOBS, { run: runUrl, conclusions, cursor, ...options }));
        if (!resource?.checkSuite) throw new Error('GitHub returned no workflow run that this token may see');
        const { totalCount, pageInfo, nodes } = resource.checkSuite.checkRuns;
        total = totalCount;
        for (const node of nodes ?? []) {
          // GitHub's schema allows a null entry and a check run without a job ID. Skip both.
          if (!node?.databaseId) continue;
          const job = {
            id: node.databaseId,
            name: node.name,
            conclusion: (node.conclusion ?? '').toLowerCase(),
            annotations: (node.annotations?.nodes ?? [])
              .filter(annotation => annotation?.annotationLevel === 'FAILURE')
              .map(({ message, title, path, location }) => ({
                annotation_level: 'failure', message, title, path,
                start_line: location.start.line, end_line: location.end.line,
              })),
            log_path: '',
          };
          if (wanted(job)) jobs.set(job.id, job);
        }
        cursor = pageInfo.hasNextPage ? pageInfo.endCursor : null;
        if (!cursor) break;
      }
    } catch (error) {
      core.warning(`Unable to list every ${description}; continuing with the jobs collected. (${error.message})`);
    }
    return total;
  };
  const total = await list(['FAILURE', 'TIMED_OUT', 'STARTUP_FAILURE', 'ACTION_REQUIRED'], 'failed job');
  const listed = jobs.size;
  // GitHub reports a job that it stopped at a time limit as cancelled, like a job that a
  // person cancelled. Only the job that was stopped has this annotation.
  await list(['CANCELLED'], 'job stopped at a time limit', job =>
    job.annotations.some(({ message }) => /exceeded the maximum execution time/.test(message)));

  const failed = [...jobs.values()].sort((a, b) => a.id - b.id);
  for (const job of failed) {
    try {
      const log = await rest(github.rest.actions.downloadJobLogsForWorkflowRun, { job_id: job.id });
      save(`job-${job.id}.log`, log);
      job.log_path = `${dir}/job-${job.id}.log`;
    } catch (error) {
      core.warning(`Unable to collect logs for job ${job.id}; continuing with job metadata. (${error.message})`);
    }
  }
  save('jobs.json', JSON.stringify(failed, null, 2));

  // Tell the next steps whether there is anything to triage, and how many failed jobs
  // GitHub counted that jobs.json does not list ("null" if no page of the list was read).
  core.setOutput('has_evidence', failed.some(job => job.log_path || job.annotations.length > 0));
  core.setOutput('unlisted_failed_job_count', total === null ? 'null' : Math.max(total - listed, 0));
};
