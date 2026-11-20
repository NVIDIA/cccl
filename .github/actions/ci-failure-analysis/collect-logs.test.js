// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Run offline: node --test .github/actions/ci-failure-analysis/collect-logs.test.js
//
// actions/github-script gives the collector its GitHub client and retries the requests
// that fail. These tests give it a fake client instead, which answers from a scenario
// and records each request. A request to the fake can fail, succeed, or come back with
// an empty body, which is what the real client returns when a reply breaks off after its
// headers.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const collect = require('./collect-logs.js');

const read = name => fs.readFileSync(path.join(__dirname, name), 'utf8');
const MESSAGE = '\u001b[31m\u{1F6E0} build failed\u001b[0m'; // with color codes, as real annotations have
const LOG = '\u{1F6E0} build failed\n';
// The fake lists a warning for every job and, for an annotated job, two failures as well.
const listedAnnotation = ([annotationLevel, message, title, start, end]) => ({
  annotationLevel, message, title, path: '.github', location: { start: { line: start }, end: { line: end } },
});
const LISTED_ANNOTATIONS = [
  ['WARNING', 'deprecated', null, 1, 1],
  ['FAILURE', 'Process completed with exit code 1.', '', 12, 12],
  ['FAILURE', MESSAGE, 'Build', 7, 9],
].map(listedAnnotation);
// GitHub reports job 5, which it stopped at a time limit, and job 7, which a person
// cancelled, both as cancelled. Only job 5 has the annotation about the time limit.
const STOPPED = 'The job has exceeded the maximum execution time of 6h0m0s';
const CANCELLED_ANNOTATIONS = {
  5: [['FAILURE', STOPPED, null, 1, 1], ['FAILURE', 'The operation was canceled.', null, 1, 1]],
  7: [['FAILURE', 'The operation was canceled.', null, 1, 1]],
};
const STOPPED_ANNOTATIONS = [STOPPED, 'The operation was canceled.'].map(message => ({
  annotation_level: 'failure', message, title: null, path: '.github', start_line: 1, end_line: 1,
}));
const ANNOTATIONS = [
  { annotation_level: 'failure', message: 'Process completed with exit code 1.', title: '', path: '.github', start_line: 12, end_line: 12 },
  { annotation_level: 'failure', message: MESSAGE, title: 'Build', path: '.github', start_line: 7, end_line: 9 },
];
// The fake cannot run GraphQL, so it requires the exact query text, ignoring whitespace.
const QUERY = 'query($run: URI!, $conclusions: [CheckConclusionState!], $cursor: String) { '
  + 'resource(url: $run) { ... on WorkflowRun { '
  + 'checkSuite { checkRuns( first: 100 after: $cursor '
  + 'filterBy: { checkType: LATEST, conclusions: $conclusions } ) { totalCount '
  + 'pageInfo { hasNextPage endCursor } nodes { databaseId name conclusion '
  + 'annotations(first: 100) { nodes { annotationLevel message title path '
  + 'location { start { line } end { line } } } } } } } } } }';
const LISTING = 'Unable to list every failed job; continuing with the jobs collected. (';
const LOG1 = 'Unable to collect logs for job 1; continuing with job metadata. (';
const LOG3 = 'Unable to collect logs for job 3; continuing with job metadata. (';
const STOPPED_LISTING = 'Unable to list every job stopped at a time limit; continuing with the jobs collected. (';
const DIFF = 'Unable to collect the pull request diff; continuing without it. (';
const RUN = "Unable to collect the run's details; continuing with its ID and number. (";
const OUT_OF_TIME = 'the collector has used its ten minutes)';
const INCOMPLETE = "GitHub's reply was empty or incomplete)";
const TIMED_OUT = 'GitHub took longer than two minutes to reply)';
const EARLIER = ['run.json', 'jobs.json', 'pr.diff', 'job-1.log', 'job-99.log', 'notes.txt'];
const HEALTHY = {
  pages: [[1, 3]], // failed job IDs on each page of the listing
  total: undefined, // GitHub's count of failed jobs, or one count per page; defaults to the jobs listed
  annotated: [1, 3], // jobs with failure annotations
  cancelled: [[]], // cancelled job IDs on each page of that listing
  names: { 1: 'job 1 \u{1F6E0}', 3: 'job 3', 5: 'job 5', 7: 'job 7' },
  failing: [], // requests that fail
  empty: {}, // request -> how many of its first replies come back with an empty body
  timedOut: [], // requests whose two minutes run out while the reply arrives
  slow: {}, // request -> seconds it takes
  listedAs: {}, // job ID -> fields that the listing reports differently
  nulls: false, // the listing has a null job and null annotations, where the schema allows
  noNodes: false, // the listing has null in place of its jobs
  noCursor: false, // the listing promises more pages but gives no cursor
  invisible: false, // GitHub returns no run, as it does when the token may not see it
  run: { id: 123, run_number: 42, head_sha: 'c0ffee', event: 'schedule' }, // GitHub's record of the run
  server: 'https://github.com',
};

function listingPage(scenario, pages, number) {
  if (scenario.invisible) return { resource: null };
  const nodes = pages[number - 1].map(id => ({
    databaseId: id,
    name: scenario.names[id],
    conclusion: { 1: 'FAILURE', 3: 'TIMED_OUT', 5: 'CANCELLED', 7: 'CANCELLED' }[id],
    annotations: {
      nodes: CANCELLED_ANNOTATIONS[id]?.map(listedAnnotation)
        ?? LISTED_ANNOTATIONS.slice(0, scenario.annotated.includes(id) ? 3 : 1),
    },
    ...scenario.listedAs[id],
  }));
  if (scenario.nulls) {
    // GitHub's schema allows these entries to be null.
    for (const node of nodes) {
      if (scenario.annotated.includes(node.databaseId)) node.annotations.nodes.push(null);
      else node.annotations = null;
    }
    nodes.push(null);
  }
  const total = (pages === scenario.pages && scenario.total) || new Set(pages.flat()).size;
  return {
    resource: {
      checkSuite: {
        checkRuns: {
          totalCount: Array.isArray(total) ? total[number - 1] : total,
          pageInfo: {
            hasNextPage: number < pages.length,
            endCursor: scenario.noCursor ? null : String(number + 1),
          },
          nodes: scenario.noNodes ? null : nodes,
        },
      },
    },
  };
}

// Set (or, for null, unset) environment variables for the duration of a test.
function setEnvironment(t, variables) {
  for (const [name, value] of Object.entries(variables)) {
    const before = process.env[name];
    t.after(() => {
      if (before === undefined) delete process.env[name];
      else process.env[name] = before;
    });
    if (value === null) delete process.env[name];
    else process.env[name] = value;
  }
}

// Stands in for the client that github-script provides.
function fakeGitHub(t, scenario) {
  const calls = [];
  // Record each time limit the collector creates, so that request() can check that every
  // request carries its own limit and no other option.
  const limits = [];
  const timeLimit = t.mock.method(AbortSignal, 'timeout', () => {
    limits.push(new AbortController());
    return limits.at(-1).signal;
  });
  // noBody is what the real client returns when a reply breaks off after its headers.
  const request = (name, { request, ...parameters }, noBody, answer) => {
    calls.push(name);
    assert.deepEqual(Object.keys(request), ['signal'], `${name}: unexpected request options`);
    assert.equal(request.signal, limits.at(-1).signal, `${name} does not carry its own time limit`);
    assert.equal(limits.length, calls.length);
    assert.ok(!request.signal.aborted);
    t.mock.timers.tick((scenario.slow[name] ?? 0) * 1000);
    if (scenario.failing.includes(name)) throw new Error(`${name} failed`);
    if (scenario.timedOut.includes(name)) {
      limits.at(-1).abort();
      return noBody;
    }
    if (calls.filter(call => call === name).length <= (scenario.empty[name] ?? 0)) return noBody;
    return answer(parameters);
  };
  const github = {
    graphql: async (query, options) => {
      const cancelled = options.conclusions?.[0] === 'CANCELLED';
      const name = `${cancelled ? 'cancelled' : 'page'} ${options.cursor ?? 1}`;
      return request(name, options, undefined, ({ run, conclusions, cursor }) => {
        assert.equal(query.trim().split(/\s+/).join(' '), QUERY);
        assert.equal(run, `${scenario.server}/NVIDIA/cccl/actions/runs/123`);
        assert.deepEqual(
          conclusions,
          cancelled ? ['CANCELLED'] : ['FAILURE', 'TIMED_OUT', 'STARTUP_FAILURE', 'ACTION_REQUIRED'],
        );
        return listingPage(scenario, cancelled ? scenario.cancelled : scenario.pages, Number(cursor ?? 1));
      });
    },
    rest: {
      actions: {
        getWorkflowRun: async options => request('run', options, { data: '' }, parameters => {
          assert.deepEqual(parameters, { owner: 'NVIDIA', repo: 'cccl', run_id: '123' });
          return { data: scenario.run };
        }),
        downloadJobLogsForWorkflowRun: async options => request(`log ${options.job_id}`, options, { data: '' }, parameters => {
          assert.deepEqual(parameters, { owner: 'NVIDIA', repo: 'cccl', job_id: options.job_id });
          return { data: LOG };
        }),
      },
      pulls: {
        get: async options => request('diff', options, { data: '' }, ({ pull_number: number, ...parameters }) => {
          assert.deepEqual(parameters, { owner: 'NVIDIA', repo: 'cccl', mediaType: { format: 'diff' } });
          if (number !== '456') throw new Error('Not Found');
          return { data: 'pull request diff\n' };
        }),
      },
    },
  };
  return { github, calls, timeLimit };
}

async function check(t, {
  ownRun = true, // examining the run the collector is part of
  pullNumber = null, // PR_NUMBER; null leaves it unset
  runId = '123', // RUN_ID; null leaves it unset
  earlier = false, // the directory holds files of an earlier collection
  // Expected outcome.
  rejects, // the error, if the collection must fail
  left = [], // the files that must remain in the directory after that failure
  runRequests = 1, // requests for the run's record, which come before all others
  runRecord = true, // run.json holds GitHub's record, not just the job's own ID and number
  ids = [1, 3], // jobs in jobs.json
  conclusions = { 1: 'failure', 3: 'timed_out', 5: 'cancelled' },
  logged = [1, 3], // jobs whose log was saved
  hasEvidence = true,
  unlisted = 0, // unlisted_failed_job_count
  diff = false, // pr.diff was saved
  calls = ['page 1', 'log 1', 'log 3'], // the other requests, in order
  cancelledRequests = ['cancelled 1'], // requests for cancelled jobs, which come before the logs
  warnings = [], // how each warning starts, in order
  ...overrides // what GitHub does; see HEALTHY
}) {
  const scenario = { ...HEALTHY, ...overrides };
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'collect-logs-'));
  const dir = path.join(root, 'new directory', 'ci logs');
  setEnvironment(t, { RUN_ID: runId, PR_NUMBER: pullNumber });
  t.after(() => fs.rmSync(root, { recursive: true }));
  if (earlier) {
    fs.mkdirSync(dir, { recursive: true });
    for (const name of EARLIER) fs.writeFileSync(path.join(dir, name), 'from an earlier collection');
  }
  t.mock.timers.enable({ apis: ['Date'] });
  const { github, calls: made, timeLimit } = fakeGitHub(t, scenario);
  const context = {
    repo: { owner: 'NVIDIA', repo: 'cccl' },
    runId: ownRun ? 123 : 999,
    runNumber: 44,
    serverUrl: scenario.server,
  };
  const core = {
    outputs: {},
    warnings: [],
    setOutput(name, value) { this.outputs[name] = value; },
    warning(message) { this.warnings.push(message); },
  };
  const files = () => (fs.existsSync(dir) ? fs.readdirSync(dir).sort() : []);

  const collection = collect({ github, context, core }, dir);
  if (rejects) {
    await assert.rejects(collection, rejects);
    assert.deepEqual(made, [...Array(runRequests).fill('run'), ...calls]);
    assert.deepEqual(core.outputs, {});
    assert.deepEqual(files(), [...left].sort());
    return;
  }
  await collection;

  const logs = calls.findIndex(call => call.startsWith('log '));
  const beforeLogs = logs < 0 ? calls.length : logs;
  assert.deepEqual(made, [
    ...Array(runRequests).fill('run'), ...calls.slice(0, beforeLogs), ...cancelledRequests, ...calls.slice(beforeLogs),
  ]);
  // The collector gives every request two minutes.
  assert.deepEqual(timeLimit.mock.calls.map(call => call.arguments), made.map(() => [120_000]));
  assert.equal(core.warnings.length, warnings.length, core.warnings.join('\n'));
  warnings.forEach((start, index) => assert.ok(core.warnings[index].startsWith(start), core.warnings[index]));
  assert.deepEqual(core.outputs, {
    has_evidence: hasEvidence,
    unlisted_failed_job_count: unlisted === null ? 'null' : unlisted,
  });
  // Nothing else is in the directory: no lost diff, no empty log, nothing from before.
  assert.deepEqual(
    files(),
    ['run.json', 'jobs.json', ...(diff ? ['pr.diff'] : []), ...logged.map(id => `job-${id}.log`)].sort(),
  );
  const saved = name => fs.readFileSync(path.join(dir, name), 'utf8');
  assert.deepEqual(JSON.parse(saved('run.json')), runRecord ? scenario.run : { id: 123, run_number: 44 });
  assert.deepEqual(JSON.parse(saved('jobs.json')), ids.map(id => ({
    id,
    name: scenario.names[id],
    conclusion: conclusions[id],
    annotations: id === 5 ? STOPPED_ANNOTATIONS : scenario.annotated.includes(id) ? ANNOTATIONS : [],
    log_path: logged.includes(id) ? path.join(dir, `job-${id}.log`) : '',
  })));
  for (const id of logged) assert.equal(saved(`job-${id}.log`), LOG);
  if (diff) assert.equal(saved('pr.diff'), 'pull request diff\n');
}

// Each case overrides the healthy scenario and its expected outcome; see check().
const twoPages = ['page 1', 'page 2', 'log 1', 'log 3'];
const withDiff = ['diff', 'page 1', 'log 1', 'log 3'];
const nothingListed = { ids: [], logged: [], hasEvidence: false, unlisted: null, calls: ['page 1'] };
const badRunId = {
  earlier: true, rejects: /RUN_ID must be a positive integer/, left: EARLIER, runRequests: 0, calls: [],
};
const cases = {
  'one page': {},
  'two pages': { pages: [[1], [3]], calls: twoPages },
  'later page lost': {
    pages: [[1], [3]], failing: ['page 2'], ids: [1], logged: [1], unlisted: 1,
    calls: ['page 1', 'page 2', 'log 1'], warnings: [`${LISTING}page 2 failed)`],
  },
  'first page lost': { failing: ['page 1'], ...nothingListed, warnings: [LISTING] },
  'run not visible to the token': {
    invisible: true, ...nothingListed,
    warnings: [LISTING, STOPPED_LISTING].map(start => `${start}GitHub returned no workflow run that this token may see)`),
  },
  'null entries in the listing': { nulls: true, annotated: [1] },
  'listing without its jobs': {
    noNodes: true, ids: [], logged: [], hasEvidence: false, unlisted: 2, calls: ['page 1'],
  },
  'job listed without an ID': {
    listedAs: { 1: { databaseId: null } }, ids: [3], logged: [3], unlisted: 1, calls: ['page 1', 'log 3'],
  },
  'job listed with ID 0': {
    listedAs: { 1: { databaseId: 0 } }, ids: [3], logged: [3], unlisted: 1, calls: ['page 1', 'log 3'],
  },
  'job listed without a conclusion': {
    listedAs: { 3: { conclusion: null } }, conclusions: { 1: 'failure', 3: '' },
  },
  'more pages promised without a cursor': {
    pages: [[1], [3]], noCursor: true, ids: [1], logged: [1], unlisted: 1, calls: ['page 1', 'log 1'],
  },
  'page limit': {
    pages: Array.from({ length: 22 }, (_, index) => [index % 2 ? 3 : 1]), total: 5, unlisted: 3,
    calls: [...Array.from({ length: 20 }, (_, index) => `page ${index + 1}`), 'log 1', 'log 3'],
  },
  'job repeated on a later page': { pages: [[1, 3], [3]], calls: twoPages },
  'jobs listed out of order': { pages: [[3, 1]] },
  'two jobs with the same name': { names: { 1: 'build', 3: 'build' } },
  'job name with half a surrogate pair': { names: { 1: 'job \ud83d', 3: 'job 3' } },
  'more jobs listed than counted': { total: 1 },
  'more jobs fail while listing': { pages: [[1], [3]], total: [2, 4], unlisted: 2, calls: twoPages },
  'no failed jobs': { pages: [[]], ids: [], logged: [], hasEvidence: false, calls: ['page 1'] },
  'log lost, annotations remain': { failing: ['log 1'], logged: [3], warnings: [`${LOG1}log 1 failed)`] },
  'only one job has annotations': { annotated: [3] },
  'logs only': { annotated: [] },
  'annotations only': { failing: ['log 1', 'log 3'], logged: [], warnings: [LOG1, LOG3] },
  'no evidence': {
    annotated: [], failing: ['log 1', 'log 3'], logged: [], hasEvidence: false, warnings: [LOG1, LOG3],
  },
  'diff collected': { pullNumber: '456', diff: true, calls: withDiff },
  'diff lost': { pullNumber: '456', failing: ['diff'], calls: withDiff, warnings: [`${DIFF}diff failed)`] },
  'empty pull request number': { pullNumber: '' },
  'pull request that GitHub does not know': {
    pullNumber: '4 5', calls: withDiff, warnings: [`${DIFF}Not Found)`],
  },
  // GitHub reports a job that it stopped at a time limit as cancelled. The collector keeps
  // that job and leaves out jobs that were cancelled for another reason.
  'job stopped at a time limit': {
    cancelled: [[5, 7]], ids: [1, 3, 5], logged: [1, 3, 5], calls: ['page 1', 'log 1', 'log 3', 'log 5'],
  },
  'only a job stopped at a time limit': {
    pages: [[]], cancelled: [[5]], ids: [5], logged: [5], calls: ['page 1', 'log 5'],
  },
  'cancelled jobs on two pages': {
    cancelled: [[7], [5]], cancelledRequests: ['cancelled 1', 'cancelled 2'], ids: [1, 3, 5],
    logged: [1, 3, 5], calls: ['page 1', 'log 1', 'log 3', 'log 5'],
  },
  'cancelled jobs cannot be listed': {
    cancelled: [[5]], failing: ['cancelled 1'], warnings: [`${STOPPED_LISTING}cancelled 1 failed)`],
  },
  'failed jobs cannot be listed, but a stopped job can': {
    failing: ['page 1'], cancelled: [[5]], ids: [5], logged: [5], unlisted: null,
    calls: ['page 1', 'log 5'], warnings: [LISTING],
  },
  'stopped jobs do not count as listed failed jobs': {
    total: 3, cancelled: [[5]], ids: [1, 3, 5], logged: [1, 3, 5], unlisted: 1,
    calls: ['page 1', 'log 1', 'log 3', 'log 5'],
  },
  // The run's record is extra context for this job's own run, and required for another.
  'run record lost': { failing: ['run'], runRecord: false, warnings: [`${RUN}run failed)`] },
  'run record breaks off, then arrives': { empty: { run: 1 }, runRequests: 2 },
  'run record keeps breaking off': {
    empty: { run: 3 }, runRequests: 3, runRecord: false, warnings: [RUN + INCOMPLETE],
  },
  'run record without a run number': {
    run: { id: 123 }, runRecord: false, warnings: [`${RUN}GitHub did not report the run's number)`],
  },
  'another run': { ownRun: false },
  'another run that GitHub does not know is fatal': {
    ownRun: false, failing: ['run'], rejects: /run failed/, calls: [],
  },
  'another run whose number GitHub does not report is fatal': {
    ownRun: false, run: { id: 123 }, rejects: /did not report the run's number/, calls: [],
  },
  'another run whose record keeps breaking off is fatal': {
    ownRun: false, empty: { run: 3 }, runRequests: 3, rejects: /empty or incomplete/, calls: [],
  },
  'another GitHub server': { server: 'https://github.example.com' },
  'files left by an earlier collection': {
    earlier: true, pullNumber: '456', failing: ['diff', 'log 1'], logged: [3], calls: withDiff,
    warnings: [DIFF, LOG1],
  },
  // When a reply breaks off after its headers, its body is empty. The collector makes up
  // to three attempts and never saves an empty body.
  'diff breaks off, then arrives': { pullNumber: '456', empty: { diff: 1 }, diff: true, calls: ['diff', ...withDiff] },
  'diff keeps breaking off': {
    pullNumber: '456', empty: { diff: 3 }, calls: ['diff', 'diff', ...withDiff], warnings: [DIFF + INCOMPLETE],
  },
  'listing breaks off twice, then arrives': {
    empty: { 'page 1': 2 }, calls: ['page 1', 'page 1', 'page 1', 'log 1', 'log 3'],
  },
  'listing keeps breaking off': {
    empty: { 'page 1': 3 }, ...nothingListed, calls: ['page 1', 'page 1', 'page 1'],
    warnings: [LISTING + INCOMPLETE],
  },
  'log breaks off, then arrives': { empty: { 'log 1': 1 }, calls: ['page 1', 'log 1', 'log 1', 'log 3'] },
  'empty log': {
    empty: { 'log 1': 3 }, logged: [3], calls: ['page 1', 'log 1', 'log 1', 'log 1', 'log 3'],
    warnings: [LOG1 + INCOMPLETE],
  },
  'reply cut off by its time limit is not asked for again': {
    timedOut: ['log 1'], logged: [3], warnings: [LOG1 + TIMED_OUT],
  },
  // The collector sends no request once ten minutes have passed.
  'request ends just inside the ten minutes': { slow: { 'log 1': 599 } },
  'request ends as the ten minutes run out': {
    slow: { 'log 1': 600 }, logged: [1], calls: ['page 1', 'log 1'],
    warnings: [LOG3 + OUT_OF_TIME],
  },
  'empty reply is not asked for again once the ten minutes are up': {
    slow: { 'log 1': 600 }, empty: { 'log 1': 1 }, logged: [], calls: ['page 1', 'log 1'],
    warnings: [LOG1 + OUT_OF_TIME, LOG3 + OUT_OF_TIME],
  },
  'listing uses up the ten minutes': {
    pages: [[1], [3]], slow: { 'page 1': 600 }, ids: [1], logged: [], unlisted: 1, calls: ['page 1'],
    cancelledRequests: [], warnings: [LISTING + OUT_OF_TIME, STOPPED_LISTING + OUT_OF_TIME, LOG1 + OUT_OF_TIME],
  },
  'diff uses up the ten minutes': {
    pullNumber: '456', slow: { diff: 600 }, diff: true, ...nothingListed, calls: ['diff'],
    cancelledRequests: [], warnings: [LISTING + OUT_OF_TIME, STOPPED_LISTING + OUT_OF_TIME],
  },
  // The collector checks the run ID before it requests, writes or removes anything.
  'run ID 0': { runId: '0', ...badRunId },
  'run ID with letters': { runId: '12a', ...badRunId },
  'run ID that is a path': { runId: '../123', ...badRunId },
  'no run ID': { runId: null, ...badRunId },
};
for (const [name, options] of Object.entries(cases)) {
  test(name, t => check(t, options));
}

// The whole step. The test compares all of it, so no line can move between its blocks
// unnoticed.
const STEP = `    - name: Collect workflow logs
      id: collect-logs
      uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9
      env:
        PR_NUMBER: \${{ inputs.pr_number }}
        RUN_ID: \${{ inputs.run_id }}
        COLLECTOR_PATH: \${{ github.action_path }}/collect-logs.js
      with:
        # Try each request up to three times: GitHub's API sometimes fails.
        retries: 2
        script: |
          const collect = require(process.env.COLLECTOR_PATH);
          await collect({ github, context, core });
`;

test('the action step runs the collector with retries', async t => {
  const action = read('action.yml');
  assert.ok(action.includes(`  steps:\n${STEP}\n    - name: Prepare analysis environment\n`));
  // Run the step's script as github-script does, with the collector replaced by a stub.
  const script = STEP.split('        script: |\n')[1];
  const stub = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'collect-logs-')), 'stub.js');
  fs.writeFileSync(stub, 'module.exports = async (...given) => { module.exports.given = given; };');
  setEnvironment(t, { COLLECTOR_PATH: stub });
  t.after(() => fs.rmSync(path.dirname(stub), { recursive: true }));
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  const [github, context, core] = [{}, {}, {}];
  await new AsyncFunction('github', 'context', 'core', 'require', script)(github, context, core, require);
  // One argument, so the collector writes to its default directory.
  const { given } = require(stub);
  assert.equal(given.length, 1);
  assert.deepEqual(Object.keys(given[0]), ['github', 'context', 'core']);
  assert.ok(given[0].github === github && given[0].context === context && given[0].core === core);
});

test('the later steps read the directory that the collector writes by default', () => {
  assert.ok(read('collect-logs.js').includes("module.exports = async ({ github, context, core }, dir = '/tmp/ci-logs') => {"));
  const action = read('action.yml');
  for (const use of [
    '        CI_LOG_DIR: /tmp/ci-logs\n',
    '          --slurpfile run /tmp/ci-logs/run.json \\\n',
    '          --slurpfile jobs /tmp/ci-logs/jobs.json \\\n',
  ]) {
    assert.ok(action.includes(use), use);
  }
  assert.equal(action.split('/tmp/ci-logs').length - 1, 3);
  assert.ok(read('config.toml').includes('"/tmp/ci-logs" = "read"\n'));
});
