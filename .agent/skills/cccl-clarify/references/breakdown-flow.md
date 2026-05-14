# Breakdown flow

Used when the user selects the "walk me through it" option on a non-trivial `AskUserQuestion`. Four phases, executed in order.

## Phase 1 — Further research

Offer a multi-select list of research directions relevant to the question. Include a "None — proceed to overview" option. Execute selected directions before continuing.

## Phase 2 — Overview

Write a 200–400 word summary covering:

- **Problem statement** — what needs to be decided and why it matters.
- **Ordered decision points** — the sequence of choices, not a flat list.
- **Tradeoffs** — what each option gains and costs; cite specific files or repo facts where available.
- **What's already settled** — constraints that are not up for debate.

Keep the overview factual. No recommendations yet — the goal is shared understanding before the user commits to anything.

## Phase 3 — Point-by-point walk

Work through each decision point from the overview in sequence.

- Present one question at a time via `AskUserQuestion`.
- Dependent questions wait for their prerequisite answer before being posed — never run them in parallel.
- After each answer, summarize the implication briefly in chat before moving to the next point.
- If an answer makes a later decision point moot, skip it and say so.

## Phase 4 — Confirm chosen path

After all decision points are resolved:

1. Print a concise summary of the full chosen path: each decision point and the selected answer, in order.
2. Ask the user to confirm before acting.
3. On confirmation, hand off to the calling skill or proceed with the action.

If the user changes an earlier answer during confirmation, re-walk only the affected downstream points.

## Constraints

- Never skip Phase 4 — acting without confirmation violates the breakdown contract.
- Keep each `AskUserQuestion` focused on one decision; don't bundle multiple questions into one prompt.
- The breakdown branch is for non-trivial forks only. Single-question decisions do not need a breakdown.
