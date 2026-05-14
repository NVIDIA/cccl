---
description: "Decision-point escalation when default reasoning cannot resolve a question — tricky tradeoffs, scarce evidence, ambiguous intent, or a hard-to-reverse fork. Other cccl-* skills route user-question moments here. Triggers: \"I'm stuck\", \"should I X or Y\", \"help me decide\", \"not sure how to proceed\"."
---

# cccl-clarify

Surfaces decisions that default reasoning cannot confidently close. Three-step ladder; stop at the first level that produces a confident answer.

## Step 1 — Default reasoning

Resolve from existing context: prompt, conversation, files read, `AGENTS.md`, `cccl` skill, memory. Escalate if tradeoffs are balanced, evidence is thin, the decision is hard to reverse, or intent is genuinely ambiguous.

## Step 2 — Self-research

Cheapest source first: code, memory, in-repo docs (`AGENTS.md`, `CONTRIBUTING.md`, `ci-overview.md`), upstream library docs, web, Explore subagent. Time-box. Two or three rounds without confidence moving = escalate.

## Step 3 — Ask the user

When research won't close the gap:

1. **Print context in chat.** Tool output isn't visible to the user. Frame the decision, what was tried, the tradeoff axis — in your text, not in the question prompt.
2. **`AskUserQuestion` correctly.** 2–4 mutually-exclusive options (or `multiSelect`). Lead with the recommendation and suffix `(Recommended)` when evidence favours it. Each option's `description` carries the substance. Don't add "Other" — UI handles it.
3. **Offer a breakdown branch** for non-trivial questions — a "walk me through it" option that lets the user defer the pick. See `references/breakdown-flow.md` for the full walkthrough protocol.

## When NOT to invoke

- Single-line obvious fixes.
- Conversational questions — answer them.
- Decisions whose default is obvious enough that asking is noise.
- Questions answered in `AGENTS.md`, the `cccl` skill, or memory.

## Hard prohibitions

- Never invoke recursively.
- Never use to defer a decision the user already made.

## Additional resources

- `references/breakdown-flow.md` — full breakdown branch walkthrough: research phase, overview format, point-by-point sequencing, confirmation step.
