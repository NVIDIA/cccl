---
name: cccl-clarify
description: "Decision-point escalation. Use when you cannot resolve a question through default reasoning — tricky tradeoffs, scarce evidence, ambiguous user intent, or a fork in the road that needs human judgment. Triggered by phrases like \"I'm stuck\", \"not sure how to proceed\", \"should I X or Y\", \"help me decide\". Also invoked by other cccl-* skills when they need to surface a question to the user. Walks the three-step escalation (default reasoning → self-research → ask the user) and the \"how to ask well\" rules — print context in chat, AskUserQuestion with breakdown branch, point-by-point dialogue."
---

# cccl-clarify

## Escalation ladder

Stop at the first level that produces a confident answer.

1. **Default reasoning** — resolve from existing context: prompt, conversation, files read, `AGENTS.md`, `cccl`
   skill, memory. Escalate if the tradeoffs are balanced, evidence is thin, the decision is hard to reverse, or
   intent is genuinely ambiguous.
2. **Self-research** — cheapest source first: code, memory, in-repo docs (`AGENTS.md`, `CONTRIBUTING.md`,
   `ci-overview.md`), upstream library docs, web, Explore subagent. Time-box. Two or three rounds without
   confidence moving = escalate.
3. **Ask the user** — when research won't close the gap.

## How to ask well

1. **Print context in chat.** Tool output isn't visible to the user. Frame the decision, what was tried, the
   tradeoff axis — in your text, not just in the question prompt.
2. **`AskUserQuestion` correctly.** 2–4 mutually-exclusive options (or `multiSelect`). Lead with the recommendation
   and suffix `(Recommended)` when evidence favours it. Each option's `description` carries the substance. Don't
   add "Other" — UI handles it.
3. **Offer a breakdown branch** for non-trivial questions — a "walk me through it" option that lets the user defer
   the pick.
4. **Breakdown flow.** Offer further research (multi-select with "None — overview"). Then a 200–400 word overview:
   problem, ordered decision points, tradeoffs, what's already decided. Walk point-by-point — dependent questions
   sequential, not parallel. Confirm the chosen path end-to-end before acting.

## When NOT to invoke

- Single-line obvious fixes.
- Conversational questions — answer them.
- Decisions whose default is so obvious that asking is noise.
- Questions answered in `AGENTS.md`, the `cccl` skill, or memory.

## Hard prohibitions

- Never invoke recursively.
- Never use to defer a decision the user already made.
