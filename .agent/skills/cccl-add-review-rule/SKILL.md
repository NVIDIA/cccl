---
name: cccl-add-review-rule
description: Use when adding or extending a rule in docs/cccl/development/review_guidelines.md; describes how to research, word, place, and validate a new review guideline.
---

# Adding a CCCL Review Guideline

## Workflow

1. Read `docs/cccl/development/review_guidelines.md` and follow its `Format` section exactly.
   List the historical regressions in the provenance comment (one `#intro→#fix note` per line),
   or `<!-- provenance: manually added -->` for a convention rule.
2. Research first: read the relevant PRs and/or a few representative code sites. Verify every
   symbol, macro, and path the rule names exists in the repo (grep) — never name an API from memory.
3. Word the rule diff-triggered ("when a diff does X, flag it unless Y"), state why the defect
   compiles/passes CI yet still breaks, keep one concrete example with real names, and name the
   acceptable-case escape hatch. Be concise: keep the body within the Format section's 3-8 lines,
   cutting examples before mechanism. If a grep could find the pattern, end with
   "Candidate for a pre-commit grep."
4. Insert the rule in area order (see the Format section).
5. Blind-test (required for critical/important rules): take a diff exhibiting the defect (the
   historical introducing PR, or revert the convention at one real code site), copy it to a neutral
   filename, and have a fresh reviewer agent with no other context review it per
   `.agent/skills/cccl-review/SKILL.md` against a provenance-stripped guidelines copy containing the
   candidate rule — no hints. PASS iff it flags the defect at important/critical with the right file
   and mechanism. Run once without the rule: if that baseline already flags it, the rule is not
   load-bearing. On failure, sharpen the trigger and retest (up to 3 attempts).
6. Report the final rule text and the validation outcome.
