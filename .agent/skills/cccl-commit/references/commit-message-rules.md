# Commit message rules

Used by `cccl-commit` Step 5.2.

## Subject line

- 72 characters maximum.
- Imperative mood: "Add X", "Fix Y", not "Added X" or "Fixes Y".
- No trailing period.
- Match CCCL's prefix convention — inspect `git log --oneline -20` before drafting.

Common prefix patterns (verify against log; do not invent):

```
[libcudacxx] ...
[cub] ...
[thrust] ...
[cudax] ...
[ci] ...
[docs] ...
```

## Body

- Wrap lines at ~72 characters.
- Separate from subject with one blank line.
- Explain what changed and why; omit what is obvious from the diff.
- No story paragraphs ("Surfaced while debugging …", "Found during …").

## Detail tiers

| Tier     | When                              | Content                               |
|----------|-----------------------------------|---------------------------------------|
| Trivial  | Mechanical change, obvious from diff | Subject only                        |
| Standard | Most commits                      | Subject + 1–6 body lines              |
| Detailed | Complex change, non-obvious rationale | Subject + multi-paragraph body      |

## Skip tags

`[skip-*]` tags scope a single CI push and belong only on the **last commit's last line**.
They block merge if left in place — remind the user to remove them before final merge.

## Prohibited content

- No co-author lines (`Co-authored-by:`, `Co-Authored-By:`).
- No tool-attribution footers ("Generated with …", "AI-assisted").
- No marketing adjectives ("powerful", "robust", "comprehensive").
