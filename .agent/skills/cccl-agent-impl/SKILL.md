---
name: cccl-agent-impl
description: "How skills and agents work in the CCCL repository. Filesystem layout, invocation, frontmatter, allow-list semantics, intent-driven auto-discovery. Load this skill when you land in the CCCL repo cold and don't know what skills or agents are, when you see references to `.agent/skills` or `.agent/agents` and want to understand them, or when authoring a new CCCL skill or agent."
---

# cccl-agent-impl

## Filesystem

```
<repo>/.agent/
  skills/<name>/SKILL.md
  agents/<name>.md

<repo>/.claude/
  skills  -> ../.agent/skills    (directory symlink)
  agents  -> ../.agent/agents    (directory symlink)
  settings.json
```

Canonical files live under `.agent/`. Claude Code reads `.claude/skills/` and `.claude/agents/`; Codex reads
`.agent/`.

## Skills

`.agent/skills/<name>/SKILL.md`. Frontmatter:

```yaml
---
name: <kebab-case>
description: "<trigger surface — used for intent matching>"
---
```

Invoke via the **Skill tool** with `skill: <name>`. Not reentrant.

## Agents

`.agent/agents/<name>.md`. Frontmatter:

```yaml
---
name: <name>
description: "<what and when>"
model: haiku
tools: Read, Grep, Bash
---
```

CCCL agents are **non-interactive** — no `AskUserQuestion`. User dialogue belongs in the calling skill (often via
`cccl-clarify`). Pick `model:` per workload: `haiku` for mechanical tasks (log parsing, jq munging, SHA
verification); `sonnet` for multi-file reasoning or judgment (e.g. `cccl-ci-overrides`).

Dispatch via the **Agent tool** with `subagent_type: <name>`. The agent runs to completion and returns one message.
