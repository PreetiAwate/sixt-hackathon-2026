# CodeRabbit for the hackathon: setup guide

Automatic AI code review for pull requests, tuned for hackathon teams. The baseline config is
[sample_project/.coderabbit.yaml](sample_project/.coderabbit.yaml). It covers Java, Python, Go and C, keeps reviews short, and never blocks a merge.
It is generic: it knows nothing about a specific challenge, so each team adds its own details.

## What the config does

| Setting | Effect |
| --- | --- |
| `profile: chill` and `tone_instructions` | Short reviews, only demo-breaking bugs, leaked keys, personal data and wrong results. No style or refactoring comments |
| Language `path_instructions` | Java, Python, Go and C checks: missing timeouts, unchecked errors, money in floating point, buffer overflows, leaks |
| `pre_merge_checks.custom_checks` | Two checks, both `warning` mode: **No Committed Secrets** and **Generative AI Safety**. Title, description, docstring and issue checks are off |
| `auto_review` | Reviews every target branch and drafts, skips bot PRs, pauses after 3 reviewed pushes (comment `@coderabbitai review` to resume) |
| `knowledge_base.code_guidelines` | Uses your challenge README as review context, so findings can point at what your project must do |
| `auto_approvals`, `request_changes_workflow` | Off. Nothing ever blocks a merge |
| `inheritance: true` | Your team's own rules add to this baseline instead of replacing it |

Every setting in the file has a comment explaining it.

## What teams fill in

Lines marked `# TEAM:` in the config are yours to complete:

1. **Challenge description**: point `knowledge_base.code_guidelines.filePatterns` at the file that describes your challenge. It defaults to `README.md`.
2. **API description** (optional): add an OpenAPI file or Postman collection to the same list.
3. **Path instructions**: add rules for your own folders or file types.
4. **Custom checks**: add checks for your own challenge (name, `mode: warning`, and when it applies, fails and passes). A commented example is in the file.

## Step by step: configure it for your team

1. **Install the CodeRabbit GitHub app** on your fork (fork owner only): https://github.com/apps/coderabbitai. Choose "Only select repositories" and pick your fork. Log in at https://app.coderabbit.ai/login.
2. **Copy the config to the repo root** under the name `.coderabbit.yaml`:
   ```bash
   cp CodeRabbit_Python/sample_project/.coderabbit.yaml .coderabbit.yaml
   ```
   Alternatively, an admin can paste the same content into Organization Settings in the CodeRabbit app, which applies it to every repository at once.
3. **Fill in the `# TEAM:` lines** (see above).
4. **Commit and push** the file to `main` of your fork.
5. **Open a pull request** inside your fork (for example from a team branch into your fork's `main`). CodeRabbit posts a summary and comments within a few minutes.
6. **Talk to it** in the PR, for example `@coderabbitai explain this change` or `@coderabbitai review` to re-run.

CodeRabbit only reads `.coderabbit.yaml` from the root of a repository.

## Review before you push: CLI and IDE

- **CLI**: `curl -fsSL https://cli.coderabbit.ai/install.sh | sh` (or `brew install coderabbit`), then `cr auth login` and `cr review` inside the repository. Docs: https://docs.coderabbit.ai/cli
  - Claude Code integration: https://docs.coderabbit.ai/cli/claude-code-integration
  - Cursor integration: https://docs.coderabbit.ai/cli/cursor-integration
- **IDE** (VS Code, Cursor, Windsurf): https://docs.coderabbit.ai/ide and https://docs.coderabbit.ai/ide/install

## Try it on the sample project

[sample_project/](sample_project/) is a tiny Python app with deliberate problems: a mutable default argument, a bare `except`, an unused import, a TODO, a wrong docstring, a hardcoded key and a weak test name. Copy it to a branch, open a PR and see what CodeRabbit flags.

Check locally what the Ruff part will find:
```bash
cd CodeRabbit_Python/sample_project
pip install ruff pytest
ruff check .
pytest
```

## Validate your config

The first line of the YAML points to the official schema, so editors like VS Code (YAML extension) flag typos. A mistyped key is silently ignored by CodeRabbit, so validate before pushing.

## Tips

- Keep your additions short. Add instructions only for things you care about.
- Never put secrets or real data in review instructions or code.
- Reviews apply only to repos where the app is installed. Nothing is reviewed until step 1 is done.

## Other files in this folder

- [coderabbit-python.yaml](coderabbit-python.yaml) is the earlier Python-only config (Google style, pytest, ruff). Kept for reference; the baseline above replaces it.
- [.claude/agents/coderabbit-config-tester.md](../.claude/agents/coderabbit-config-tester.md) is a Claude Code agent that checks a config against the sample project. [TEST_REPORT.md](TEST_REPORT.md) is its last run, made against the earlier Python-only config, so it does not cover the baseline.

## Support

Kevin Holler (holler@coderabbit.ai) and gal@coderabbit.ai are available for questions before and during the hackathon.
