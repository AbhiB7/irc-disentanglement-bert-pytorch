# Agent Instructions

> [!IMPORTANT]
> **<u>ANTI-DRIFT RULE</u>**: This file is for **STABLE BEHAVIORAL RULES** only.
> - **DO NOT** add project status, task progress, or technical research data here.
> - **DO NOT** add "Next Steps" or "Recent Work" here.
> - **ONLY** edit this if the fundamental way the agent should behave needs to change.

This file contains stable behavioral rules and constraints for the AI agent. It defines how the agent should interact with the codebase and the user.

## Response Style & Formatting
- **Language**: Always speak and think in English (en).
- **Markdown**: All file references or code constructs must be clickable links (e.g., [`filename.py`](path/to/filename.py) or [`class.method()`](path/to/file.py:line)).
- **Directness**: Be technical and direct. Avoid conversational filler like "Certainly" or "Okay."

## Tool Use Constraints
- **Atomic Edits**: Prefer `apply_diff` for surgical changes. Use `write_to_file` only for new files or total rewrites.
- **Verification**: Always wait for tool execution confirmation before proceeding to the next step.
- **Completeness**: When writing files, never use placeholders like `// rest of code unchanged`. Always provide the full content.

## Operational Rules
- **Workspace**: Operate strictly within the project root. Do not attempt to `cd` into outside directories.
- **Safety**: Before executing complex CLI commands, analyze the system environment (Windows/cmd.exe) to ensure compatibility.
- **Context Management**: Keep the three context files (`instructions.md`, `context.md`, `progress.md`) separated by their defined responsibilities.
