"""Model Context Protocol server used to drive the CLI agent backends.

`claude -p` and `codex exec` have no forced-tool-choice flag and no way to
hand structured output back to a caller. This package closes both gaps: it
exposes the project's existing `record_*` schemas and research tools over MCP
stdio, so the CLI agent's own tool loop produces exactly the structured
records and URL provenance the agents already expect.
"""
