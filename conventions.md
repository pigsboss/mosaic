# Human‑AI Collaboration Conventions

This file is loaded at the start of each working session to establish
a common understanding between the human designer(s) and the AI assistant(s).

## Core Principle: Clean separation between document output and tool commands

- When writing or editing a Markdown file (or any document that mixes natural language
  with code blocks):
  1. The AI **must** announce: “Switching to document‑output mode.”
  2. During the document output, no other tool calls (file creation, execution, etc.)
     shall be made unless explicitly requested by the human.
  3. After the document content has been fully written, the AI **must** state:
     “Document output complete. Resuming normal interaction mode.”

- This convention avoids the “Inception” problem where code fragments inside
  a document are misinterpreted as executable commands, polluting the conversation
  context.

## Session startup

- At the beginning of every session, the AI shall confirm that it has read
  (or re‑read) this file.
- If the human asks for a task that involves a document, the AI should
  recall the document‑output mode rule.

## Additional preferences

- The human works in a Jupyter notebook environment with `.py` and `.md` files.
- All code modifications must be explicitly requested by the human.
- The AI should provide brief explanations before code changes.
