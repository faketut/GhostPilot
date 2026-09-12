# Knowledge base (RAG)

Put your local materials here, then the app will load them on startup and use them as retrieval context.

Recommended files:

- `resume.md`: your resume in bullet points
- `cheatsheet.md`: quick interview cheatsheet
- `notes/`: any topic notes (`system-design.md`, `python.md`, ...)

Supported formats: `.md`, `.txt`

## `algorithm.md` is special

Algorithm turns don't retrieve — they get this one file injected **whole**
(`ALGORITHM_KNOWLEDGE_FILE`, default `algorithm.md`). The shipped file is a
starter: pattern triggers, loop shapes, traps.

Why whole instead of retrieved: a cheatsheet's value is its structure, and a
900-char chunk of it reads as a fragment. It is re-read on every algorithm
question, so edits apply immediately — no restart, no rebuild button.

It is reference material, not instructions. The output format (UMPIR + one
fenced code block, no comments in code) comes from `prompts/algorithm.md` and
overrides anything written here.

Keep an eye on size: it is sent on **every** algorithm turn. The overlay footer
shows `algo:Nc`; the 2.1 KB default costs ≈0.6 k input tokens per turn.
`ALGORITHM_KNOWLEDGE_MAX_CHARS` (default 8000) bounds it.


