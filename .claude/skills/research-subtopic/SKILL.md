---
name: research-subtopic
description: Research one interview-prep subtopic from authoritative sources and write its knowledge-base page using the standard template. Use when asked to research, write, fill in or regenerate a subtopic page, or to batch-process a topic's subtopics.
---

# research-subtopic

Writes one page of the interview-prep knowledge base (Zensical/MkDocs site in this repo) for a single subtopic.

## Inputs
- A subtopic name, or a topic + subtopic number (e.g. `kafka 7` or `"Error handling: retry topics, DLQ"`).
- Optional: `--batch <topic-slug>` to process every `todo` subtopic of one topic, one page at a time.

## Files to use
- `planning/topics.json`: source of truth for topic slugs, priority, subtopic slugs, ★ resume flags and status.
- `Vishal_Hulawale_Resume_10012026.pdf`: read it (`pdftotext -layout`) for the "How this connects to my experience" section.
- `.claude/skills/research-subtopic/template.md`: the page template. Follow it exactly.
- Output: `docs/<topic-slug>/<subtopic-slug>.md`.

## Procedure
1. **Locate** the subtopic in `planning/topics.json`. Note its priority (P0/P1/P2) and ★ flag.
2. **Research** with web search/fetch. Prefer, in order:
   1. Official docs and specs (docs.oracle.com, JEPs, docs.spring.io, kafka.apache.org, react.dev, MDN, AWS docs, RFCs, OWASP).
   2. Recognised engineering sources (Confluent, Netflix/Uber/LinkedIn engineering blogs, Martin Fowler, Baeldung for Spring specifics).
   3. Well-regarded books (cite by title).
   Avoid low-quality listicles and unverified Q&A dumps. Cross-check any number, default or version claim against official docs.
3. **Target current versions**: Java 21/25 LTS, Spring Boot 3.x, Spring Security 6.x, Kafka 3.x/4.x (KRaft), React 19, TypeScript 5.x, Node LTS. Call out version differences interviewers ask about (e.g. ZooKeeper vs KRaft, Java 8 vs 21).
4. **Write the page** from the template:
   - Teach from first principles to senior level. Explain the *why* before the *how*.
   - Use **Mermaid** diagrams (sequence, flowchart, state, class) for any flow, architecture or lifecycle. Each diagram has a one-line caption saying what to notice.
   - Code: idiomatic, production-style Java 21+ / Spring Boot 3 / TypeScript, commented on the key lines. Show **wrong vs right** with content tabs where a common mistake exists.
   - Use comparison tables for alternatives.
   - Use admonitions: `!!! tip`, `!!! warning` (production gotchas), `!!! question` (interview angle).
   - **Interview questions**: P0 = 12–20, P1 = 8–12, P2 = 5–8. Group as Fundamentals / Intermediate / Senior / Scenario. Each answer goes in a collapsible `??? success "Answer"` block with: model answer, key points the interviewer listens for, common wrong answer.
   - If ★: make "How this connects to my experience" specific (project, bullet, a likely follow-up chain and how to answer it). If not ★: give a short, honest bridge to a project, or say "not used directly; position as transferable knowledge".
   - Never invent facts about my experience beyond the resume. Mark assumptions as *[confirm]* for me to fill in (e.g. exact traffic numbers).
5. **Link it up**:
   - Add the page under its topic in `mkdocs.yml` `nav` (after `Overview`, in subtopic order).
   - In `docs/<topic-slug>/index.md`, turn the subtopic name into a link and set status to `:material-check-circle: Done`.
   - Set the subtopic's `status` to `done` in `planning/topics.json`.
   - Shortcut for all three: `python3 .claude/skills/research-subtopic/mark_done.py <topic-slug>` (syncs nav, index table and status from pages on disk).
6. **Verify**: run `zensical build` (install with `pip install zensical` if missing) and fix warnings or broken links (if `site/` can't be cleaned, build a scratch copy: `cp -r docs mkdocs.yml /tmp/kb && cd /tmp/kb && zensical build`). Validate diagrams with `bash .claude/skills/research-subtopic/validate_mermaid.sh docs/<topic>/*.md` (needs mermaid-cli). Mermaid pitfalls: avoid participant IDs that are keywords (`In`, `Off`, `end`, `loop`); quote labels containing `()`, `:` or `/`; no `;` inside sequence-diagram messages (it ends the statement).
7. **Report** back in 3–5 lines: page path, key sources, anything marked *[confirm]*.

## Batch mode
Process subtopics in order. After each page: build, then continue. Commit per topic: `docs(<topic-slug>): add <n> subtopic pages`. Stop and report if a build fails twice.

## Quality bar
- Accurate over exhaustive; every non-obvious claim traceable to a source in the Sources section.
- Scannable: short paragraphs, headings every few screens, a "Key takeaways" box anyone could revise in one minute.
- Length guide: P0 ≈ 2,500–4,500 words, P1 ≈ 1,500–3,000, P2 ≈ 800–1,500.
