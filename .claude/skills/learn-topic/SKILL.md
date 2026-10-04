---
name: learn-topic
description: Teach one interview-prep topic or subtopic in depth from the knowledge base, tied to Vishal's resume, with diagrams, illustrations and animations, code, real-world examples, frequently asked questions and an optional mock interview. Use when asked to learn, study, revise, explain or quiz a topic (e.g. "learn-topic kafka", "learn-topic GraphQL N+1", "mock interview on Spring Security").
---

# learn-topic

Interactive study session over this repo's knowledge base (Zensical site under `docs/`).

## Inputs
- `learn-topic <topic or subtopic>`, e.g. `learn-topic kafka`, `learn-topic "DataLoader"`.
- Optional flags: `--mock` (mock interview only), `--quick` (cheat-sheet revision only), `--level senior|lead`.

## Sources of truth
- `planning/topics.json`: topic slugs, priorities, subtopics, ★ resume flags, status.
- `docs/<topic-slug>/*.md`: the researched pages.
- `Vishal_Hulawale_Resume_10012026.pdf`: read with `pdftotext -layout`.
- `docs/progress/tracker.md`: mock-interview history and gaps (create it if missing).

## Procedure
1. **Locate** the topic or subtopic (fuzzy match on titles in `topics.json`). If the page is missing or `todo`, run the `research-subtopic` skill first, then continue.
2. **Read** the relevant page(s) plus the resume.
3. **Teach in depth**, building from first principles to senior/architecture level:
   - Start with the *why*: the problem the concept solves and what came before.
   - Explain internals and how it works under the hood, not just the API.
   - Use **Mermaid diagrams** (architecture, sequence, state, flow) with one-line captions on what to notice.
   - Use analogies, worked examples and comparison tables for alternatives.
   - Add **media** (illustrations and animations) where a picture explains better than a Mermaid box diagram; see *Media* below.
4. **Code and real-life examples**, where applicable:
   - Production-style Java 21+/Spring Boot 3.x or TypeScript/React 19 code, commented on the key lines.
   - Wrong vs right side by side.
   - Config snippets (application.yml, Kafka/Redis/K8s/Terraform) when the topic is config-heavy.
   - Real-world usage: how large companies use it, known failure modes, relevance to healthcare/banking/cloud security.
5. **Connect to the resume:** talking points, one STAR story (Situation, Task, Action, Result), the claims to defend, and the follow-up chains interviewers use. Never invent experience: mark gaps as *[confirm]* and ask Vishal to fill them in.
6. **Frequently asked interview questions**, grouped Fundamentals → Intermediate → Senior → Scenario:
   - 15–25 per main topic, 8–12 per subtopic, focused on what is actually asked in Senior/Lead interviews.
   - Each with a crisp model answer, the key points interviewers listen for, and the common wrong answer.
   - Include gotcha and output-prediction questions where relevant (Java concurrency, JS event loop).
7. **Cheat sheet:** a one-page summary of facts, defaults, numbers and trade-offs for last-minute revision.
8. **Mock interview** (offer it, or run it with `--mock`): ask one question at a time, wait for the answer, score it 1–5, show what a strong answer adds, ask the realistic follow-up, and repeat 5–10 times. Then summarise strengths and gaps.
9. **Save:**
   - Append a dated entry to `docs/progress/tracker.md` (topic, score, gaps, next review date using spaced repetition: +1d, +3d, +7d, +21d).
   - If the session produced material missing from the page (better example, missing question, *[confirm]* answered by Vishal), update the page and run `python3 .claude/skills/research-subtopic/mark_done.py <topic-slug>`.
   - If the page has fewer than two illustrations, add them (see *Media*) and save them with the page.
   - Make sure the site still builds (`zensical build`, or a scratch copy if `site/` can't be cleaned).

## Media
Every subtopic page should carry **2–3 illustrations**, at least one animated when the concept is a process over time (a race, a retry, a rebalance, a rollout). They complement the Mermaid diagrams; they don't replace them.

- **What earns a picture:** state changing over time (replication lag, a circuit breaker tripping, a rolling update), a race or failure the reader must *see* happen (lost update, dual write, stale lock), a data structure's shape (hash ring, B-tree, log segments), or a wrong-vs-right comparison. Skip pictures that just restate a table or a list.
- **Format:** hand-written SVG, static or animated with CSS keyframes (the site's GIF equivalent: crisp, small, theme-aware). Use GIF/PNG/WebP/MP4 only for real screenshots or recordings (e.g. a Grafana panel, browser DevTools) that you made or that are licensed for reuse, with the source credited in the caption. Never hotlink remote media.
- **File:** `docs/<topic>/images/<nn>-<short-name>.svg`, where `<nn>` is the subtopic number. Keep each file under ~30 KB (most are 5–10 KB).
- **SVG conventions** (copy an existing file such as `docs/microservices/images/06-circuit-breaker-window.svg` as a starting point):
  - `viewBox="0 0 800 …"`, `role="img"`, a `<title>` and a `<desc>` that states what the picture shows in full sentences.
  - Figtree font stack, the site palette (indigo `#4f46e5`, teal `#0ea5a4`, green / red / amber for ok / bad / warning), and a `@media (prefers-color-scheme: dark)` block redefining every colour.
  - Animations loop (about 8–16 s), and a `@media (prefers-reduced-motion: reduce)` block stops them on the most informative frame.
  - Label the numbers you show; say in the `desc` when they're simplified compared with the page.
- **Embedding:** right after the paragraph it illustrates:
  ```markdown
  ![Animation: <what happens, step by step, as alt text>](images/<nn>-<short-name>.svg){ loading=lazy }
  *<One line on what to watch for.>*
  ```
  Start alt text with `Animation:` for animated files.
- **Check:** open the SVG in a browser in light and dark mode (Playwright + Chromium is available for screenshots), confirm nothing overlaps or clips, and run `zensical build`.

## Style
- Teach, don't dump: short sections, check understanding with a quick question after each major concept in interactive mode.
- Be honest about uncertainty and version differences. Cite official docs for non-obvious claims.
