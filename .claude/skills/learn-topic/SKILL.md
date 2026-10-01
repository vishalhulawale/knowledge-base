---
name: learn-topic
description: Teach one interview-prep topic or subtopic in depth from the knowledge base, tied to Vishal's resume, with diagrams, code, real-world examples, frequently asked questions and an optional mock interview. Use when asked to learn, study, revise, explain or quiz a topic (e.g. "learn-topic kafka", "learn-topic GraphQL N+1", "mock interview on Spring Security").
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
   - Make sure the site still builds (`zensical build`, or a scratch copy if `site/` can't be cleaned).

## Style
- Teach, don't dump: short sections, check understanding with a quick question after each major concept in interactive mode.
- Be honest about uncertainty and version differences. Cite official docs for non-obvious claims.
