# Interview Prep Knowledge Base — Master Prompt

## Context
I'm a Lead Software Engineer / Technical Lead with 9+ years of experience (Publicis Sapient, Deloitte, Coriolis, Johnson Controls), targeting **Senior / Lead / Staff Engineer** roles. My resume is at `Vishal_Hulawale_Resume_10012026.pdf` in this repo — treat it as the source of truth for my experience.

Core stack: Java, Spring Boot (Cloud, Security), Kafka, GraphQL, MongoDB, PostgreSQL, Redis, ReactJS (Redux, React Query, micro-frontends), TypeScript, AWS, Azure, Kubernetes, Docker, Terraform, OAuth2/JWT/PingFederate, AWS KMS/HSM.

Interviewers will probe **every claim on my resume**, so preparation must cover both the fundamentals of each technology and how I would defend the specific things I say I built (e.g. Kafka retry/DLQ workflows, Redis caching, the GraphQL integration layer over 5 upstream systems, micro-frontend architecture, key rotation with HSMs).

## Goal
Build a single, version-controlled interview-prep knowledge base that is my only source for preparation:
- Written in Markdown, built with **Zensical** (successor to Material for MkDocs; keep config in `mkdocs.yml` so it stays MkDocs-compatible).
- Source in my GitHub repo `vishalhulawale/knowledge-base`.
- Deployed to **Cloudflare Pages** via its GitHub integration (auto-deploy on push to `main`). Replace the existing GitHub Pages workflow.

We work **phase by phase**. Do not start a phase until I've approved the output of the previous one.

---

## Phase 1 — Main topic list
1. Read my resume fully.
2. Produce a list of main topics I need to prepare. Include technologies on the resume **and** the cross-cutting areas expected at my level even if not listed as a skill:
   - System Design (HLD) and Low-Level Design / design patterns
   - Distributed systems concepts (consistency, idempotency, resiliency, observability)
   - Data structures & algorithms (coding round)
   - Security (authN/authZ, OAuth2/OIDC, encryption, key management)
   - Leadership & behavioral (STAR stories from my projects)
3. For each topic return a table with: **Topic | Why it matters (resume evidence) | Priority (P0/P1/P2) | Target depth (Awareness / Working / Expert)**.
4. Flag anything on my resume that is a likely interview trap (claims that invite deep follow-ups).
5. Stop and wait for me to add, remove, or reprioritise.

## Phase 2 — Subtopics, research skill, and site scaffold
1. For each approved main topic, list subtopics ordered from fundamentals → advanced → senior-level/architecture. Mark which subtopics map directly to my resume.
2. Create a reusable **`research-subtopic` skill/agent** (stored in the repo under `.claude/`) that, given a subtopic:
   - Researches it from authoritative sources (official docs, specs, well-known engineering blogs) and cites them.
   - Writes one Markdown page using a fixed template (below).
   - Targets the latest stable versions (e.g. Java 21/25 LTS, Spring Boot 3.x, React 19) and notes version differences where interviewers ask about them.
3. Scaffold the site: `mkdocs.yml`, `docs/<topic>/<subtopic>.md` structure, navigation, search, code highlighting, Mermaid diagrams, and the Cloudflare Pages build settings (build command, output dir, Python version).
4. Run the skill on one subtopic as a sample, show it to me, and only then batch the rest topic by topic.

**Page template**
- TL;DR (5 bullets)
- Core concepts, explained simply, with diagrams where useful
- Code examples (Java / TypeScript as relevant)
- Trade-offs, pitfalls, and production gotchas
- **How this connects to my experience** (specific resume project/bullet)
- Interview questions: basic → advanced → scenario-based, with model answers
- Follow-up questions an interviewer is likely to chain
- Sources

## Phase 3 — `learn-topic` skill/agent
Create a skill I can invoke as `learn-topic <topic or subtopic>` that teaches the topic in depth and prepares me to answer interview questions on it.

1. **Gather:** read the relevant knowledge-base pages first; research from authoritative sources to fill gaps if a page is missing or thin, and update the page with what it finds.
2. **Explain the topic in detail**, building from first principles to senior/architecture level:
   - Start with the "why": the problem the concept solves and what came before it.
   - Explain internals and how it works under the hood, not just the API surface.
   - **Use diagrams effectively** (Mermaid, so they render on the site): architecture diagrams, sequence diagrams for flows (e.g. OAuth2 authorization code flow, Kafka producer → broker → consumer), state diagrams, and component/data-flow diagrams. Each diagram must have a short caption explaining what to notice.
   - Use analogies and worked examples to make abstract concepts concrete.
   - Use comparison tables for alternatives (e.g. Kafka vs RabbitMQ, REST vs GraphQL, SQL vs NoSQL).
3. **Code samples and real-life examples**, where applicable:
   - Runnable, idiomatic, production-style code (Java 21+/Spring Boot 3.x, TypeScript/React 19), with comments on the key lines.
   - Show the common mistake vs the correct approach side by side where useful.
   - Include config snippets (application.yml, Kafka/Redis/K8s/Terraform) when the topic is configuration-heavy.
   - Real-world scenarios: how large companies use it, well-known production incidents or failure modes, and how it applies in domains I've worked in (healthcare, banking, cloud security).
4. **Connect to my resume:**
   - Talking points that tie the topic to my actual projects.
   - A STAR story (Situation, Task, Action, Result) from my experience.
   - Resume claims on this topic I must be ready to defend, plus the follow-up chains interviewers usually ask.
5. **Frequently asked interview questions:**
   - Grouped by level: Fundamentals → Intermediate → Advanced/Senior → Scenario/Design-based.
   - At least 15–25 questions per main topic, focused on what is actually asked frequently in Senior/Lead interviews.
   - Each with a crisp model answer (diagram or code where it helps), the key points interviewers listen for, and the common wrong answers.
   - Tricky/"gotcha" questions and output-prediction questions where relevant (e.g. Java concurrency, JavaScript event loop).
6. **Quick revision:** end with a one-page cheat sheet (key facts, numbers, defaults, trade-offs) for last-minute review.
7. **Optional mock interview:** ask questions one at a time, evaluate my answers, show what a strong answer would add, and log gaps to a progress tracker page.
8. **Save the output** to the knowledge base (enrich the topic page or add a `learn/` page) so the site improves every time I use it.

---

## Working rules
- Keep everything in the repo; commit after each approved step with clear messages.
- Prefer depth on P0 topics over breadth on P2.
- Ask me questions when something is ambiguous rather than guessing.
- If I share a job description, re-weight priorities against it.

Start with **Phase 1** now.
