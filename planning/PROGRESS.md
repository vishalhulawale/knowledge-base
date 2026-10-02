# Content progress (saved 2026-10-02)

## P0 done, audited and fact-checked

| Topic | Pages |
|---|---|
| Core Java | 10/10 |
| Java Concurrency & JVM | 10/10 |
| Spring Boot & Spring Core | 10/10 |
| Spring Security, OAuth2/OIDC & JWT | 9/9 |
| Microservices Architecture & Patterns | 12/12 |
| Apache Kafka & Event-Driven Architecture | 12/12 |
| GraphQL | 9/9 |
| ReactJS | 10/10 |
| JavaScript & TypeScript | 9/9 |
| AWS | 12/12 |
| System Design — HLD | 10/10 |
| Low-Level Design & Design Patterns | 7/7 |
| Distributed Systems Concepts | 9/9 |
| Leadership & Behavioral | 12/12 |

## P0 still to write (promoted in the October 2026 audit)

| Topic | Pages |
|---|---|
| Frontend Architecture | 0/6 |
| Redis & Caching | 0/7 |
| MongoDB | 0/7 |
| PostgreSQL / SQL | 0/8 |
| JPA / Hibernate | 0/6 |
| Docker & Kubernetes | 0/8 |
| Cryptography & Key Management | 0/7 |
| Data Structures & Algorithms | 0/11 |
| API Design | 0/8 |

## Not started (P1/P2)
See `planning/phase-1-topics.md` and `planning/topics.json` (status = todo).

## Audit standard for P0 pages (applied 2026-10-02)
- Every question has a model answer, **Interviewer listens for** and **Common wrong answer**.
- At least 12 questions across Fundamentals, Intermediate, Senior and Scenario-based.
- Long inline "(1) (2) (3)" answers are written as numbered lists.
- Code is compiled or run before it is published; version-specific facts are dated.

## How to resume
- Per page: run the `research-subtopic` skill for each `todo` subtopic in `planning/topics.json`.
- After each topic: `python3 .claude/skills/research-subtopic/mark_done.py <topic-slug>`, validate Mermaid, `zensical build`, commit and push.
- Pages contain `*[confirm]*` markers where resume details need Vishal's input: `grep -rn "\[confirm" docs/`.
