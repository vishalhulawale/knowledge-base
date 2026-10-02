# Content progress (saved 2026-10-02)

## Done and fact-checked (committed)
| Topic | Pages |
|---|---|
| Kafka | 12/12 |
| GraphQL | 9/9 |
| Core Java | 10/10 |
| Java Concurrency & JVM | 10/10 |
| Spring Boot & Spring Core | 10/10 |

## In progress
| Topic | Pages | Next step |
|---|---|---|
| Spring Security, OAuth2/OIDC & JWT | 8/9 (all fact-checked) | Write and check 09 (service-to-service auth) |

## Not started (P0)
Microservices (12), ReactJS (10), JavaScript & TypeScript (9), AWS (12), System Design (10), LLD & Design Patterns (7), Distributed Systems (9), Leadership & Behavioral (12)

## Not started (P1/P2)
See `planning/phase-1-topics.md` and `planning/topics.json` (status = todo).

## How to resume
- Per page: run the `research-subtopic` skill for each `todo` subtopic in `planning/topics.json`.
- Review pass for draft pages: fact-check against official docs, fix, validate Mermaid (`.claude/skills/research-subtopic/validate_mermaid.sh`), then delete the `!!! warning "Draft: not yet fact-checked"` block.
- After each topic: `bash planning/finish_topic.sh <topic-slug>` (syncs nav/status, test-builds, commits).
- Pages contain `*[confirm]*` markers where resume details need Vishal's input: `grep -rn "\[confirm" docs/`.
