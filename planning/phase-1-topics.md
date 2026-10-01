# Phase 1 — Main Topic List (APPROVED 2026-10-02)

Target roles: Senior / Lead / Staff Engineer (Java full-stack, cloud-native).
Depth: **Expert** = can go deep on internals and design trade-offs · **Working** = solid hands-on answers · **Awareness** = can explain and compare.

## A. Core technologies (from resume)

| # | Topic | Why it matters (resume evidence) | Priority | Depth |
|---|---|---|---|---|
| 1 | Core Java (OOP, Collections, Generics, Java 8–21 features) | Primary language across all 4 roles | P0 | Expert |
| 2 | Java Concurrency & JVM (threads, executors, CompletableFuture, virtual threads, memory model, GC) | Standard senior Java round; high-throughput services serving 750K+ users | P0 | Expert |
| 3 | Spring Boot & Spring Core (IoC/DI, auto-config, AOP, profiles, actuator) | Every project since 2017 | P0 | Expert |
| 4 | Spring Security, OAuth2/OIDC & JWT | OAuth2 + PingFederate + AD at Optum; JWT/SSO ownership at Johnson Controls | P0 | Expert |
| 5 | Microservices architecture & patterns (API gateway, saga, CQRS, circuit breaker, service discovery, Spring Cloud) | Monolith-to-microservices migration; microservices at Optum and Deloitte | P0 | Expert |
| 6 | Apache Kafka & event-driven architecture | Kafka workflows with retry + DLQ handling (Optum); event-driven analytics (Deloitte) | P0 | Expert |
| 7 | GraphQL (schema design, resolvers, N+1/DataLoader, federation, security) | Owned the GraphQL Consumer Service over 5 upstream systems | P0 | Expert |
| 8 | ReactJS (hooks, rendering, state, performance, React 19) | Built the React app from scratch (Optum) | P0 | Expert |
| 9 | JavaScript & TypeScript fundamentals (closures, event loop, promises, `this`, types) | Required for every front-end round | P0 | Expert |
| 10 | Frontend architecture (micro-frontends, Redux, React Query, Storybook, MUI, web performance) | Established micro-frontend architecture | P1 | Working → Expert |
| 11 | Redis & caching strategies | Redis caching for queries and reference data | P1 | Working → Expert |
| 12 | MongoDB (data modelling, indexing, aggregation, transactions) | Primary datastore at Optum | P1 | Working |
| 13 | PostgreSQL / SQL (indexing, query plans, transactions, isolation levels) | Listed skill; SQL is asked in almost every backend loop | P1 | Working → Expert |
| 14 | JPA / Hibernate (entity lifecycle, N+1, caching, locking) | Listed skill; common Spring follow-up | P1 | Working |
| 15 | AWS (compute, Lambda, ECS/EKS, API Gateway, S3, SQS/SNS, DynamoDB, RDS, IAM, KMS) | AWS SA certification; Deloitte cloud-native platform | P0 | Expert |
| 16 | Docker & Kubernetes (EKS/AKS) | Listed skill; deployments on EKS/AKS | P1 | Working |
| 17 | CI/CD & DevOps (GitLab CI, Jenkins, Terraform, deployment strategies) | Set engineering and CI/CD standards; Terraform at Deloitte | P2 | Working |
| 18 | Testing strategy (JUnit 5, Mockito, Testcontainers, Jest, React Testing Library, contract tests) | Set testing standards at Optum | P1 | Working |
| 19 | Cryptography & key management (encryption, envelope encryption, KMS, HSM, key rotation) | CipherTrust CCKM — 3 years of key management work | P1 | Working → Expert |
| 20 | Azure (core services, AKS) | Listed skill | P2 | Awareness |
| 21 | Messaging alternatives: RabbitMQ, SQS/SNS | Listed; "Kafka vs RabbitMQ vs SQS" is a common question | P2 | Awareness |
| 22 | Elasticsearch & DynamoDB | Search and NoSQL at Deloitte | P2 | Awareness |
| 23 | Python | Listed language | P2 | Awareness |

## B. Cross-cutting areas (expected at Lead level)

| # | Topic | Why it matters | Priority | Depth |
|---|---|---|---|---|
| 24 | System Design — HLD (scalability, CAP, sharding, caching, rate limiting, classic designs) | Dedicated round for every Senior/Lead role | P0 | Expert |
| 25 | Low-Level Design & design patterns (SOLID, GoF, clean code) | Machine-coding / LLD round | P0 | Expert |
| 26 | Distributed systems concepts (consistency, idempotency, exactly-once, retries, distributed transactions) | Underpins Kafka, microservices and system design answers | P0 | Expert |
| 27 | Data Structures & Algorithms | Coding rounds at most product companies | P1 (P0 for product companies) | Working |
| 28 | API design (REST best practices, versioning, pagination, error handling, idempotency keys) | Built secure enterprise APIs; integration layer | P1 | Expert |
| 29 | Observability & production support (logging, metrics, tracing, incident handling) | Owned production support and critical services | P1 | Working |
| 30 | Application security (OWASP Top 10, CORS, CSRF, XSS, secrets management) | Healthcare/banking/security domains | P1 | Working |
| 31 | Leadership & behavioral (STAR stories, mentoring, conflict, estimation, hiring, stakeholder management) | Led 8–10 engineers, mentored 5+, ran interviews | P0 | Expert |

## C. Resume claims likely to draw deep follow-ups

1. **"Owned the GraphQL Consumer Service … between 5 upstream systems"** — schema stitching vs federation, N+1, partial failures and timeouts across upstreams, caching, auth propagation.
2. **"Kafka workflows with retry and DLQ handling"** — retry topics vs blocking retries, ordering guarantees, idempotent consumers, poison messages, DLQ replay, offset commits.
3. **"Redis-based caching"** — cache-aside vs write-through, TTL and invalidation, stampede/thundering herd, consistency with MongoDB.
4. **"Established a micro-frontend architecture"** — Module Federation vs other approaches, shared dependencies, routing, independent deployment, why not a monolith.
5. **"OAuth2, PingFederate, Active Directory"** — which grant types and why, token validation, refresh tokens, PKCE, resource server setup.
6. **"Serving 750K+ users"** — expected traffic, latency, scaling approach, bottlenecks you hit.
7. **"Automated key rotation workflows and HSM integrations"** — envelope encryption, rotation without downtime, BYOK/HYOK across clouds.
8. **"Led a team of 8–10 / mentored 5+ / conducted interviews"** — concrete stories: conflict, underperformer, missed deadline, technical disagreement with an architect.
9. **"Migration of a monolith to microservices"** — strangler fig, data decomposition, what went wrong.
10. **AWS Solutions Architect certification** — expect AWS architecture scenario questions.

## E. Additional topics commonly expected at 9–10 years (not on resume)

| # | Topic | Why interviewers ask it at this level | Priority | Depth |
|---|---|---|---|---|
| 32 | Domain-Driven Design & clean/hexagonal architecture | Bounded contexts, aggregates and service boundaries come up in every microservices/design discussion with leads | P1 | Working → Expert |
| 33 | Performance engineering (JVM tuning, profiling, heap/thread dumps, load testing, latency analysis) | "How did you find and fix a slow/leaking service?" is a standard senior question | P1 | Working |
| 34 | Networking & web fundamentals (HTTP/1.1 vs 2 vs 3, TLS handshake, DNS, load balancers, CDNs, proxies) | Basis for system design and API questions; "what happens when you type a URL" | P1 | Working |
| 35 | Reactive programming & Spring WebFlux (Project Reactor, backpressure, when to choose it vs virtual threads) | Common Spring follow-up; trade-off question in 2026 with virtual threads | P2 | Working |
| 36 | GenAI for engineers (LLM APIs, RAG, embeddings/vector DBs, Spring AI, AI-assisted development) | Increasingly asked in 2026 for lead roles: integrating AI features and using AI tools responsibly | P1 | Working |
| 37 | Browser, HTML/CSS & web performance (rendering pipeline, Core Web Vitals, accessibility, bundling with Vite/Webpack) | Senior frontend rounds go beyond React into how the browser works | P1 | Working |
| 38 | Next.js & server-side rendering (SSR/SSG, React Server Components) | Most React job descriptions in 2026 list Next.js or RSC | P2 | Working |
| 39 | Real-time & inter-service communication (gRPC/Protobuf, WebSockets, SSE, webhooks) | "REST vs GraphQL vs gRPC" and real-time design questions | P2 | Working |
| 40 | Data modelling & database selection (SQL vs NoSQL, normalisation, partitioning, replication, migrations) | Core of system design; you've used 5 databases, so expect "why this one?" | P1 | Expert |
| 41 | Service mesh & cloud-native platform (Istio/Linkerd, API gateways, config/secrets, 12-factor apps) | Natural follow-up to Kubernetes and microservices at lead level | P2 | Awareness |
| 42 | Engineering practices (Git branching/trunk-based, code review, tech debt management, ADRs, DORA metrics) | Leads are asked how they run engineering quality, not just write code | P1 | Working |
| 43 | Agile delivery & estimation (Scrum/Kanban, story points, risk management, release planning) | You led sprint planning and releases — expect process questions | P2 | Working |
| 44 | Compliance & data privacy (HIPAA, PCI-DSS, GDPR, PII/PHI handling, audit logging) | Your healthcare and banking domains make this a likely domain question | P2 | Awareness |

## D. Notes for review
- Resume says **9+ years**; aligned target level is Senior/Lead. Tell me if you're also targeting Staff/Architect, which would raise System Design depth further.
- Share a target job description (or company type: product vs services) to re-weight DSA and cloud priorities.
- Topics 20–23 can be dropped or folded into others if time is short.
