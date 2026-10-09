# Phase 2 — Subtopic Tree (APPROVED 2026-10-02)

Order within each topic: fundamentals → advanced → senior/architecture.
★ = directly tied to a resume claim (expect deep follow-ups). Each subtopic becomes one page.

## P0 topics

### 01 Core Java
1. OOP principles, `equals`/`hashCode` contract, immutability
2. String internals (pool, immutability, StringBuilder)
3. Collections framework (List/Set/Map/Queue implementations and complexity)
4. HashMap & ConcurrentHashMap internals
5. Generics (wildcards, PECS, type erasure)
6. Exceptions (checked vs unchecked, try-with-resources, best practices)
7. Functional Java: lambdas, functional interfaces, Streams, Optional
8. Modern Java 9–25: records, sealed classes, pattern matching, switch expressions, text blocks
9. Serialization, reflection & annotations
10. Java memory basics: stack vs heap, pass-by-value, object lifecycle

### 02 Java Concurrency & JVM
1. Threads, lifecycle, `Runnable`/`Callable`
2. Synchronization, `volatile`, Java Memory Model, happens-before
3. Locks: ReentrantLock, ReadWriteLock, StampedLock; deadlock/livelock/starvation
4. Executors & thread pools (sizing, rejection policies, ForkJoinPool)
5. CompletableFuture & async composition
6. Concurrent collections & atomics (CAS, LongAdder)
7. Virtual threads (Project Loom) & structured concurrency
8. JVM architecture: class loading, JIT, memory areas
9. Garbage collection (G1, ZGC, tuning, GC logs)
10. Diagnosing production issues: thread dumps, heap dumps, memory leaks

### 03 Spring Boot & Spring Core
1. IoC & dependency injection, bean scopes & lifecycle
2. Auto-configuration & starters (how Boot works internally)
3. Configuration: properties, profiles, `@ConfigurationProperties`
4. AOP & proxies (JDK vs CGLIB, self-invocation pitfall)
5. Spring MVC request lifecycle, filters vs interceptors, exception handling
6. Transactions: `@Transactional`, propagation, isolation, rollback rules
7. Spring Data (repositories, projections, pagination)
8. Actuator, health checks, metrics (Micrometer)
9. Validation, REST clients (RestClient, WebClient, Feign)
10. Spring Boot 3.x: Jakarta EE, GraalVM native image, virtual threads, observability

### 04 Spring Security, OAuth2/OIDC & JWT
1. Spring Security architecture: filter chain, SecurityContext, authentication providers
2. Authentication vs authorization; method security
3. Sessions vs tokens; CSRF & CORS in Spring
4. JWT: structure, signing (HS256 vs RS256), validation, revocation ★
5. OAuth2 roles & grant types (auth code + PKCE, client credentials, refresh token) ★
6. OpenID Connect: ID token, userinfo, discovery
7. Resource server & client configuration in Spring ★
8. SSO, SAML vs OIDC, enterprise IdPs (PingFederate, Active Directory/Entra ID) ★
9. Service-to-service auth (mTLS, token exchange, propagation through gateways) ★

### 05 Microservices Architecture & Patterns
1. Monolith vs microservices; when not to use microservices
2. Decomposition strategies & monolith migration (strangler fig) ★
3. API gateway & BFF pattern
4. Service discovery & client-side load balancing
5. Inter-service communication: sync vs async
6. Resilience: circuit breaker, retry, bulkhead, timeout, rate limiter (Resilience4j)
7. Distributed transactions: Saga (choreography vs orchestration), outbox pattern
8. CQRS & event sourcing
9. Configuration & secrets management (Spring Cloud Config, Vault)
10. Distributed tracing & correlation IDs
11. Deployment strategies: blue-green, canary, feature flags
12. Microservice testing strategy & contract testing

### 06 Apache Kafka & Event-Driven Architecture
1. Event-driven architecture fundamentals (events vs commands, pub/sub, choreography)
2. Kafka architecture: brokers, topics, partitions, replication, ISR, KRaft
3. Producers: acks, batching, idempotent producer, keys & partitioning
4. Consumers: consumer groups, rebalancing, offset management, commit strategies
5. Delivery semantics: at-most/at-least/exactly-once, transactions ★
6. Ordering guarantees & partition key design ★
7. Error handling: retry topics, DLQ, poison messages, replay ★
8. Idempotent consumers & deduplication ★
9. Spring Kafka (listeners, error handlers, `@RetryableTopic`) ★
10. Schema management: Avro/Protobuf, Schema Registry, compatibility
11. Kafka Streams & Kafka Connect (overview)
12. Performance tuning, consumer lag & monitoring

### 07 GraphQL
1. GraphQL fundamentals vs REST; schema, types, queries, mutations, subscriptions
2. Schema design best practices (nullability, pagination, errors, versioning) ★
3. Resolvers & execution model; Spring for GraphQL / DGS ★
4. N+1 problem & DataLoader batching ★
5. Aggregating multiple upstream systems: orchestration, timeouts, partial failures ★
6. Federation vs schema stitching ★
7. Caching (client, server, persisted queries) ★
8. Security: auth, query depth/complexity limits, introspection ★
9. Performance & observability of GraphQL services

### 08 ReactJS
1. JSX, components, props vs state, reconciliation & virtual DOM
2. Hooks deep dive: useState, useEffect, useRef, useMemo, useCallback, custom hooks
3. Rendering behaviour: re-render triggers, keys, batching
4. Context API & component composition patterns
5. Forms & controlled vs uncontrolled components
6. Performance optimisation: memoization, code splitting, lazy loading, virtualization
7. Error boundaries & Suspense
8. React 18/19: concurrent rendering, transitions, Actions, `use`, React Compiler
9. Routing (React Router) & auth-protected routes ★
10. Testing React (Jest, React Testing Library)

### 09 JavaScript & TypeScript
1. Scope, hoisting, closures, `this`, prototypes & inheritance
2. Event loop, microtasks vs macrotasks
3. Promises, async/await, error handling
4. ES6+ features (destructuring, modules, spread, optional chaining, iterators/generators)
5. Equality, type coercion, immutability, shallow vs deep copy
6. Debounce, throttle, memoization & common JS coding questions
7. TypeScript type system: interfaces vs types, generics, unions, narrowing
8. Advanced TypeScript: utility, mapped & conditional types
9. Output-prediction & "gotcha" questions

### 15 AWS
1. Global infrastructure, Well-Architected Framework
2. IAM: users, roles, policies, least privilege ★
3. Compute: EC2, Auto Scaling, ELB/ALB/NLB
4. Containers: ECS vs EKS vs Fargate ★
5. Serverless: Lambda (cold starts, concurrency, limits), API Gateway ★
6. Storage: S3 (classes, consistency, security), EBS vs EFS ★
7. Databases: RDS/Aurora, DynamoDB (keys, GSI/LSI, capacity) ★
8. Messaging: SQS vs SNS vs EventBridge vs Kinesis ★
9. Networking: VPC, subnets, security groups vs NACLs, Route 53, CloudFront
10. Security: KMS, Secrets Manager, encryption at rest/in transit ★
11. Observability: CloudWatch, X-Ray
12. Architecture scenarios: HA, DR strategies, cost optimisation

### 24 System Design — HLD
1. Approach & framework for the design interview
2. Back-of-the-envelope estimation
3. Scalability: vertical vs horizontal, stateless services, load balancing
4. Caching strategies & CDN
5. Database scaling: replication, sharding, partitioning, consistent hashing
6. CAP & PACELC, consistency models
7. Message queues & async processing
8. Rate limiting & API gateway design
9. Availability, fault tolerance, disaster recovery
10. Case studies: URL shortener, rate limiter, notification system, news feed, chat, payment system, healthcare/prescription platform ★

### 25 Low-Level Design & Design Patterns
1. SOLID, DRY, KISS, YAGNI, composition over inheritance
2. Creational patterns (Singleton, Factory, Builder, Prototype)
3. Structural patterns (Adapter, Decorator, Proxy, Facade, Composite)
4. Behavioural patterns (Strategy, Observer, Template, Chain of Responsibility, Command, State)
5. UML & class diagram basics for interviews
6. LLD approach: requirements → entities → relationships → APIs
7. Case studies: parking lot, LRU cache, rate limiter, elevator, splitwise, library system

### 26 Distributed Systems Concepts
1. Fallacies of distributed computing, failure modes
2. Consistency models & replication (leader/follower, quorum)
3. Consensus basics (Raft, leader election)
4. Idempotency & idempotency keys ★
5. Retries, backoff, jitter, timeouts ★
6. Distributed transactions: 2PC vs Saga
7. Exactly-once processing in practice ★
8. Clocks, ordering, distributed locks
9. Backpressure & load shedding

### 31 Leadership & Behavioral
1. STAR framework & building a story bank ★
2. "Tell me about yourself" & project deep-dive narrative (OptumRx Meteor) ★
3. Leading a team of 8–10: delegation, ownership, accountability ★
4. Mentoring & growing engineers ★
5. Handling conflict & technical disagreements (incl. with architects) ★
6. Managing underperformance & difficult conversations
7. Estimation, deadlines & stakeholder management ★
8. Production incidents & postmortems ★
9. Hiring & interviewing others ★
10. Driving engineering standards & technical decisions ★
11. Failures, mistakes & lessons learned
12. Questions to ask the interviewer; salary/role conversations

## P1 topics

### 10 Frontend Architecture
1. Micro-frontends: approaches (Module Federation, single-spa, iframes) & trade-offs ★
2. Shared dependencies, routing & communication between micro-frontends ★
3. State management: Redux Toolkit, React Query/TanStack Query, when to use which ★
4. Design systems & component libraries (Storybook, MUI) ★
5. Frontend security (XSS, token storage, CSP) ★
6. Frontend CI/CD & monorepos

### 11 Redis & Caching
1. Redis data structures & use cases
2. Caching patterns: cache-aside, read/write-through, write-behind ★
3. TTL, eviction policies & invalidation strategies ★
4. Cache stampede, penetration & avalanche ★
5. Spring Cache abstraction with Redis ★
6. Persistence (RDB/AOF), replication, Sentinel & Cluster
7. Distributed locks, rate limiting & pub/sub with Redis

### 12 MongoDB
1. Document model & schema design (embed vs reference) ★
2. CRUD, query operators & aggregation pipeline
3. Indexing (compound, multikey, TTL) & explain plans ★
4. Replica sets, read/write concerns
5. Sharding & shard key selection
6. Transactions & consistency
7. Spring Data MongoDB ★

### 13 PostgreSQL / SQL
1. SQL essentials: joins, group by, window functions, CTEs
2. Indexes (B-tree, hash, GIN, partial, composite) & EXPLAIN ANALYZE
3. Transactions, ACID & isolation levels (anomalies)
4. MVCC, locking & deadlocks
5. Normalisation vs denormalisation
6. Query optimisation & common performance issues
7. Partitioning, replication & connection pooling (HikariCP)
8. Common SQL interview queries (nth highest salary, etc.)

### 14 JPA / Hibernate
1. Entity lifecycle & persistence context
2. Relationships & fetching (lazy vs eager)
3. N+1 problem & solutions (fetch joins, entity graphs, batch size)
4. First- and second-level cache
5. Optimistic vs pessimistic locking
6. Schema migration with Liquibase/Flyway ★

### 16 Docker & Kubernetes
1. Containers vs VMs; Docker images, layers, multi-stage builds
2. Kubernetes architecture: control plane, nodes, etcd
3. Pods, Deployments, ReplicaSets, StatefulSets, DaemonSets
4. Services, Ingress & networking
5. ConfigMaps, Secrets & volumes
6. Probes, resource requests/limits, autoscaling (HPA)
7. Rolling updates, rollbacks & Helm
8. EKS/AKS specifics & troubleshooting pods ★

### 18 Testing Strategy
1. Test pyramid & testing strategy for microservices ★
2. JUnit 5 & Mockito
3. Spring Boot test slices & integration tests
4. Testcontainers
5. Contract testing (Spring Cloud Contract, Pact)
6. Frontend testing: Jest, React Testing Library, E2E (Playwright/Cypress) ★
7. TDD, coverage & quality gates (SonarQube) ★

### 19 Cryptography & Key Management
1. Symmetric vs asymmetric encryption, hashing, MAC, digital signatures
2. TLS & certificates (PKI, mTLS)
3. Envelope encryption & data keys ★
4. Cloud KMS (AWS KMS, Azure Key Vault, GCP KMS) ★
5. HSMs & BYOK/HYOK ★
6. Key rotation strategies without downtime ★
7. Password hashing & secrets management ★

### 27 Data Structures & Algorithms
1. Complexity analysis (Big-O)
2. Arrays & strings (two pointers, sliding window, prefix sums)
3. Hashing patterns
4. Linked lists, stacks & queues
5. Trees & BST; tree traversal patterns
6. Heaps & priority queues (top-K)
7. Graphs: BFS, DFS, topological sort, shortest path
8. Binary search patterns
9. Recursion & backtracking
10. Dynamic programming patterns
11. Greedy & intervals

### 28 API Design
1. REST principles, resource modelling & HTTP semantics
2. Status codes & error format (RFC 9457 problem details)
3. Pagination, filtering & sorting
4. Versioning strategies
5. Idempotency keys & safe retries ★
6. API security & rate limiting ★
7. OpenAPI & API-first development
8. REST vs GraphQL vs gRPC ★

### 29 Observability & Production Support
1. Logs, metrics & traces (three pillars)
2. Structured logging & correlation IDs
3. Metrics with Micrometer/Prometheus & dashboards (Grafana)
4. Distributed tracing with OpenTelemetry
5. SLIs, SLOs, SLAs & error budgets
6. Alerting & on-call
7. Incident management & postmortems ★

### 30 Application Security
1. OWASP Top 10
2. XSS, CSRF, SQL injection & prevention
3. CORS explained
4. Input validation & output encoding
5. Secrets management & secure configuration ★
6. Security headers & dependency scanning
7. Threat modelling basics

### 32 Domain-Driven Design & Clean Architecture
1. Strategic DDD: bounded contexts, ubiquitous language, context mapping
2. Tactical DDD: entities, value objects, aggregates, domain events
3. Hexagonal / ports-and-adapters & clean architecture
4. Using DDD to define microservice boundaries ★

### 33 Performance Engineering
1. Performance methodology: measure, profile, fix, verify
2. JVM profiling (JFR, async-profiler, VisualVM)
3. Memory leaks & GC tuning in practice
4. Database & query performance
5. Load testing (JMeter, Gatling, k6) & capacity planning
6. Latency: percentiles, tail latency

### 34 Networking & Web Fundamentals
1. OSI/TCP-IP, TCP vs UDP
2. HTTP/1.1 vs HTTP/2 vs HTTP/3
3. TLS handshake
4. DNS resolution
5. Load balancers (L4 vs L7), reverse proxies & CDNs
6. "What happens when you type a URL"

### 36 GenAI for Engineers
1. LLM fundamentals: tokens, context window, temperature, hallucination
2. Prompt engineering & structured output
3. Embeddings & vector databases
4. RAG architecture
5. Spring AI & LLM API integration
6. Agents & tool calling
7. AI-assisted development, evaluation & responsible AI

### 37 Browser, HTML/CSS & Web Performance
1. Browser rendering pipeline, reflow vs repaint
2. HTML semantics & accessibility (WCAG, ARIA)
3. CSS layout: box model, flexbox, grid, specificity
4. Core Web Vitals & performance optimisation
5. Bundlers & build tooling (Vite, Webpack)
6. Browser storage, cookies & caching

### 40 Data Modelling & Database Selection
1. SQL vs NoSQL: choosing the right database ★
2. Data modelling for relational, document, key-value & wide-column stores
3. Replication, partitioning & indexing trade-offs
4. Schema evolution & zero-downtime migrations ★
5. Polyglot persistence in microservices ★

### 42 Engineering Practices
1. Git internals & branching strategies (GitFlow vs trunk-based)
2. Code review best practices ★
3. Technical debt management
4. Architecture Decision Records & documentation
5. DORA metrics & engineering effectiveness ★

## P2 topics (light coverage — 2 to 4 pages each)

### 17 CI/CD & DevOps
1. CI/CD pipeline design (GitLab CI, Jenkins, GitHub Actions) ★
2. Terraform fundamentals: state, modules, plan/apply ★
3. Infrastructure as Code best practices
4. GitOps & deployment strategies

### 20 Azure
1. Core services mapped to AWS equivalents
2. AKS, App Service & Azure Functions
3. Entra ID & Key Vault ★

### 21 Messaging Alternatives
1. RabbitMQ: exchanges, queues, routing
2. Kafka vs RabbitMQ vs SQS — choosing ★

### 22 Elasticsearch & DynamoDB
1. Elasticsearch: inverted index, analyzers, relevance, sharding ★
2. DynamoDB data modelling & single-table design ★

### 23 Python
1. Python essentials for Java developers
2. Common Python interview questions

### 35 Reactive Programming & Spring WebFlux
1. Reactive Streams & Project Reactor (Mono/Flux)
2. Backpressure
3. WebFlux vs MVC vs virtual threads — when to choose

### 38 Next.js & SSR
1. Rendering strategies: CSR, SSR, SSG, ISR
2. App Router & React Server Components
3. Data fetching & caching in Next.js

### 39 Real-time & Inter-service Communication
1. gRPC & Protocol Buffers
2. WebSockets vs SSE vs long polling
3. Webhooks design

### 41 Service Mesh & Cloud-Native Platform
1. Service mesh concepts (Istio/Linkerd), sidecar vs ambient
2. 12-factor apps & cloud-native principles

### 43 Agile Delivery & Estimation
1. Scrum vs Kanban
2. Estimation techniques & planning ★
3. Release & risk management ★

### 44 Compliance & Data Privacy
1. HIPAA & PHI handling ★
2. PCI-DSS, GDPR & PII
3. Audit logging & data retention

## Forward Deployed Engineer track

### 45 FDE Role & Interview Loop
1. What an FDE is: Palantir origins (Echo/Delta), FDE vs SWE vs solutions engineer vs consultant
2. The AI-lab FDE model: OpenAI, Anthropic, Google, Databricks, Scale & services-led growth
3. The FDE interview loop mapped: screens, take-home, practical coding, decomposition, learning, customer simulation, behavioral
4. Positioning a consulting & services background (Publicis Sapient, Deloitte) for FDE ★
5. "Why FDE, why this company", travel, level & compensation conversations

### 46 Problem Decomposition & Scoping
1. Framework for ambiguous prompts: users, decisions, data, constraints, success metrics
2. From vague business goal to data & object model (ontology thinking)
3. Decomposing into a working, extensible MVP in a live pairing session
4. Prioritisation, trade-off calls & cutting scope under time pressure ★
5. Worked decomposition prompts: scheduling, logistics, marketplace, operations dashboard
6. The learning round: picking up an unfamiliar API, language or library fast

### 47 Customer Discovery & Stakeholder Management
1. Discovery interviews: workflow mapping, hidden constraints, restating the real need
2. Writing the scope brief: success criteria, assumptions, out of scope ★
3. Pilot → proof of concept → production: time-boxing, exit criteria, measuring business impact
4. Saying no & managing scope creep without losing trust ★
5. Talking to executives vs engineers: demos, executive pitch, status updates ★
6. Customer simulation round: role-play scenarios & how they are scored
7. Field feedback to product & research; codifying repeatable deployment patterns

### 48 Practical Coding for FDE
1. Practical coding round format: narrating, edge cases first, using AI tools in the round
2. Third-party API integration: auth, pagination, rate limits, retries with backoff ★
3. Webhooks: signature verification, idempotent consumers, duplicate & out-of-order delivery ★
4. Debugging & reading an unfamiliar codebase fast
5. Refactoring & extending messy code without breaking its tests
6. Python fluency for FDE: scripting, data wrangling, FastAPI services
7. Rapid full-stack prototyping: TypeScript service plus React UI for a demo ★

### 49 Data Integration & Pipelines
1. Integrating with legacy systems of record: databases, files/SFTP, SOAP/REST, CDC ★
2. ETL vs ELT, batch vs streaming, orchestration (Airflow, Dagster) & dbt
3. SQL for take-homes: multi-table joins, window functions, NULL handling, deduplication
4. Data quality, schema drift, idempotent loads & backfills ★
5. Semantic layer & ontology: modelling the customer's domain (Foundry-style)
6. Spark/PySpark & lakehouse basics (Databricks)

### 50 Applied LLM Engineering for Deployments
1. Model selection & routing: quality vs latency vs cost across providers
2. Production prompt engineering, structured outputs & prompt caching
3. Production RAG: chunking, hybrid search, reranking, permission-aware retrieval
4. Agents in production: tool use, MCP servers, sub-agents, skills, human-in-the-loop
5. Evals: golden datasets, LLM-as-judge, retrieval vs answer metrics, regression tests in CI
6. Guardrails: prompt injection, PII/PHI redaction, grounding checks, audit logging ★
7. LLM observability, latency budgets & token cost control ★
8. Prompting vs RAG vs fine-tuning: choosing the right lever

### 51 Enterprise Deployment Environments
1. Deployment models: hosted API vs customer VPC vs on-prem & air-gapped ★
2. Enterprise identity: SSO (SAML/OIDC), SCIM, RBAC & permission propagation ★
3. Network & data constraints: private endpoints, proxies, egress allowlists, data residency ★
4. Security reviews & compliance questionnaires: SOC 2, HIPAA, GDPR, DPAs ★
5. Packaging & delivery into a customer account: Docker, Helm, Terraform ★
6. Cloud AI platforms: Amazon Bedrock, Azure OpenAI, Vertex AI ★
7. Production rollout, monitoring, on-call & handoff to the customer's team ★

### 52 FDE System Design
1. FDE system design framework: customer context, deployment constraint, rollout plan
2. Case: enterprise knowledge assistant (RAG over internal docs with access control)
3. Case: agentic workflow automation with human approval (claims or ticket triage) ★
4. Case: document extraction pipeline (forms, invoices) with evals
5. Case: customer-support agent with escalation & handoff
6. Case: unifying siloed systems into an operational dashboard ★
7. Defending trade-offs: build vs buy, latency vs cost, cloud vs on-prem

### 53 FDE Behavioral, Take-Home & Deep Dive
1. FDE story bank: ambiguity, no-docs systems, scope pushback, client conflict, quick fix vs proper fix ★
2. Owning a customer outcome end to end ★
3. Recovering trust after a failed delivery or incident ★
4. Take-home project: building, writing up & presenting it
5. Project deep dive: defending every line of the resume ★
