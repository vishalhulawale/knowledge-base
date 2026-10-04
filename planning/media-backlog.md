# Media backlog

Researched pages that still need illustrations or animations, per the *Media* section of the `learn-topic` skill (2–3 per page, at least one animated for a process over time). Audited on 2026-10-04 across the 23 topics whose pages are written; the 22 `todo` topics get media when they're researched.

Tick a page off once its SVGs are in `docs/<topic>/images/` and embedded.

## Summary

| Status | Topics |
|---|---|
| **No media** (6 topics, 49 pages; 48 with none) | Kafka (one SVG, on page 02), LLD & Design Patterns, MongoDB, Docker & Kubernetes, Cryptography & Key Management, API Design |
| **Thin pages** (19 pages with one illustration) | Leadership & Behavioral (8), Spring Boot (3), GraphQL (2), JavaScript/TypeScript (2), Distributed Systems, Frontend Architecture, PostgreSQL, React |
| **Done** (2+ per page) | Core Java, Java Concurrency & JVM, Spring Security & OAuth2, Microservices, AWS, System Design, Redis & Caching, JPA & Hibernate, DSA |

## Priority 1: topics without media

Suggested pictures per page; ⏵ marks an animation.

### Kafka (P0)
- [x] 01 Event-driven fundamentals: ⏵ choreography vs orchestration message flow; pub/sub vs queue fan-out
- [x] 02 Architecture: ⏵ ISR and high watermark already there; add partition leaders spread across brokers, and log segments with the active segment
- [x] 03 Producers: ⏵ batching by `linger.ms` / `batch.size`; ⏵ retry without idempotence duplicates vs sequence numbers deduplicate
- [x] 04 Consumers: ⏵ eager vs cooperative rebalance (stop-the-world vs incremental); offsets: committed vs position vs log-end (lag)
- [x] 05 Delivery semantics: ⏵ commit-before vs commit-after processing on crash (lost vs duplicated); transaction markers and `read_committed`
- [x] 06 Ordering and keys: hot key skewing one partition; ⏵ retry reordering with `max.in.flight > 1` and no idempotence
- [x] 07 Error handling: ⏵ non-blocking retry topics with back-off and DLT; poison pill blocking a partition
- [x] 08 Idempotent consumers: ⏵ check-then-insert race vs unique-constraint insert
- [x] 09 Spring Kafka: listener container threads per partition; AckMode commit points on a timeline
- [x] 10 Schema registry: schema id in the wire format; compatibility modes as which side can upgrade first
- [x] 11 Streams and Connect: ⏵ co-partitioned join vs mismatched partitions; Debezium outbox relay path
- [x] 12 Performance and lag: ⏵ lag growing when consume rate < produce rate; where producer latency goes

### LLD & Design Patterns (P0)
- [x] 01 SOLID: LSP violation (Square/Rectangle); composition vs inheritance change ripple
- [ ] 02 Creational: double-checked locking race without `volatile`; builder vs telescoping constructor
- [ ] 03 Structural: ⏵ decorator wrapping order; adapter vs decorator vs proxy wrappers compared
- [ ] 04 Behavioural: ⏵ chain of responsibility handing a request along; state machine transitions
- [ ] 05 UML: the six relationships' arrow notation; aggregation vs composition lifetimes
- [ ] 06 LLD approach: the 7-step process with time budget for a 45-minute round
- [ ] 07 Case studies: ⏵ LRU cache map + doubly linked list on get/put; ⏵ elevator SCAN scheduling

### MongoDB (P0)
- [ ] 01 Document model: embed vs reference; unbounded array growing to the 16 MB limit
- [ ] 02 CRUD and aggregation: ⏵ documents flowing through `$match` → `$group` → `$sort`; array query semantics trap
- [ ] 03 Indexing: ESR rule on a compound index; COLLSCAN vs IXSCAN docs examined
- [ ] 04 Replica sets: ⏵ election and rollback of unreplicated `w:1` writes; write concern `majority` acknowledgment
- [ ] 05 Sharding: ⏵ monotonic shard key hot-spotting the last chunk vs hashed; targeted vs scatter-gather via mongos
- [ ] 06 Transactions: snapshot isolation write skew; transient error label retry loop
- [ ] 07 Spring Data MongoDB: `save()` whole-document overwrite lost update vs `$set`; optimistic locking `@Version` conflict

### Docker & Kubernetes (P0)
- [ ] 01 Containers and images: containers vs VMs stack; layer cache invalidation (copy deps before source)
- [ ] 02 Architecture: ⏵ life of `kubectl apply` through API server, etcd, scheduler, kubelet; reconciliation loop
- [ ] 03 Workloads: ⏵ graceful termination (preStop, SIGTERM, endpoint removal race); StatefulSet ordered pods and stable volumes
- [ ] 04 Networking: ⏵ request path internet → LB → Ingress → Service → pod; NetworkPolicy default deny
- [ ] 05 Config and volumes: ConfigMap as env (frozen) vs mounted (refreshes); PV/PVC/StorageClass binding
- [ ] 06 Probes and autoscaling: ⏵ liveness restart storm vs readiness removing from endpoints; CPU throttling at limit
- [ ] 07 Rollouts and Helm: ⏵ rolling update with `maxSurge` / `maxUnavailable`; rollback as ReplicaSet scale swap
- [ ] 08 EKS/AKS and troubleshooting: VPC CNI IP budget per node; CrashLoopBackOff back-off timeline

### Cryptography & Key Management (P0)
- [ ] 01 Primitives: symmetric vs asymmetric key use; ⏵ AES-GCM nonce reuse leaking XOR of plaintexts
- [ ] 02 TLS: ⏵ TLS 1.3 handshake (1-RTT); certificate chain up to a trusted root
- [ ] 03 Envelope encryption: ⏵ DEK wrapped by KEK, encrypt and decrypt paths; KEK rotation re-wraps only DEKs
- [ ] 04 Cloud KMS: key policy vs IAM policy evaluation; AWS / Azure / GCP key hierarchy side by side
- [ ] 05 HSMs and BYOK/HYOK: who holds the key in KMS vs BYOK vs HYOK; key import wrapping flow
- [ ] 06 Rotation: ⏵ overlap window (add new, switch writers, retire old); JWKS `kid` rotation
- [ ] 07 Passwords and secrets: fast hash vs bcrypt/Argon2 guesses per second; ⏵ rehash on login upgrade

### API Design (P0)
- [ ] 01 REST and HTTP semantics: ⏵ lost update vs `If-Match` / ETag 412; safe vs idempotent methods
- [ ] 02 Status codes and errors: decision tree for 400/401/403/404/409/422; Problem Details anatomy
- [ ] 03 Pagination: ⏵ offset skipping rows on insert vs keyset stable cursor
- [ ] 04 Versioning: expand-and-contract API change; date-based version pinning (Stripe model)
- [ ] 05 Idempotency keys: ⏵ timeout then retry with the same key returning the stored response; concurrent duplicate gets 409
- [ ] 06 Security and rate limiting: ⏵ token bucket refill; BOLA (object-level authz) check
- [ ] 07 OpenAPI: design-first pipeline (spec → lint → codegen → contract test)
- [ ] 08 REST vs GraphQL vs gRPC: same screen as N REST calls vs one GraphQL query vs gRPC stream

## Priority 2: thin pages (one illustration)

- [ ] distributed-systems/01 Fallacies of distributed computing
- [ ] frontend-architecture/04 Design systems and component libraries
- [ ] graphql/01 GraphQL fundamentals vs REST
- [ ] graphql/03 Resolvers and execution model
- [ ] javascript-typescript/05 Equality, coercion, immutability
- [ ] javascript-typescript/09 Output prediction and gotchas
- [ ] leadership-behavioral/04 Mentoring and growing engineers
- [ ] leadership-behavioral/05 Handling conflict and technical disagreements
- [ ] leadership-behavioral/06 Managing underperformance
- [ ] leadership-behavioral/08 Production incidents and postmortems
- [ ] leadership-behavioral/09 Hiring and interviewing others
- [ ] leadership-behavioral/10 Driving engineering standards
- [ ] leadership-behavioral/11 Failures, mistakes and lessons learned
- [ ] leadership-behavioral/12 Questions to ask, salary and role conversations
- [ ] postgresql-sql/05 Normalisation vs denormalisation
- [ ] react/09 Routing and protected routes
- [ ] spring-boot/02 Auto-configuration and starters
- [ ] spring-boot/05 MVC request lifecycle, filters vs interceptors
- [ ] spring-boot/10 Boot 3.x, Jakarta EE, GraalVM, virtual threads

Leadership pages are mostly narrative, so one picture may be enough there; add a second only where it shows something (a timeline, a decision ladder).
