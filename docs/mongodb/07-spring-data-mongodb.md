---
title: "Spring Data MongoDB"
description: "Using MongoDB from Spring Boot 3: mapping (@Document, @Field, _class, Decimal128), repositories (derived queries, @Query, @Aggregation, projections, Page vs Slice), MongoTemplate and Criteria/Update for atomic operations, auditing and @Version optimistic locking, transactions with MongoTransactionManager, index management (auto-index-creation is off by default), and the traps of save(), all verified with Spring Boot 3.5 tests against a MongoDB 8.0 replica set."
tags: [mongodb, P0]
---

# Spring Data MongoDB

!!! abstract "TL;DR"
    - Spring Data MongoDB maps classes to documents (`@Document`, `@Id`, `@Field`) and gives you two APIs: **repositories** (derived queries, `@Query`, `@Aggregation`, projections, paging) and **`MongoTemplate`** (`Query`/`Criteria`/`Update`, aggregations, bulk ops) for anything precise or atomic.
    - **`@Indexed(unique = true)` does nothing by default**: `auto-index-creation` has been `false` since Spring Data MongoDB 3.0. Measured: only `_id_` existed and a **duplicate `claimNo` was accepted**. With indexes created, the duplicate threw `DuplicateKeyException`. Create indexes through migrations or `IndexOperations` at startup.
    - **`save()` replaces the whole document**: a field written by another service disappeared after `save()`, while `updateFirst` with `$set` kept it. Use `@Version` to stop lost updates (a stale save threw `OptimisticLockingFailureException`) and `Update` operations for partial changes. 500 find-and-save calls took **2.3 s**, and one `updateMulti` over 6,665 docs took **0.34 s**.
    - Map money as **Decimal128** (`@Field(targetType = DECIMAL128)` or `BigDecimalRepresentation.DECIMAL128`). The default stores `BigDecimal` as a **string** in Spring Data MongoDB 4.x.
    - **`Page` runs an extra count query**: 50 pages took 1,534 ms with `Page` vs 773 ms with `Slice`. **Transactions** need a `MongoTransactionManager` bean (a rollback undid an insert and an update), and they need retries for `TransientTransactionError`.

## Why it matters

Most Java teams use MongoDB through Spring Data, so interviewers test whether you know what the abstraction does underneath: which query a derived method generates, why `save()` can lose data, how optimistic locking works, why indexes declared on entities don't exist in production, and when to drop to `MongoTemplate`. These are also exactly the bugs that appear in real services.

Every behaviour on this page was verified with a Spring Boot 3.5.6 / Spring Data MongoDB 4.5.4 test project against a local MongoDB 8.0.4 replica set, with 20,000 claims, while writing this page.

## Core concepts

### Architecture

```mermaid
flowchart LR
    S["Service"] --> R["ClaimRepository<br/>(MongoRepository proxy)"]
    S --> T["MongoTemplate"]
    R --> T
    T --> C["MappingMongoConverter<br/>entity ↔ BSON, conversions, _class"]
    C --> D["MongoDB Java driver<br/>(sync or reactive)"]
    D --> M[("Replica set")]
    TX["MongoTransactionManager"] -. "binds ClientSession" .-> T
```
*Notice that repositories are a convenience layer over `MongoTemplate`. Anything a repository can't express (atomic updates, bulk operations, complex aggregations) you do with the template directly.*

### Mapping

```java
@Document("claims")                                         // collection name
@CompoundIndex(name = "member_date", def = "{'memberId': 1, 'serviceDate': -1}")
public class Claim {
    @Id String id;                                          // ObjectId ↔ String
    @Indexed(unique = true) String claimNo;                 // only created if index creation runs!
    String memberId;
    String status;
    @Field(targetType = FieldType.DECIMAL128) BigDecimal amount;   // money as Decimal128
    Instant serviceDate;
    List<Line> lines;                                       // embedded documents
    @Version Long version;                                  // optimistic locking
    @CreatedDate Instant createdAt;                         // auditing (@EnableMongoAuditing)
    @LastModifiedDate Instant updatedAt;
}
```

Stored document (measured):

```json
{"_id": {"$oid": "6ac0960e…"}, "claimNo": "C1", "memberId": "M1", "status": "DENIED",
 "amount": {"$numberDecimal": "393.80"}, "serviceDate": {"$date": "2026-06-04T00:00:00Z"},
 "lines": [{"ndc": "N18", "qty": 60, "paid": {"$numberDecimal": "10"}}],
 "version": 0, "createdAt": {"$date": "…"}, "updatedAt": {"$date": "…"}, "_class": "kb.App$Claim"}
```

- **`_class`** stores the Java type for polymorphic reads. It couples data to class names (refactors break reads of old documents), so set `@TypeAlias("claim")`, or disable it with a custom `MappingMongoConverter`/`DefaultMongoTypeMapper(null)` if you don't need polymorphism.
- **BigDecimal:** Spring Data MongoDB 4.x defaults to storing `BigDecimal` as a **string** (which compares and sorts lexically). Use `@Field(targetType = DECIMAL128)` per field or `MongoCustomConversions.create(c -> c.bigDecimal(BigDecimalRepresentation.DECIMAL128))` globally.
- **Records and immutable classes** work through constructor binding. `@Version` and auditing fields need to be settable (or use `with…` methods).

### Repositories

| Feature | Example | Generated or behaviour (measured) |
|---|---|---|
| Derived query | `findByMemberIdAndStatusOrderByServiceDateDesc(m, s)` | `{memberId: m, status: s}` sorted by date: 4 results |
| `@Query` | `@Query(value = "{ 'lines': { $elemMatch: { 'ndc': ?0, 'qty': { $gte: ?1 } } } }", fields = "{ 'claimNo': 1, 'amount': 1 }")` | 71 results. Unprojected fields come back **null** |
| Interface projection | `List<ClaimSummary> findByMemberId(m)` | JDK proxy (`$Proxy130`) exposing only `getClaimNo()`/`getAmount()`, and only those fields are fetched |
| `@Aggregation` | `$match` → `$group` → `$sort` → `$limit` | Mapped to `record MemberTotal(@Id String memberId, BigDecimal total)` |
| Paging | `Page<Claim> findByStatus(s, pageable)` | Find + **count** query |
| Slicing | `Slice<Claim> findSliceByStatus(s, pageable)` | Fetches size + 1 to know if there's a next page, **no count** |
| Count | `long countByStatus(s)` | `countDocuments` |

**Page vs Slice, measured:** 50 sequential pages of 20 PAID claims took **1,534 ms** with `Page` and **773 ms** with `Slice`. The count runs on every page request. Use `Slice` (or keyset pagination on an indexed field, via `ScrollPosition`/`Window` in Spring Data 3.1+) for infinite scroll, and `Page` only when the UI really shows a total.

### MongoTemplate: precise and atomic

```java
// Atomic partial update with a state guard (no read-modify-write)
UpdateResult r = mongo.updateFirst(
    Query.query(Criteria.where("claimNo").is(claimNo).and("status").is("PENDING")),
    new Update().set("status", "PAID").inc("version", 1).currentDate("updatedAt"),
    Claim.class);

// Bulk update in one command
mongo.updateMulti(Query.query(Criteria.where("status").is("PENDING")),
                  new Update().set("notes", "bulk"), Claim.class);

// Find-and-modify (claim the next job atomically)
Job next = mongo.findAndModify(
    Query.query(Criteria.where("status").is("READY")).with(Sort.by("priority").descending()),
    new Update().set("status", "RUNNING").set("owner", workerId),
    FindAndModifyOptions.options().returnNew(true), Job.class);
```

Measured: 500 × (`findById` → modify → `save()`) took **2,337 ms**, and one `updateMulti` touching 6,665 documents took **342 ms**.

### save() semantics and lost data

```mermaid
sequenceDiagram
    participant A as Service A (Spring Data)
    participant DB as MongoDB
    participant B as Service B
    A->>DB: findByClaimNo C6 (doc without extraField)
    B->>DB: $set extraField = "x"
    A->>DB: save(claim) = replaceOne whole document
    Note over DB: extraField is gone
```
*Notice that `save()` writes the whole entity as it exists in memory. Any field another writer added, or that your class doesn't map, is lost. `@Version` turns this silent overwrite into an exception.*

Measured: after another writer set `extraField`, `repo.save(claim)` removed it (present? **false**). The same change done with `updateFirst(... new Update().set("notes", …))` kept it (present? **true**). With `@Version`, two copies loaded at version 0: the first save succeeded (version 1), the second threw **`OptimisticLockingFailureException`**.

Also dangerous: saving an entity that was loaded with a **projection** (`fields = …`). The unprojected fields are null in memory, and `save()` writes those nulls back.

## In practice: code & configuration

### Configuration

```yaml
spring:
  data:
    mongodb:
      uri: mongodb+srv://${MONGO_USER}:${MONGO_PWD}@cluster0.example.mongodb.net/claims?retryWrites=true&w=majority
      auto-index-creation: false       # default; create indexes via migrations instead
      uuid-representation: standard
logging:
  level:
    org.springframework.data.mongodb.core.MongoTemplate: DEBUG   # log generated queries in dev
```

```java
@Configuration
@EnableMongoAuditing
class MongoConfig {
    @Bean MongoTransactionManager transactionManager(MongoDatabaseFactory f) {
        return new MongoTransactionManager(f);      // without this, @Transactional does nothing for Mongo
    }
    @Bean MongoCustomConversions conversions() {
        return MongoCustomConversions.create(c ->
            c.bigDecimal(MongoCustomConversions.BigDecimalRepresentation.DECIMAL128));
    }
}
```

### Indexes: the auto-index trap

=== "❌ Common mistake"

    ```java
    @Document("claims")
    public class Claim {
        @Indexed(unique = true) String claimNo;   // looks enforced...
    }
    // ...but with the default auto-index-creation=false, no index exists.
    // Measured: indexes = [_id_], and inserting a duplicate claimNo succeeded.
    ```

=== "✅ Better"

    ```java
    // Create indexes deliberately at startup (or via Mongock/Liquibase migrations)
    @Component
    class IndexInitializer {
        IndexInitializer(MongoTemplate mongo, MongoMappingContext ctx) {
            IndexResolver resolver = new MongoPersistentEntityIndexResolver(ctx);
            IndexOperations ops = mongo.indexOps(Claim.class);
            resolver.resolveIndexFor(Claim.class).forEach(ops::ensureIndex);   // claimNo unique, member_date
        }
    }
    // Measured with indexes in place: indexes = [_id_, member_date, claimNo] and the duplicate threw DuplicateKeyException.
    ```

For large production collections, prefer reviewed migrations (Mongock, Liquibase MongoDB extension) so index builds are planned. Building a unique index fails if duplicates already exist, which is exactly what happens after months without it.

### Optimistic locking and partial updates

```java
@Service
class ClaimService {
    private final ClaimRepository repo;
    private final MongoTemplate mongo;

    // Whole-aggregate edits: @Version + retry on conflict
    @Retryable(retryFor = OptimisticLockingFailureException.class, maxAttempts = 3)
    public Claim addNote(String claimNo, String note) {
        Claim c = repo.findByClaimNo(claimNo).orElseThrow();
        c.setNotes(note);
        return repo.save(c);              // filter includes version; mismatch → exception
    }

    // Single-field state changes: atomic update, no read
    public boolean markPaid(String claimNo) {
        return mongo.updateFirst(
            Query.query(Criteria.where("claimNo").is(claimNo).and("status").is("PENDING")),
            new Update().set("status", "PAID").currentDate("updatedAt").inc("version", 1),
            Claim.class).getModifiedCount() == 1;
    }
}
```

### Transactions

```java
@Transactional                              // requires the MongoTransactionManager bean
public void voidAndReissue(String claimNo, Claim replacement) {
    mongo.updateFirst(Query.query(Criteria.where("claimNo").is(claimNo)),
                      new Update().set("status", "VOID"), Claim.class);
    repo.insert(replacement);
}
```

Measured with `TransactionTemplate`: inserting `TX1` and updating `C8` to `VOID` and then throwing left the count unchanged and `C8` still `PENDING`. Add retries for `TransientTransactionError` outside the transactional method, and keep side effects out (use an outbox). See [transactions and consistency](06-transactions-and-consistency.md).

### Testing

```java
@SpringBootTest
@Testcontainers
class ClaimRepositoryIT {
    @Container @ServiceConnection
    static MongoDBContainer mongo = new MongoDBContainer("mongo:8.0");   // single-node replica set: transactions work

    @Autowired ClaimRepository repo;
    // assert on generated queries, optimistic locking, index behaviour...
}
```

`@DataMongoTest` gives a sliced context (repositories, template, converters) for faster tests. Point it at Testcontainers, not an embedded fake, so behaviour matches production.

## Real-world usage

- **CRUD-heavy microservices** use repositories for simple reads and `MongoTemplate` for state transitions and bulk updates.
- **GraphQL backends** (Spring for GraphQL) map resolvers to repository projections, and `@BatchMapping` with `findAllById` (`$in`) avoids N+1 lookups.
- **Reactive stacks** use `ReactiveMongoRepository`/`ReactiveMongoTemplate` with WebFlux, the same mapping model on the reactive driver.
- **Change streams** via `MessageListenerContainer` or `ReactiveMongoTemplate.changeStream` feed cache invalidation and outbox relays.
- **Migrations** with Mongock (Java-based changesets) handle index creation and data reshaping across deployments.

## Trade-offs & production gotchas

!!! warning "Spring Data MongoDB pitfalls"
    - **Indexes not created:** `@Indexed`/`@CompoundIndex` are metadata unless index creation runs (duplicate accepted, measured).
    - **`save()` replaces the document:** fields unknown to the class or set by others disappear. Use `@Version` and `Update` for partial changes.
    - **Saving projected entities:** unprojected fields are null and get written back.
    - **BigDecimal as string by default** in 4.x: wrong sorting and comparisons. Use Decimal128.
    - **`Page` count cost:** an extra `countDocuments` per page (2× slower here). Use `Slice` or keyset pagination.
    - **`@DBRef`:** lazy or eager N+1 loading and no `$lookup` support in queries. Prefer manual references (store ids) plus batch loading, or embedding.
    - **`findAll()` on big collections:** loads everything into memory. Use streams (`Stream<T>` repository methods inside try-with-resources) or paging.
    - **`@Transactional` without `MongoTransactionManager`:** silently non-transactional.
    - **Derived query explosion:** 80-character method names are unreadable. Switch to `@Query` or `Criteria`.
    - **`_class` coupling:** renaming packages breaks polymorphic reads. Use `@TypeAlias`.

- **Repositories vs template:** repositories are concise and testable for reads. The template is explicit and supports atomic operators and bulk writes. Most real services use both.
- **Blocking vs reactive:** reactive repositories pay off only with an end-to-end non-blocking stack. With virtual threads (Java 21), the blocking driver is often simpler.

## How this connects to my experience

- **Resume bullet (OptumRx):** "Designed and developed microservices using Java, Spring Boot, Kafka, MongoDB, Redis, and GraphQL."
- **How to talk about it:** Spring Data MongoDB repositories for straightforward reads and projections behind GraphQL resolvers, `MongoTemplate` for state changes and aggregations, `@Version` or guarded updates for concurrency, and indexes managed deliberately rather than by annotations alone. *[confirm: repositories vs template mix, whether you used @Version, transactions, Mongock/migrations, Decimal128 for amounts, Testcontainers]*
- **Talking points:**
    - "For state transitions I use `MongoTemplate.updateFirst` with the expected state in the filter, which is atomic and doesn't overwrite fields others changed."
    - "I don't trust `@Indexed` to create indexes: auto-index-creation is off by default, so indexes come from migrations or explicit initialisation."
    - "For lists I use `Slice` or keyset pagination, because `Page` runs a count query on every request."
- **Likely follow-up chain:** "Repository or template?" → "How does a derived query map to MongoDB?" → "How do you handle concurrent updates?" (`@Version`, guarded updates) → "How are indexes created?" → "Transactions in Spring with MongoDB?" → "How do you test it?" (Testcontainers).

## Interview questions

### Fundamentals

??? question "Q1. MongoRepository vs MongoTemplate: when do you use each?"
    **Answer:** Repositories give CRUD, derived queries, `@Query`, `@Aggregation`, projections and paging with almost no code, which is good for straightforward reads and simple saves. `MongoTemplate` gives full control: `Query`/`Criteria`, atomic `Update` operators (`$set`, `$inc`, `$push`), `findAndModify`, `updateMulti`, bulk operations, complex aggregations and change streams. Use the template for state transitions, partial updates and bulk work (measured: one `updateMulti` over 6,665 docs in 342 ms vs 500 find-and-save calls in 2,337 ms). Most services use both, often via a custom repository fragment that uses the template.

    **Interviewer listens for:** the strengths of each, atomic updates via the template, and the combination.

    **Common wrong answer:** "Repositories for everything. The template is legacy."

??? question "Q2. What does @Indexed(unique = true) do in Spring Boot 3?"
    **Answer:** By itself, nothing at runtime. Since Spring Data MongoDB 3.0, `spring.data.mongodb.auto-index-creation` defaults to `false`, so index annotations are just metadata. Measured: only `_id_` existed, and a duplicate `claimNo` was inserted without error. With index creation enabled (or explicit `IndexOperations.ensureIndex` from resolved metadata), the unique index existed and the duplicate threw `DuplicateKeyException`. In production, create indexes via migrations or explicit startup code, planned like schema changes.

    **Interviewer listens for:** the default, the consequence, and how to create indexes properly.

    **Common wrong answer:** "It creates a unique index when the app starts."

??? question "Q3. What does repository.save() do for an existing document?"
    **Answer:** It replaces the entire document with the entity's current state (a `replaceOne` by `_id`, plus the version check if `@Version` exists). Fields not in the Java object, added by another service or a newer version of the app, are removed, and concurrent changes are overwritten. Measured: `extraField` set by another writer vanished after `save()`, but survived an `updateFirst` with `$set`. Use `@Version` for whole-entity edits and `Update` operations for partial changes.

    **Interviewer listens for:** whole-document replacement, the lost-field and lost-update risks, and the alternatives.

    **Common wrong answer:** "save() only updates changed fields, like JPA dirty checking."

??? question "Q4. How does optimistic locking work in Spring Data MongoDB?"
    **Answer:** Annotate a numeric field with `@Version`. On insert it starts at 0. On `save()` of an existing entity, Spring adds the current version to the update filter and increments it. If another writer changed the document first, the filter doesn't match, and Spring throws `OptimisticLockingFailureException` (measured: the second stale save failed, and the version went 0 → 1). Handle it by reloading and retrying, or by returning a conflict to the client. For single-field transitions, a guarded `updateFirst` with the expected state is simpler.

    **Interviewer listens for:** the version in the filter, the exception, and retry handling.

    **Common wrong answer:** "MongoDB locks the document between find and save."

### Intermediate

??? question "Q5. Page or Slice for pagination: what's the difference?"
    **Answer:** `Page` runs the find plus a `countDocuments` for the total, on every request, so it can show "page 3 of 334". `Slice` fetches `size + 1` documents to know whether there's a next page, with no count. Measured: 50 pages took 1,534 ms with `Page` vs 773 ms with `Slice`. For large collections and deep pages, both suffer from `skip`, so use keyset or scroll pagination (`Window`/`ScrollPosition` with `KeysetScrollPosition` in Spring Data 3.1+) on an indexed sort key.

    **Interviewer listens for:** the count query cost, the size+1 trick, and keyset pagination.

    **Common wrong answer:** "They're the same. Slice is just older."

??? question "Q6. How do you map BigDecimal, and why does it matter?"
    **Answer:** Spring Data MongoDB 4.x converts `BigDecimal` to a **string** by default, so amounts sort and compare lexically ("100" < "20"), and range queries and `$sum` aggregations misbehave. Store money as Decimal128: per field with `@Field(targetType = FieldType.DECIMAL128)`, or globally with `MongoCustomConversions.create(c -> c.bigDecimal(BigDecimalRepresentation.DECIMAL128))`. Verified: the stored amount was `{"$numberDecimal": "393.80"}`, type Decimal128, and the `@Aggregation` sum worked. Changing the representation on existing data needs a migration.

    **Interviewer listens for:** the default, its consequences, and both configuration options.

    **Common wrong answer:** "Use double; it's fine for money."

??? question "Q7. What is the _class field, and when is it a problem?"
    **Answer:** `MappingMongoConverter` writes the fully qualified class name in `_class` (measured `"kb.App$Claim"`) so it can instantiate the right subtype for polymorphic collections. Problems: refactoring package or class names breaks reading old documents, other languages' services see a Java-specific field, and it adds bytes per document. Use `@TypeAlias("claim")` for stable short names, or disable type information if you don't use polymorphism.

    **Interviewer listens for:** its purpose, the refactoring risk, and TypeAlias.

    **Common wrong answer:** "It's an internal MongoDB field."

??? question "Q8. How do you make @Transactional work with MongoDB in Spring?"
    **Answer:** Register a `MongoTransactionManager` bean (Spring Boot doesn't auto-configure one for MongoDB) and run against a replica set or sharded cluster. Then `@Transactional` methods bind a `ClientSession` and the template and repositories participate. Verified: an insert and an update followed by an exception were both rolled back. Add retry logic for `TransientTransactionError` around the transactional method, set the transaction's write and read concerns (majority and snapshot), and keep external side effects out.

    **Interviewer listens for:** the explicit bean, the replica set requirement, retries, and side effects.

    **Common wrong answer:** "Just add @Transactional; Boot configures it."

??? question "Q9. What's the risk of saving an entity loaded with a projection?"
    **Answer:** Fields excluded by the projection are null in the loaded object (measured: `memberId` was null for a `@Query(fields = …)` result). Passing that object to `save()` replaces the document with those nulls, wiping data. Use interface or DTO projections (read-only types that can't be saved) for partial reads, and do partial writes with `Update`. Code review should flag `save()` on objects that came from projected queries.

    **Interviewer listens for:** nulls from projection, data loss through `save()`, and the safe patterns.

    **Common wrong answer:** "Spring only saves non-null fields."

### Senior

??? question "Q10. How would you structure the data access layer of a MongoDB-backed Spring service?"
    **Answer:** A repository interface per aggregate for simple reads (derived queries, projections, `Slice`). A custom fragment (`ClaimRepositoryCustom` + `Impl` using `MongoTemplate`) for atomic state transitions, bulk updates and complex aggregations. Entities designed around the document model, with Decimal128 for money, `@Version` where whole-aggregate edits happen, and auditing. Indexes managed by migrations (Mongock) and checked in integration tests. `MongoTransactionManager` only where invariants span documents. An outbox for events. Testcontainers-based integration tests asserting behaviour and query plans for critical queries. Logging of slow queries via the driver's command listener and Micrometer metrics.

    **Interviewer listens for:** a clear split between repository and template, a migrations strategy, and testing and observability.

    **Common wrong answer:** "Generate everything from repository method names."

??? question "Q11. Why avoid @DBRef, and what do you use instead?"
    **Answer:** `@DBRef` stores `{$ref, $id}` and resolves it with a separate query per reference (eagerly, or lazily via proxies), so lists of entities cause N+1 queries. It can't be used inside queries or `$lookup` efficiently, and it couples documents tightly. Instead: embed when data is owned and bounded, store plain ids (manual references) and batch-load with `findAllById` (`$in`), use `@DocumentReference` (Spring Data 3.3+, which supports custom lookups and `$lookup`-friendly ids), or copy display fields (extended reference). In GraphQL, batch with `@BatchMapping` or DataLoader.

    **Interviewer listens for:** the N+1 issue, the alternatives, and batching.

    **Common wrong answer:** "@DBRef is like a JPA @ManyToOne join."

??? question "Q12. How do you find and fix slow queries generated by Spring Data repositories?"
    **Answer:** Log generated queries in development (the `MongoTemplate` DEBUG logger), and in production use the MongoDB profiler or Atlas Query Profiler, plus a driver `CommandListener` with Micrometer (`MongoMetricsCommandListener`) for latency per command. Reproduce a slow query with `explain("executionStats")`: look for COLLSCAN, blocking SORT, or a high docsExamined/nReturned ratio. Fix with a compound index (ESR), projections or a `Slice` instead of `Page`, rewrite derived queries as `@Query` or `Criteria` with `$elemMatch` where needed, and add a test that asserts the index is used for critical queries. See [indexing](03-indexing-and-explain-plans.md).

    **Interviewer listens for:** observability, explain, concrete fixes, and regression protection.

    **Common wrong answer:** "Add @Indexed to the fields." That's not even created by default.

### Scenario-based

??? question "Q13. Duplicate claim numbers appeared in production even though the entity has @Indexed(unique = true). Explain and remediate."
    **Answer:** With the default `auto-index-creation=false`, the unique index was never created (I reproduced it: only `_id_` existed and a duplicate insert succeeded). Remediation: find the duplicates with an aggregation (`$group` by `claimNo`, `count > 1`), resolve them with the business (merge, renumber, void), then create the unique index (it fails while duplicates remain) through a migration, and add the index check to integration tests and deployment. Also make claim creation idempotent (an upsert by `claimNo`) so retries don't create duplicates.

    **Interviewer listens for:** the root cause, cleanup before indexing, migrations, and idempotency.

    **Common wrong answer:** "MongoDB doesn't enforce unique indexes."

??? question "Q14. After deploying a new version of service B that adds a field to claims, service A's updates keep erasing it. Why, and how do you fix it?"
    **Answer:** Service A loads claims into its own class, which doesn't know the new field, and calls `save()`, which replaces the whole document and drops unknown fields (measured: `extraField` vanished after `save()`). Fix: change A to use targeted `Update` operations for its changes (`$set` only the fields it owns, which kept `extraField` in my test), or add the field to A's model, or keep the raw `Document` for passthrough. Add `@Version` so concurrent whole-document saves fail instead of overwriting. Longer term, define field ownership per service, or give each service its own collection.

    **Interviewer listens for:** whole-document replacement, partial updates, versioning, and ownership.

    **Common wrong answer:** "Service B should write the field again after every A update."

## Cheat sheet

| Topic | Remember |
|---|---|
| APIs | Repositories (derived, `@Query`, `@Aggregation`, projections) + `MongoTemplate` (Criteria, Update, bulk) |
| Indexes | `auto-index-creation` false by default → `@Indexed` not created (duplicate accepted) |
| save() | Replaces the whole doc; drops unknown fields; use `Update` for partial changes |
| @Version | Stale save → `OptimisticLockingFailureException` |
| Projections | Interface/DTO for reads; never `save()` a projected entity (nulls) |
| BigDecimal | String by default in 4.x → Decimal128 via `@Field` or `BigDecimalRepresentation` |
| _class | Type hint; `@TypeAlias` for stability |
| Paging | `Page` = + count (1,534 vs 773 ms); `Slice` / keyset (`Window`) |
| Bulk | `updateMulti` 342 ms vs 500 × save 2,337 ms |
| Transactions | `MongoTransactionManager` bean + replica set + retries |
| @DBRef | N+1 → manual refs + `findAllById`, `@DocumentReference`, embedding |
| Testing | Testcontainers `MongoDBContainer` + `@ServiceConnection` |

## Sources
1. [Spring Data MongoDB reference](https://docs.spring.io/spring-data/mongodb/reference/) (mapping, repositories, template, aggregation, transactions).
2. [Spring Data MongoDB: Index creation (auto-index-creation off by default since 3.0)](https://docs.spring.io/spring-data/mongodb/reference/mongodb/mapping/mapping.html#mapping.index-creation).
3. [Spring Data MongoDB: Custom conversions (BigDecimal representation)](https://docs.spring.io/spring-data/mongodb/reference/mongodb/mapping/custom-conversions.html) and [type mapping (_class, @TypeAlias)](https://docs.spring.io/spring-data/mongodb/reference/mongodb/converters-type-mapping.html).
4. [Spring Data MongoDB: Optimistic locking](https://docs.spring.io/spring-data/mongodb/reference/mongodb/template-crud-operations.html#mongo-template.optimistic-locking) and [document references](https://docs.spring.io/spring-data/mongodb/reference/mongodb/mapping/document-references.html).
5. [Spring Data Commons: Scrolling (Window, KeysetScrollPosition)](https://docs.spring.io/spring-data/commons/reference/repositories/scrolling.html).
6. [Spring Boot reference: MongoDB and Testcontainers service connections](https://docs.spring.io/spring-boot/reference/data/nosql.html#data.nosql.mongodb).
7. Demonstrations on this page: Spring Boot 3.5.6 + Spring Data MongoDB 4.5.4 integration tests against MongoDB 8.0.4, run while writing this page (mapping, indexes, derived/@Query/@Aggregation, Page vs Slice, auditing, @Version, save() field loss, duplicate key, transaction rollback, bulk update).
