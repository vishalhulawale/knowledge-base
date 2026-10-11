---
title: Spring AI & LLM API integration
description: How a Java service talks to an LLM, raw HTTP API vs vendor SDK vs Spring AI 2.0, the ChatClient and advisor chain, tool calling, structured output, chat memory, and the production work around it (timeouts, retries, rate limits, cost, observability and secrets).
tags: [genai, P1]
---

# Spring AI & LLM API integration

!!! abstract "Key takeaways"
    - An LLM API is a **stateless, slow, metered, rate-limited HTTP dependency**: each request carries the whole conversation, latency is seconds, you pay per token, and 429/529 errors are normal. Give it timeouts, one retry layer, a concurrency cap, a circuit breaker and a fallback.
    - **Spring AI 2.0** (GA June 2026, current 2.0.1; Spring Boot 4.x, Spring Framework 7, Jackson 3) gives you one portable API over providers. **`ChatClient`** is the user-facing fluent API; **`ChatModel`** is the low-level, per-provider building block. Core providers now wrap the **vendor SDKs** (Anthropic Java SDK, OpenAI Java SDK, Google GenAI SDK).
    - Cross-cutting behaviour runs in an ordered **advisor chain** (memory, RAG, logging, guardrails); in 2.0 the **`ToolCallingAdvisor`** owns the tool-call loop and `ChatModel.call()` no longer executes tools.
    - **Structured output**: `.entity(MyRecord.class)` maps the reply to a Java type. `useProviderStructuredOutput()` sends the JSON Schema to the provider as a hard constraint; `validateSchema()` adds a validate-and-retry loop.
    - Options are model-specific: Claude Opus 5.5 **rejects** `temperature`/`top_p`/`top_k`, thinking budgets and forced `tool_choice` with a 400, so tutorial config breaks. Pin options per model and test against the real API.

## Why it matters

Most enterprise GenAI work is **integrating** a hosted model into an existing service: a claims assistant behind an API, a classifier on a Kafka consumer, a batch summariser. Senior Java interviews test whether you do it like integrating a payment gateway, with typed contracts, failure handling, cost control, security and observability, not a `RestTemplate` call and string concatenation.

Calling a provider's REST API or SDK directly works for one feature. By the fourth, every team has re-implemented prompt templates, the tool-calling loop, JSON parsing, chat history and metrics, and changing provider means a rewrite. Spring AI (and LangChain4j, the main Java alternative) applies the Spring Data idea: a portable API plus auto-configuration.

## Core concepts

### The LLM API as a dependency

Whatever the provider, a chat API call has the same shape. Using Anthropic's Messages API (`POST /v1/messages`) as the example:

- **Request**: `model`, `max_tokens` (a hard cap on output), a `system` prompt, the full `messages` history (alternating `user`/`assistant`), optional `tools` (JSON Schema per tool) and output settings (structured output schema, effort).
- **Response**: content blocks (`text`, `tool_use`, `thinking`), a `stop_reason` (`end_turn`, `max_tokens`, `tool_use`, `refusal`, …) and `usage` (input, output and cache read/write tokens), which is your bill.
- **Statelessness**: the API remembers nothing. "Chat memory" is your code resending history, so every turn costs more input tokens than the last (see [LLM fundamentals](01-llm-fundamentals-tokens-context-window-temperature-hallucina.md) for context windows).
- **Errors**: 400 (including unsupported parameters), 413 (too large), **429** rate limit with `retry-after`, **529** overloaded and other 5xx. The official SDKs retry connection errors, 408, 409, 429 and 5xx with backoff (2 retries by default).

Three ways to call it from Java:

| Approach | Pros | Cons | Use when |
|---|---|---|---|
| Raw HTTP (`RestClient`/`WebClient`) | No dependency, full control | You own JSON, streaming (SSE), retries, tool loop | Proxies, gateways, unusual runtimes |
| Vendor SDK (e.g. `com.anthropic:anthropic-java`) | Typed, day-one access to new features, built-in retries | Locked to one provider; you still build memory, RAG, tool loop | One provider, need the newest API features |
| Spring AI 2.0 | Portable API, Boot auto-config, advisors, tools, memory, vector stores, Micrometer | Abstraction lags new provider features; options differ per model anyway | Spring services, multiple features or providers |
| LangChain4j | Similar scope, framework-agnostic, `@AiService` interfaces | Different idioms from Spring; a second ecosystem | Quarkus/plain Java, or team preference |

### Spring AI 2.0 building blocks

```mermaid
flowchart LR
    C["Controller or service"] --> CC["ChatClient (fluent API)"]
    CC --> A1["Advisors in order: memory +200, RAG, ToolCallingAdvisor +300, logging"]
    A1 --> CM["ChatModel (AnthropicChatModel, OpenAiChatModel ...)"]
    CM --> SDK["Vendor Java SDK (HTTP, retries, SSE)"]
    SDK --> API["Provider API"]
    A1 -.-> T["@Tool methods in your app"]
    A1 -.-> VS["VectorStore (pgvector, Redis, ...)"]
    A1 -.-> MEM["ChatMemoryRepository (JDBC, Redis, ...)"]
```
*Notice that your code talks only to `ChatClient`; memory, retrieval and the tool loop are advisors around a provider-specific `ChatModel`, which itself delegates to the vendor SDK.*

- **`ChatModel`**: one interface per provider with `call(Prompt)` and `stream(Prompt)` (a Reactor `Flux`). In 2.0 it returns the raw model response; it no longer runs tools.
- **`ChatClient`**: the fluent API (`prompt().system(...).user(...).call().content()`), built from an auto-configured, prototype-scoped `ChatClient.Builder`. Defaults (system prompt, options, advisors, tools) are set once on the builder.
- **Options**: immutable, builder-based in 2.0 (`AnthropicChatOptions.builder()`), with defaults in the options class (Anthropic: model `claude-haiku-4-5`, `maxTokens` 4096, 60 s timeout, 2 retries). The `.options` segment disappeared from property keys (`spring.ai.anthropic.chat.model`).
- **Advisors**: interceptors (`CallAdvisor`, `StreamAdvisor`) ordered by `getOrder()`; lower order runs first on the way in, last on the way out. Built-ins include `MessageChatMemoryAdvisor`, `QuestionAnswerAdvisor` and `RetrievalAugmentationAdvisor` (RAG, see [RAG architecture](04-rag-architecture.md)), `SimpleLoggerAdvisor` and `SafeGuardAdvisor`.
- **Starters**: `spring-ai-starter-model-anthropic`, `-openai`, and so on, plus the Spring AI BOM. MCP clients and servers (`@McpTool`) are part of core in 2.0.

### The tool-calling loop

The model cannot run code. It asks you to: the response stops with `tool_use` and the arguments, your application executes the method, and you send the result back in a second request. Spring AI 2.0 moved that loop out of each `ChatModel` into the **`ToolCallingAdvisor`**, auto-registered by `ChatClient` whenever tools are present.

```mermaid
sequenceDiagram
    participant App as ChatClient
    participant TCA as ToolCallingAdvisor
    participant M as ChatModel
    participant API as Claude API
    App->>TCA: prompt + tool definitions
    TCA->>M: call 1
    M->>API: messages + tools
    API-->>M: stop_reason tool_use, getClaimStatus C-104
    M-->>TCA: hasToolCalls() = true
    Note over TCA: ToolCallingManager runs the @Tool method in your JVM
    TCA->>M: call 2 with tool_result appended
    M->>API: messages + tool_use + tool_result
    API-->>M: stop_reason end_turn, final text
    M-->>TCA: no tool calls
    TCA-->>App: final ChatResponse
```
*Notice that one "chat" call is two billed API calls, and the tool runs in your service with your credentials, so the model only ever sees its result.*

![Animation: a controller sends a prompt with a conversation id through the memory advisor; call 1 goes to the Claude API, which answers with stop reason tool_use; the ToolCallingAdvisor runs the getClaimStatus tool method; call 2 sends the tool result; the API answers end_turn with the final text, which returns to the controller; the memory advisor then saves only the user message and the final answer](images/05-advisor-chain-tool-loop.svg){ loading=lazy }
*Watch where memory sits: outside the tool loop by default (order +200 vs +300), so the stored history holds whole turns, never a dangling tool call.*

Production guard rails: `ToolCallingManager` caps calls per tool and per turn (`maxCallsPerTool`, `maxTotalToolCalls`), and `ToolContext` passes tenant or user ids to tools **without** sending them to the model. Tools are a privilege boundary: injected text can ask the model to call them, so authorise inside the tool (see [Agents & tool calling](06-agents-and-tool-calling.md)).

### Structured output

`.entity(ClaimSummary.class)` derives a JSON Schema from the record. By default Spring AI adds format instructions to the prompt and parses the reply, with no guarantee. Two switches on `EntityParamSpec` harden it: `useProviderStructuredOutput()` sends the schema to the provider as an API constraint (Anthropic requires `additionalProperties: false` and a recent model), and `validateSchema()` validates the reply and retries with the error appended (3 attempts by default). Use both when downstream code can't tolerate shape drift. The prompt side is covered in [Prompt engineering & structured output](02-prompt-engineering-and-structured-output.md).

### Chat memory

`MessageChatMemoryAdvisor` loads and saves history per `ChatMemory.CONVERSATION_ID` through a `ChatMemory` (default `MessageWindowChatMemory`: a 20-message window that evicts whole turns) and a `ChatMemoryRepository` (in-memory, JDBC, Redis, …). Behind a load balancer use a shared repository. The window is also a cost control: every remembered message is re-billed on each turn.

### Streaming

`.stream().content()` returns a `Flux<String>` of tokens (server-sent events from the provider), so a UI shows text within a second or so. It does not shorten total time, so it needs the same timeout budget, and `.entity()` is call-only.

## In practice: code & configuration

Spring AI 2.0.1, Spring Boot 4.x, Java 21. Configuration first:

```yaml
spring:
  ai:
    anthropic:
      api-key: ${ANTHROPIC_API_KEY}        # from Secrets Manager / Key Vault, never in git
      timeout: 20s                         # construction-time default (library default 60s)
      max-retries: 1                       # SDK retries 429, 5xx, timeouts with backoff
      chat:
        model: claude-opus-5-5             # pin explicitly; the library default is claude-haiku-4-5
        max-tokens: 4096
        cache-options:
          strategy: SYSTEM_AND_TOOLS       # prompt caching for the stable prefix
    chat:
      client:
        observations:
          log-prompt: false                # prompts may contain PHI/PII; keep off in prod
          log-completion: false
```

=== "❌ Common mistake"
    ```java
    // Copied from a 1.x tutorial: new model per call, sampling knobs, forced tool choice,
    // user input concatenated into the system prompt, raw string parsing.
    @GetMapping("/claims/{id}/summary")
    String summary(@PathVariable String id, @RequestParam String question) {
        var model = AnthropicChatModel.builder()                 // new SDK client + pool per request
            .options(AnthropicChatOptions.builder()
                .model("claude-opus-5-5")
                .temperature(0.2)                                // 400 on current Claude models
                .thinkingEnabled(2048L)                          // fixed budget: 400 on Opus 5.5
                .toolChoice(ToolChoiceAny.builder().build())     // forced tool use: 400 on Opus 5.5
                .build())
            .build();
        String system = "You summarise claim " + id + ". " + question;   // prompt injection
        ChatResponse r = model.call(new Prompt(system));         // 2.0: tools would NOT be executed
        return r.getResult().getOutput().getText();              // free text, parsed by regex later
    }
    ```

=== "✅ Correct approach"
    ```java
    public record ClaimSummary(String claimId, String status, List<String> nextSteps) {
        static ClaimSummary unavailable() {
            return new ClaimSummary(null, "UNAVAILABLE", List.of("Try again later"));
        }
    }

    @Configuration
    @EnableResilientMethods                                      // Spring Framework 7: @ConcurrencyLimit
    class AiConfig {
        @Bean
        ChatClient claimsClient(ChatClient.Builder builder, ChatMemory memory, ClaimTools tools) {
            return builder
                .defaultSystem("""
                    You are a claims assistant. Answer only from tool results.
                    If the data is missing, say so.""")                  // fixed, cacheable prefix
                .defaultOptions(AnthropicChatOptions.builder()            // 2.0: pass the builder
                    .effort(OutputConfig.Effort.MEDIUM)                   // cost/quality knob, no temperature
                    .timeout(Duration.ofSeconds(20)))
                .defaultTools(tools)                                      // @Tool methods
                .defaultAdvisors(MessageChatMemoryAdvisor.builder(memory).build())
                .build();                                                 // ToolCallingAdvisor auto-registered
        }
    }

    @Service
    class ClaimAssistant {
        private static final Logger log = LoggerFactory.getLogger(ClaimAssistant.class);
        private final ChatClient chat;
        ClaimAssistant(ChatClient claimsClient) { this.chat = claimsClient; }

        @ConcurrencyLimit(20)                                    // bulkhead: cap in-flight LLM calls
        ClaimSummary summarise(String conversationId, String userId, String question) {
            try {
                return chat.prompt()
                    .user(question)                              // user text stays in the user turn
                    .toolContext(Map.of("userId", userId))       // for authorisation in tools, never sent to the model
                    .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, conversationId))
                    .call()
                    .entity(ClaimSummary.class, spec -> spec
                        .useProviderStructuredOutput()           // schema enforced by the API
                        .validateSchema());                      // validate + retry on drift
            } catch (RuntimeException e) {                       // SDK already retried 429/5xx once
                log.warn("LLM unavailable, degrading", e);
                return ClaimSummary.unavailable();               // deterministic fallback
            }
        }
    }

    @Component
    class ClaimTools {
        private final ClaimRepository claims;
        ClaimTools(ClaimRepository claims) { this.claims = claims; }

        @Tool(description = "Get the status of a claim the current user may see")
        ClaimStatus getClaimStatus(@ToolParam(description = "Claim id, e.g. C-104") String claimId,
                                   ToolContext ctx) {
            String userId = (String) ctx.getContext().get("userId");
            return claims.findVisibleTo(claimId, userId)         // authorise here, not in the prompt
                .orElseThrow(() -> new IllegalArgumentException("not found or not permitted"));
                // RuntimeException message goes back to the model as a tool error (default)
        }
    }
    ```

The fix: one shared client, no parameters the model rejects, `effort` for cost/quality, user text kept in the user turn, tools that authorise server-side, a schema-enforced result, and a concurrency cap, short timeout and fallback. Add a Resilience4j circuit breaker around `summarise` (see [Resilience patterns](../microservices/06-resilience-circuit-breaker-retry-bulkhead-timeout-rate-limit.md)).

![Three timelines for a slow provider: with defaults, three 60 second attempts hold the thread for 180 seconds; with a 15 second timeout and one retry, the fallback answers at about 32 seconds; with the breaker open the fallback answers immediately](images/05-timeout-retry-budget.svg){ loading=lazy }
*The defaults are reasonable for a batch job and terrible for a request thread: decide the budget per use case.*

## Real-world usage

- **Patterns that ship**: drafting and summarising behind human review, classification on a Kafka consumer (latency hidden, retry topics and a DLQ), RAG assistants over internal docs, and read-only tool lookups first.
- **Gateways and cloud routes**: larger organisations put an **LLM gateway** in front of providers for keys, per-team quotas, redaction and routing (Spring AI points at it via `base-url`); regulated teams often reach Claude through Amazon Bedrock or Vertex AI to stay in their cloud account and region.
- **Failure modes**: 529 overloads during launches, rate-limit storms from unbounded batch jobs, runaway tool loops, `max_tokens` truncation producing invalid JSON, and PHI in debug logs.
- **Healthcare and banking**: send the minimum data, redact before the call, check the provider agreement (BAA for PHI, data retention, zero-retention options), and keep an audit trail of prompts and outputs with access controls (see [Guardrails](../fde-applied-llm/06-guardrails-prompt-injection-pii-phi-redaction-grounding-chec.md)).

## Trade-offs & production gotchas

| Option | Pros | Cons | Use when |
|---|---|---|---|
| Blocking `call()` | Simple, works with `.entity()` | Holds a thread for seconds | Batch, back-office, virtual threads |
| Streaming `stream()` | Fast first token, better UX | Same total time; harder JSON and accounting | Chat UIs |
| Native structured output | Provider guarantees shape | Newer models only; schema rules | Machine-consumed results |
| Prompt-only structured output + `validateSchema()` | Portable | Extra calls on failure | Older or mixed providers |

!!! warning "Gotcha: 1.x and blog config on new models"
    `temperature`, `topP`, `topK`, `thinkingEnabled(budget)`, `thinkingDisabled()` and `ToolChoiceAny`/`ToolChoiceTool` are all rejected with HTTP 400 by Claude Opus 5.5; it uses adaptive thinking with `effort` (default `medium`) instead. Spring AI passes options through, so the failure appears only at runtime. Keep an integration test per configured model.

!!! warning "Gotcha: retries multiply"
    The SDK retries 429, 5xx and timeouts (2 by default). Wrapping the call in `@Retryable` or a Resilience4j retry multiplies attempts (3 × 3 = 9 calls), amplifies load on an already overloaded provider, and stretches latency to timeout × attempts. Retry in **one** layer and honour `retry-after`.

!!! warning "Gotcha: one chat call is not one API call"
    Tool loops, `validateSchema()` retries and RAG query rewriting each add model calls. Budget per **completed request**: read `usage` from `ChatResponse` metadata (`responseEntity(...)` gives you both), export `gen_ai.client.token.usage` via Micrometer, and alert on cost per feature.

!!! tip "Cost levers in order"
    Prompt caching for the stable prefix (`cache-options.strategy`), a bounded memory window, the right model per route (a small model such as Claude Haiku 5.5 for classification), lower `effort` where quality holds, `max-tokens` sized to the output, and the Batches API for offline work. See [LLM observability & cost control](../fde-applied-llm/07-llm-observability-latency-budgets-and-token-cost-control.md).

## How this connects to my experience

- **Where I used it:** not a resume claim; position as transferable integration knowledge. Whether I have built anything on Spring AI or an LLM API, at work or in a side project: *[confirm]*. The closest real fits are **owning the GraphQL Consumer Service** on OptumRx Meteor, an integration layer over **5 upstream systems** (an LLM provider is one more slow, rate-limited upstream with a contract), **Redis caching** of frequent queries, **Kafka workflows with retry and DLQ**, and **secure APIs with OAuth2/PingFederate**. At Deloitte I **integrated AWS Personalize** (a managed ML service) and used **IAM, KMS and Secrets Manager**, the same controls an LLM API key and PHI-bearing prompts need.
- **Talking points:**
    - Treat the model like any upstream in the GraphQL layer: timeout budget, one retry layer, bulkhead, breaker, fallback, per-call metrics.
    - Healthcare data: minimum necessary data, redaction, no prompt logging in prod, BAA and retention checks.
    - Async where latency allows: LLM enrichment on a Kafka consumer with retry topics and a DLQ, as in the Meteor workflows.
- **Likely follow-up chain:** "How would you add an AI summary to your claims API?" → `ChatClient` with a fixed system prompt, a typed `ClaimSummary`, tools that call existing services with the user's permissions → "What if the provider is down or slow?" → 15–20 s budget, one retry layer, `@ConcurrencyLimit`/bulkhead, breaker, deterministic fallback, async for non-interactive paths → "How do you control cost?" → token metrics per feature, prompt caching, memory window, model routing, `effort`.

## Interview questions

### Fundamentals

??? question "Q1. What actually happens when a Java service calls an LLM chat API?"
    **Answer:** An HTTPS POST with the model id, `max_tokens`, a system prompt, the full message history and optional tool schemas. The provider returns content blocks, a stop reason (`end_turn`, `max_tokens`, `tool_use`, `refusal`) and token usage. It is stateless: conversation memory is the client resending history, so each turn costs more input tokens. Latency is seconds, and 429/529/5xx are expected, so it needs timeouts, retries with backoff and limits.

    **Interviewer listens for:** statelessness, stop reasons, token billing, failure modes.

    **Common wrong answer:** "The model remembers the session on its side."

??? question "Q2. What is the difference between `ChatModel` and `ChatClient` in Spring AI?"
    **Answer:** `ChatModel` is the provider-specific, low-level interface (`call`, `stream` on a `Prompt`). `ChatClient` is the fluent, portable API built on it, with defaults (system prompt, options, tools, advisors), structured output via `.entity()`, and the advisor chain. In 2.0 `ChatModel` no longer runs tools; the auto-registered `ToolCallingAdvisor` in `ChatClient` does.

    **Interviewer listens for:** layering, advisors, the 2.0 tool-loop change.

    **Common wrong answer:** "They are the same; `ChatClient` is just a newer name."

??? question "Q3. How does tool (function) calling work end to end?"
    **Answer:** You send tool names, descriptions and JSON Schemas. The model replies with `stop_reason: tool_use` and arguments; your code runs the method and sends a `tool_result` in a new request with the history; the model then answers or calls more tools. In Spring AI, `@Tool` methods become `ToolCallback`s and `ToolCallingAdvisor` drives the loop, with per-tool and total call limits.

    **Interviewer listens for:** the model never executes code, two or more API calls, loop limits.

    **Common wrong answer:** "The model calls my REST endpoint directly."

### Intermediate

??? question "Q4. How do you get reliable JSON out of an LLM in Spring AI?"
    **Answer:** Map to a record with `.entity(Type.class)`. Prefer the provider's native structured output (`useProviderStructuredOutput()`), which constrains generation to the schema, and add `validateSchema()` to validate and retry on residual failures. Keep `max_tokens` high enough to avoid truncated JSON, and still validate business rules (enums, ranges) in code.

    **Interviewer listens for:** native constraint vs prompt instructions, validation loop, truncation.

    **Common wrong answer:** "Ask nicely in the prompt and parse with a regex."

??? question "Q5. Your LLM calls see frequent 429s. What do you do?"
    **Answer:** Read the rate-limit headers (`getRateLimit()` in Spring AI), honour `retry-after`, retry in one layer, cap concurrency, queue batch work, cut tokens (caching, shorter history), route simple tasks to a smaller model, and separate interactive and batch quotas so batch can't starve users.

    **Interviewer listens for:** backpressure, not more retries; token as well as request limits.

    **Common wrong answer:** "Increase retries to 10."

### Senior

??? question "Q6. Spring AI, the vendor SDK, or LangChain4j: how do you choose?"
    **Answer:** Spring AI in Boot services needing several building blocks (tools, memory, RAG, vector stores, observability) or several providers. The vendor SDK for one provider and day-one features. LangChain4j outside Spring. Portability is partial: options, behaviour and prompts differ per model, so plan evals per model, not a config switch.

    **Interviewer listens for:** trade-offs, feature lag, honest view of portability.

    **Common wrong answer:** "Spring AI lets us swap providers with one property."

??? question "Q7. How do you make an LLM-backed endpoint production-ready in a regulated domain?"
    **Answer:** Vaulted, rotated keys; minimum necessary data, PHI redaction, BAA and retention checks; no prompt logging in prod but an access-controlled audit trail; timeout, one retry layer, bulkhead, breaker, fallback; tools that authorise server-side; validated structured output; token and cost metrics per feature; evals before model or prompt changes; human review for consequential decisions.

    **Interviewer listens for:** security, resilience, observability and evaluation together.

    **Common wrong answer:** "The provider is HIPAA compliant, so we're covered."

### Scenario-based

??? question "Q8. After upgrading to a new Claude model, every request fails with HTTP 400. What happened?"
    **Answer:** Most likely options the new model rejects: `temperature`/`top_p`/`top_k`, a fixed thinking budget or disabled thinking, or forced `tool_choice` (all 400 on Claude Opus 5.5), often set in a shared `defaultOptions` or properties file. Read the error body, remove them, use `effort` for the cost/quality trade-off, and add an integration test per configured model so the next upgrade fails in CI, not prod.

    **Interviewer listens for:** reading the 400 body, model-specific options, test per model.

    **Common wrong answer:** "Roll back the Spring AI version."

??? question "Q9. An AI summary endpoint has p99 latency of 90 s and occasionally returns truncated JSON. Fix it."
    **Answer:** Count the calls: stacked retries, the 60 s default timeout × attempts, tool loops and `validateSchema()` retries, huge histories. Set a budget, one retry layer, a concurrency cap and fallback; stream or go async. Truncation is `stop_reason: max_tokens`: raise `max-tokens`, ask for less, use native structured output.

    **Interviewer listens for:** count the calls, budgets, `max_tokens` stop reason.

    **Common wrong answer:** "Use a faster model" without measuring where time goes.

## Cheat sheet

| Concept | Remember |
|---|---|
| Spring AI version | 2.0 GA June 2026 (2.0.1 current); Boot 4.x, Framework 7, Jackson 3 |
| `ChatClient` vs `ChatModel` | Fluent API + advisors vs provider-level `call`/`stream` |
| Tool loop | `ToolCallingAdvisor` (+300), auto-registered; `ChatModel` no longer executes tools |
| Memory | `MessageChatMemoryAdvisor` (+200), 20-message window, shared repository |
| Structured output | `.entity(T)` + `useProviderStructuredOutput()` + `validateSchema()` |
| Anthropic defaults | `claude-haiku-4-5`, 4096 max tokens, 60 s timeout, 2 retries |
| Claude Opus 5.5 | No temperature/top_p/top_k, no budgets, no forced tool choice; use `effort` |
| Errors | 429 + `retry-after`, 529 overloaded, 5xx: retry in one layer |
| Resilience | Timeout budget, bulkhead (`@ConcurrencyLimit`), breaker, fallback |
| Cost | Prompt caching, memory window, model routing, effort, batch |
| Observability | Micrometer `gen_ai.client.operation`, `gen_ai.client.token.usage`; prompt logging off |

## Sources
1. [Spring AI 2.0.0 GA announcement](https://spring.io/blog/2026/06/12/spring-ai-2-0-0-GA-available-now/): Boot 4 / Framework 7 baseline, Jackson 3, vendor SDK providers, `ToolCallingAdvisor`, structured output validation, MCP in core.
2. [Spring AI 2.0.1 release notes](https://spring.io/blog/2026/08/21/spring-ai-2-0-1-available-now): current maintenance release, tool call limit, Anthropic timeout fix.
3. [Spring AI reference: Upgrade notes, Upgrading to 2.0.0](https://docs.spring.io/spring-ai/reference/upgrade-notes.html): removal of the per-model tool loop and `internalToolExecutionEnabled`, advisor orders, options immutability, property flattening.
4. [Spring AI reference: Anthropic Chat](https://docs.spring.io/spring-ai/reference/api/chat/anthropic-chat.html): properties and defaults, prompt caching strategies, structured output and effort, rate-limit metadata, sampling parameters deprecated on newer models.
5. [Spring AI reference: ChatClient](https://docs.spring.io/spring-ai/reference/api/chatclient.html), [Structured output](https://docs.spring.io/spring-ai/reference/api/structured-output.html), [Tool calling](https://docs.spring.io/spring-ai/reference/api/tools.html) and [Chat memory](https://docs.spring.io/spring-ai/reference/api/chat-memory.html): fluent API, `EntityParamSpec`, tool limits and `ToolContext`, `MessageWindowChatMemory`.
6. [Spring AI reference: Observability](https://docs.spring.io/spring-ai/reference/observability/index.html): `gen_ai.*` observations and prompt/completion logging flags.
7. [Anthropic docs: Messages API](https://docs.claude.com/en/api/messages), [Errors](https://docs.claude.com/en/api/errors) and [Rate limits](https://docs.claude.com/en/api/rate-limits): request/response shape, stop reasons, 429/529, `retry-after`.
8. [Anthropic docs: Models overview](https://docs.claude.com/en/docs/about-claude/models/overview): current model ids and per-model parameter support (sampling parameters and forced tool choice on Claude Opus 5.5).
9. [Anthropic Java SDK](https://github.com/anthropics/anthropic-sdk-java): the SDK under Spring AI's Anthropic module; default retries and timeouts.
10. [Spring Framework reference: Resilience features](https://docs.spring.io/spring-framework/reference/core/resilience.html): `@Retryable`, `@ConcurrencyLimit`, `@EnableResilientMethods`.
11. [LangChain4j documentation](https://docs.langchain4j.dev/): the main Java alternative.
