# AWS Well-Architected RAG

A retrieval-augmented Q&A system over the AWS Well-Architected Framework documentation
(agentic-ai, devops-guidance, financial-services, machine-learning, sustainability, and
other lenses), built with LangChain, Pinecone, and Mistral — with an evaluation
harness used to make evidence-based retrieval architecture decisions rather than
defaulting to whatever technique is trendiest.

## Architecture

- **Retrieval**: dense (vector), sparse (BM25), and hybrid (RRF fusion) modes, all
  implemented behind a provider-agnostic interface/factory pattern — embedding model,
  LLM, and vector store are all swappable via config without touching core RAG logic.
- **Generation**: LangChain LCEL chains, provider-swappable.
- **Production default: dense retrieval**, with a lightweight regex-based fallback to
  sparse search for queries containing exact identifiers or doc codes where lexical
  match reliably outperforms semantic search.
- **Reliability**: retry-with-exponential-backoff on every external call (retrieval,
  generation, ingestion), a semaphore capping concurrent LLM calls, checkpointed batch
  ingestion (a killed/rate-limited run resumes without re-embedding completed batches),
  and graceful degradation — retrieval or generation failures return a structured error
  response instead of crashing the request.
- **Evaluation**: [deepeval](https://github.com/confident-ai/deepeval), covering
  Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, and
  Contextual Relevancy, run against a hand-built query set spanning five categories:
  exact-keyword, semantic-paraphrase, informal/realistic phrasing, multi-hop (answers
  spanning 2+ source documents), and unanswerable (out-of-corpus) queries. Eval runs
  are resumable from a previous run's output — a rate limit or quota wall partway
  through doesn't require re-scoring already-completed rows.

## Why dense, not hybrid, is the default

Hybrid retrieval (dense + sparse via Reciprocal Rank Fusion) was fully implemented and
evaluated, not skipped. The investigation:

1. **Found and fixed a real bug** in the RRF implementation — an operator-precedence
   error meant fused scores were increasing with rank instead of decaying, silently
   inverting the intended ranking logic.
2. **Built a confusion-matrix evaluation** (queries hybrid uniquely recovers vs. queries
   it uniquely breaks, relative to dense-only) rather than trusting aggregate hit rates,
   after finding that identical aggregate scores can hide completely different
   underlying behavior.
3. **Diagnosed a structural failure mode**: on queries where one retriever ranks a
   plausible-but-incorrect document very highly, RRF's summed reciprocal-rank scoring
   can promote that document over a correct document that scores moderately-but-
   consistently across both retrievers. Confirmed against real system output (not just
   RRF math on paper), and did not resolve across a sweep of `rrf_const` values
   (tested 40–300).
4. **Result**: hybrid showed zero net recovered queries over dense-only across a
   20-query confusion-matrix test, while carrying this known failure mode. Dense-only
   was adopted as the simpler, equally-accurate default; hybrid code is retained and
   selectable via a `search_mode` flag for future re-evaluation as the query set grows.

## Evaluation results

Run via deepeval over 30 queries spanning the categories above, comparing dense-only
and sparse-only retrieval end-to-end (retrieval → generation):

| Metric | Dense | Sparse |
|---|---|---|
| Faithfulness | 0.86 | 0.93 |
| Answer Relevancy | 0.71 | 0.79 |
| Contextual Precision | 0.60 | 0.63 |
| Contextual Recall | 0.51 | 0.65 |
| Contextual Relevancy | 0.59 | 0.63 |

Sparse scored competitively or ahead of dense on this run, including on several
semantic-paraphrase queries — a result that prompted a manual review of the query set
itself. That review found that queries across several categories (paraphrase, informal,
and multi-hop) retained enough of the source documents' own terminology or doc-title
phrasing to be reasonably findable by keyword search too, meaning they didn't cleanly
isolate dense retrieval's intended semantic advantage from sparse's lexical matching.
Dense remains the production default as the standard, well-understood choice, with a
tighter-constructed query set (deliberately avoiding all source vocabulary) tracked as
follow-up work.


## Provider swapping in practice

Embedding, LLM, and vector-store providers are swappable via the interface/factory
pattern without touching core RAG logic — adding a provider means implementing the
interface and registering it in the factory, a scoped change rather than a rewrite.


## What's implemented vs. scoped out

**Implemented**: dense/sparse/hybrid retrieval, RRF fusion (with a documented and fixed
bug), provider-agnostic architecture, a 5-metric deepeval harness with resumable runs,
a categorized 30-query evaluation set, a FastAPI endpoint, LangSmith tracing, retry/
backoff/checkpointing on retrieval, generation, and ingestion.

**Scoped out, deliberately**:
- **Reranking** — no evidence surfaced of a rank-order problem (correct documents were
  either retrieved deep and needed depth tuning, or genuinely absent), neither of which
  reranking fixes.
- **CRAG / Self-RAG** style corrective or reflective retrieval loops — not implemented;
  a natural next step once a larger eval set makes their marginal value measurable.
- **Multi-turn conversational retrieval** — would require a query-rewriting step before
  retrieval that resolves references from chat history, which changes what "correct
  retrieval" means (rewrite quality and retrieval quality become two separately-graded
  stages) and needs a different eval design than the one built here.
- **Circuit breaker / dead-letter queue** — retry-with-backoff and checkpointing handle
  transient failures for the current single-caller, sequential-batch scope. A
  production deployment with concurrent traffic would additionally need a circuit
  breaker to prevent retry storms from overwhelming a degraded upstream provider and to
  fail fast for waiting users; async ingestion at scale would warrant a queue-based
  architecture with a dead-letter queue for permanently failing batches. Neither
  pattern has a load condition to justify it yet in this project's current shape.
- **Online evaluation** — evaluation here is offline (run against a fixed test set,
  decoupled from the request path, never adding judge-model latency to a live
  response). In production this would extend to async, sampled scoring of live traffic
  to catch quality drift, with flagged failures reviewed and folded into the offline
  golden dataset — still never run synchronously in the request path.
- The eval schema includes an `intent` field to support future routing across
  RAG / SQL / web-search subsystems; only the RAG path is implemented in this version.

## Running it

```bash
# one-time setup
python -m src.scripts.ingest_docs --mode full-sync
python -m src.scripts.build_sparse_index

# start the API
uvicorn src.main:app --reload

# retrieval + generation
POST /query
{ "question": "How do I prevent an agent from getting overwhelmed by tool overload?" }

# evaluation (resumable — pass --resume-from a previous run's output to continue)
python -m evals.run_eval --profile dense
python -m evals.run_eval --profile dense --resume-from deepevals/output/<previous>.json

# latency benchmark (one-off, not a production monitoring tool)
python -m src.scripts.measure_latency
```

Tracing available via LangSmith when `LANGCHAIN_TRACING_V2=true` is set in `.env`.

## Stack

LangChain · Pinecone · Mistral  · deepeval · FastAPI · LangSmith