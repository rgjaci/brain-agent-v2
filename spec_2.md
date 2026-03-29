# spec.md — Brain Agent v2: Memory-First AI Agent

## Thesis

A 4B parameter model with an exceptional memory system can match frontier models on tasks where accumulated knowledge matters. The LLM is a processor. Memory is the intelligence. This project proves or disproves that thesis through a working agent with measurable benchmarks.

Research backing: RAFT-tuned small models match 30x larger models on domain tasks. Self-RAG achieves 35% accuracy gains with 40% less compute. Hierarchical procedural memory delivers 85% efficiency gains. Cross-encoder reranking improves retrieval 33-48%. Graph-augmented memory adds 18-47% on reasoning tasks. Combined, these techniques make the architecture more important than the parameters.

---

## What This Agent Does

A standalone personal AI agent that:

1. **Remembers everything** across sessions — facts, preferences, corrections, context
2. **Learns procedures** from successful problem-solving and replays them on similar problems
3. **Gets better at retrieval** over time through a feedback loop that learns what "relevant" means for you
4. **Reads and retains** books, guides, documentation permanently
5. **Acts on your machine** via bash, file operations, web search
6. **Shows its thinking** — full debug visibility into what memory retrieves and why

The agent is also a research instrument: it measures whether memory-augmented 4B can match Claude/GPT-4 on specific task types.

---

## The 5 Techniques That Make It Exceptional

### Technique 1: Retrieval-Specialized Context Usage

The model must be exceptionally good at USING retrieved context, not just receiving it. Standard models treat retrieved text as background noise. This agent treats it as instructions.

**Implementation:** The system prompt explicitly structures how the model should use retrieved content:

```
You are an AI assistant with access to a memory system.

PROCEDURE (if provided): Follow these steps exactly. They are a proven
approach to this type of problem. Adapt specifics to the current situation.
Skip steps that don't apply and explain why.

MEMORIES: These are facts and context from previous interactions. Treat
them as ground truth unless they contradict each other (prefer newer ones).

KG CONTEXT: These are entity relationships. Use them to understand how
things connect.

CHAT HISTORY: Recent conversation for continuity.

When answering:
1. If a PROCEDURE matches, follow it
2. Cite specific memories when making claims
3. If you're unsure, say so — don't hallucinate
4. If you need more information, use tools to get it
```

**Why this matters:** A 4B model following a good procedure with good context produces better output than a 70B model reasoning from scratch. The model doesn't need to be smart — it needs to follow instructions precisely.

**Future enhancement (Phase 5+):** RAFT fine-tuning — train the Qwen 4B model on examples of (question + relevant docs + distractor docs → answer with citations). This teaches it to discriminate relevant from irrelevant retrieved content. Research shows 30-76% improvement. Requires ~1000 training examples from accumulated interaction data.

### Technique 2: Adaptive Retrieval (Know When to Search)

Not every query needs memory retrieval. "Hello" doesn't. "Set up SSH like last time" does. Retrieving when unnecessary wastes tokens and can introduce noise that confuses a small model.

**Implementation:** Confidence-based triggering:

```python
class AdaptiveRetriever:
    """Decide whether to retrieve, and how much."""

    # Queries that never need retrieval
    SKIP_PATTERNS = ['hello', 'hi', 'thanks', 'ok', 'bye', 'yes', 'no']

    # Queries that always need retrieval
    ALWAYS_PATTERNS = ['last time', 'we discussed', 'remember', 'like before',
                       'set up', 'configure', 'debug', 'fix', 'deploy']

    def should_retrieve(self, query: str) -> RetrievalPlan:
        query_lower = query.lower().strip()

        # Fast path: skip patterns
        if any(p in query_lower for p in self.SKIP_PATTERNS) and len(query.split()) < 5:
            return RetrievalPlan(retrieve=False)

        # Always retrieve for memory-dependent queries
        if any(p in query_lower for p in self.ALWAYS_PATTERNS):
            return RetrievalPlan(
                retrieve=True,
                search_memories=True,
                search_procedures=True,
                search_kg=True,
                top_k=10
            )

        # Default: retrieve with moderate depth
        word_count = len(query.split())
        has_question = '?' in query

        if word_count > 15 or has_question:
            return RetrievalPlan(
                retrieve=True,
                search_memories=True,
                search_procedures=word_count > 20,
                search_kg=word_count > 10,
                top_k=5
            )

        # Short non-question: light retrieval
        return RetrievalPlan(
            retrieve=True,
            search_memories=True,
            search_procedures=False,
            search_kg=False,
            top_k=3
        )
```

**Why this matters:** Small models degrade with context overload. Research shows k=3-5 perfect retrievals outperform k=20 decent ones for models under 13B. Less is more when precision is high.

### Technique 3: Hierarchical Procedural Memory

When the agent solves a multi-step problem successfully, it extracts the reasoning pattern as a reusable procedure. Next time, memory provides the playbook — the model just follows it.

**Three tiers of procedures:**

```
Tier 1: ATOMIC ACTIONS (low-level, specific)
  "To check if a Python package is installed: pip list | grep <package>"

Tier 2: TASK PROCEDURES (mid-level, reusable)
  "Debug Python import error:
   1. Check which python (which python)
   2. Check venv (echo $VIRTUAL_ENV)
   3. Check if installed (pip list | grep <pkg>)
   4. If wrong env: activate correct venv
   5. If not installed: pip install <pkg>"

Tier 3: STRATEGY TEMPLATES (high-level, abstract)
  "When debugging any error:
   1. Reproduce the error
   2. Read the full error message carefully
   3. Identify the component that failed
   4. Check the most common cause for that component
   5. Verify the fix
   6. If fix works, document what happened"
```

**Bayesian reliability tracking:**

Each procedure tracks success/failure counts. Selection uses Upper Confidence Bound:

```python
def select_procedure(self, candidates: list[Procedure]) -> Procedure:
    """Select best procedure using UCB1 — balance exploitation vs exploration."""
    total_uses = sum(p.times_used for p in candidates)

    scores = []
    for p in candidates:
        if p.times_used == 0:
            scores.append((p, float('inf')))  # Always try untested procedures
            continue

        success_rate = p.times_succeeded / p.times_used
        exploration_bonus = math.sqrt(2 * math.log(total_uses + 1) / p.times_used)
        score = success_rate + exploration_bonus
        scores.append((p, score))

    return max(scores, key=lambda x: x[1])[0]
```

**Procedure extraction (runs after successful multi-step interactions):**

```python
EXTRACTION_PROMPT = """
This interaction successfully solved a problem using multiple steps.
Extract a reusable procedure from it.

Conversation:
{conversation}

Return JSON:
{
  "name": "short_descriptive_name",
  "description": "When to use this procedure",
  "trigger_pattern": "keywords or patterns that indicate this procedure applies",
  "preconditions": ["what must be true before starting"],
  "steps": ["step 1", "step 2", ...],
  "warnings": ["things to watch out for"],
  "context": "specific details about the user's environment that matter"
}

Only extract if the approach was genuinely multi-step and successful.
Return null if this was a simple Q&A.
"""
```

**Why this matters:** Research shows 85% reduction in LLM calls when reusing procedures. The agent gets faster and more reliable over time. Taught procedures (via `teach` command) get higher initial confidence than auto-extracted ones.

### Technique 4: Hybrid Search + Cross-Encoder Reranking

The retrieval pipeline is the most critical component. It determines what the model sees. Bad retrieval = bad output regardless of model quality.

**Four-stage pipeline:**

```
Query
  │
  ▼
┌─────────────────────────────────────────────┐
│ STAGE 1: CANDIDATE GENERATION (50-100 results)│
│                                               │
│ Run in parallel:                              │
│ ├─ Dense vector search (Gemini embeddings)    │
│ │  Top-50 by cosine similarity                │
│ ├─ BM25 keyword search (SQLite FTS5)          │
│ │  Top-50 by term frequency                   │
│ └─ KG entity lookup + 1-2 hop traversal       │
│    All connected entities + relations          │
│                                               │
│ Merge via Reciprocal Rank Fusion:             │
│ score(d) = Σ 1/(k + rank_i) × weight_i       │
│ weights: vector=0.5, bm25=0.3, kg=0.2        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ STAGE 2: CROSS-ENCODER RERANKING             │
│                                               │
│ Take top-20 from RRF                          │
│ Score each (query, memory) pair with           │
│ cross-encoder model (~30MB, runs on CPU)       │
│                                               │
│ Model: ms-marco-MiniLM-L6-v2 (or similar)     │
│ Latency: ~100-200ms for 20 candidates          │
│ Impact: +33-48% retrieval quality              │
│                                               │
│ Fallback (Phase 1): heuristic reranking        │
│ score = 0.35*similarity + 0.20*importance +    │
│         0.15*usefulness + 0.15*recency +       │
│         0.10*confidence + 0.05*kg_connected    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ STAGE 3: STRATEGIC ORDERING                   │
│                                               │
│ Small models have strong position bias:        │
│ - Best at using content at START of context    │
│ - Decent at using content at END               │
│ - WORST at using content in MIDDLE             │
│                                               │
│ Order: [most relevant, 3rd, 5th, ...,          │
│         4th, 2nd most relevant]                │
│ (interleave to put best at edges)              │
│                                               │
│ Procedures ALWAYS go first (after sys prompt)  │
│ Cost: 0ms. Impact: +10-15% accuracy.           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ STAGE 4: TOP-K SELECTION                      │
│                                               │
│ For 4B model: strict k=3-5 for memories        │
│ (plus 1 procedure if matched)                  │
│                                               │
│ More context hurts small models:               │
│ k=3 perfect > k=20 decent                     │
│                                               │
│ Budget: ~15K tokens for retrieved content       │
│ If over budget: drop lowest-ranked first        │
└─────────────────────────────────────────────┘
```

**Why this matters:** Research shows hybrid search (BM25 + dense) gives 21% improvement over either alone. Cross-encoder reranking adds another 33-48%. Strategic ordering adds 10-15% for free. Combined, retrieval quality roughly doubles compared to naive vector search.

### Technique 5: Graph-Augmented Memory Organization

Flat vector stores miss relationships. "User prefers ed25519" and "User uses Tailscale" are separate facts in a vector store. In a knowledge graph, they connect: User → prefers → ed25519, User → uses → Tailscale, Tailscale → supports → SSH, SSH → uses → ed25519. Now when asked "set up SSH for my server," the KG retrieves the whole connected cluster.

**Implementation: Lightweight KG in SQLite (no Neo4j)**

```python
def kg_retrieve(self, entity_names: list[str], max_hops: int = 2) -> str:
    """Traverse KG from seed entities, return formatted context."""
    all_facts = []
    visited = set()

    for name in entity_names:
        # Find entity (exact match first, then vector similarity)
        entity = self.find_entity(name)
        if not entity:
            continue

        # BFS traversal
        queue = [(entity.id, 0)]
        visited.add(entity.id)

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_hops:
                continue

            relations = self.db.execute('''
                SELECT r.relation_type, r.confidence,
                       CASE WHEN r.source_entity_id = ?
                            THEN e2.name ELSE e1.name END as other_name,
                       CASE WHEN r.source_entity_id = ?
                            THEN e2.description ELSE e1.description END as other_desc
                FROM relations r
                JOIN entities e1 ON r.source_entity_id = e1.id
                JOIN entities e2 ON r.target_entity_id = e2.id
                WHERE r.source_entity_id = ? OR r.target_entity_id = ?
                ORDER BY r.confidence DESC
                LIMIT 10
            ''', (current_id, current_id, current_id, current_id))

            for rel in relations:
                fact = f"{name} {rel['relation_type']} {rel['other_name']}"
                if rel['other_desc']:
                    fact += f" ({rel['other_desc']})"
                all_facts.append(fact)

                # Continue traversal
                other_id = self.get_entity_id(rel['other_name'])
                if other_id and other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, depth + 1))

    return "\n".join(all_facts[:20])  # Cap at 20 facts
```

**Temporal validity:** Facts change. Store `valid_from` and `valid_until` on relations. When contradictions are detected, mark old fact as superseded rather than deleting (preserves history).

**Why this matters:** On multi-hop questions, vector RAG achieves ~18% accuracy while graph-augmented reaches ~73%. The gap is largest on exactly the queries where accumulated knowledge matters most — "how does X relate to Y given what we did last month."

---

## Database Schema

Single file: `~/.brain_agent/memory.db`

```sql
-- ================================================================
-- MEMORIES: Everything the agent knows (vector-indexed)
-- ================================================================
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL,           -- 'fact', 'preference', 'observation',
                                      -- 'document_chunk', 'search_result',
                                      -- 'conversation_summary', 'correction'
    source TEXT,                      -- 'conversation:{session_id}',
                                      -- 'document:{title}', 'web:{url}', 'taught'
    importance REAL DEFAULT 0.5,      -- 0.0-1.0
    confidence REAL DEFAULT 0.7,      -- 0.0-1.0
    superseded_by INTEGER,            -- points to newer version if corrected
    access_count INTEGER DEFAULT 0,
    usefulness_score REAL DEFAULT 0.5, -- learned from retrieval feedback
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                     -- JSON blob
);

CREATE VIRTUAL TABLE memory_vectors USING vec0(
    memory_id INTEGER,
    embedding float[768]
);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    content, category, source,
    content='memories', content_rowid='id'
);

-- ================================================================
-- KNOWLEDGE GRAPH: Entities and relationships
-- ================================================================
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,        -- 'person', 'tool', 'concept', 'project',
                                      -- 'file', 'service', 'language', 'config'
    description TEXT,
    properties TEXT,                  -- JSON
    importance REAL DEFAULT 0.5,
    source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE entity_vectors USING vec0(
    entity_id INTEGER,
    embedding float[768]
);

CREATE TABLE relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,      -- 'uses', 'prefers', 'part_of', 'causes',
                                      -- 'depends_on', 'contradicts', 'instance_of',
                                      -- 'located_in', 'created_by', 'configured_with'
    properties TEXT,                  -- JSON
    confidence REAL DEFAULT 0.7,
    valid_from DATETIME DEFAULT CURRENT_TIMESTAMP,
    valid_until DATETIME,             -- NULL = still valid
    source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

-- ================================================================
-- PROCEDURES: Reasoning templates
-- ================================================================
CREATE TABLE procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_pattern TEXT,             -- keywords/patterns for matching
    preconditions TEXT,               -- JSON array
    steps TEXT NOT NULL,              -- JSON array
    warnings TEXT,                    -- JSON array
    context TEXT,                     -- environment-specific notes
    -- Effectiveness tracking
    times_used INTEGER DEFAULT 0,
    times_succeeded INTEGER DEFAULT 0,
    times_failed INTEGER DEFAULT 0,
    -- Metadata
    source TEXT,                      -- 'taught', 'learned:{session}', 'documented:{file}'
    related_entities TEXT,            -- JSON array of entity IDs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used DATETIME
);

CREATE VIRTUAL TABLE procedure_vectors USING vec0(
    procedure_id INTEGER,
    embedding float[768]
);

-- ================================================================
-- RETRIEVAL FEEDBACK: Trains the learned reranker
-- ================================================================
CREATE TABLE retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    memory_id INTEGER NOT NULL,
    retrieval_method TEXT,            -- 'vector', 'bm25', 'kg', 'procedure'
    retrieval_rank INTEGER,
    retrieval_score REAL,
    was_in_context INTEGER,          -- 1 if included in final context
    was_useful INTEGER,              -- 1 if response was accepted (not corrected)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Precomputed features for reranker training
CREATE TABLE reranker_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    retrieval_log_id INTEGER NOT NULL,
    cosine_similarity REAL,
    bm25_score REAL,
    memory_importance REAL,
    memory_confidence REAL,
    memory_age_hours REAL,
    memory_access_count INTEGER,
    memory_usefulness REAL,
    memory_category TEXT,
    query_length INTEGER,
    has_kg_connection INTEGER,
    kg_hops INTEGER,
    label INTEGER,                   -- 0 or 1 (was_useful)
    FOREIGN KEY (retrieval_log_id) REFERENCES retrieval_log(id)
);

-- ================================================================
-- DOCUMENTS: Hierarchical index
-- ================================================================
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source_path TEXT,
    doc_type TEXT,                    -- 'book', 'guide', 'article', 'code'
    summary TEXT,
    total_chunks INTEGER DEFAULT 0,
    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    parent_section_id INTEGER,
    title TEXT,
    summary TEXT,
    level INTEGER DEFAULT 0,         -- 0=doc, 1=chapter, 2=section, 3=subsection
    position REAL,                   -- 0.0-1.0 within parent
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- ================================================================
-- CONVERSATIONS: Raw session log
-- ================================================================
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,               -- 'user', 'assistant', 'tool_call', 'tool_result'
    content TEXT NOT NULL,
    token_count INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                     -- JSON: tools_used, memories_retrieved, etc.
);

-- ================================================================
-- INDEXES
-- ================================================================
CREATE INDEX idx_mem_category ON memories(category);
CREATE INDEX idx_mem_importance ON memories(importance DESC);
CREATE INDEX idx_mem_usefulness ON memories(usefulness_score DESC);
CREATE INDEX idx_mem_superseded ON memories(superseded_by);
CREATE INDEX idx_ent_type ON entities(entity_type);
CREATE INDEX idx_ent_name ON entities(name);
CREATE INDEX idx_rel_source ON relations(source_entity_id);
CREATE INDEX idx_rel_target ON relations(target_entity_id);
CREATE INDEX idx_rel_type ON relations(relation_type);
CREATE INDEX idx_rel_valid ON relations(valid_until);
CREATE INDEX idx_conv_session ON conversations(session_id, timestamp);
CREATE INDEX idx_retlog_session ON retrieval_log(session_id);
CREATE INDEX idx_retlog_useful ON retrieval_log(was_useful);
```

---

## Memory Write Pipeline

Runs asynchronously after every response is sent to the user.

```python
class MemoryWriter:
    """Extract knowledge from interactions and store it."""

    async def process_interaction(self, user_msg: str, agent_msg: str,
                                  tool_calls: list, session_id: str):
        """Main entry point. Runs after response is sent."""

        # Step 1: Extract facts (LLM call, ~200 tokens)
        facts = await self.extract_facts(user_msg, agent_msg)

        # Step 2: Extract entities and relations (LLM call, ~300 tokens)
        entities, relations = await self.extract_graph(user_msg, agent_msg)

        # Step 3: Extract procedure if multi-step success (conditional LLM call)
        procedure = None
        if len(tool_calls) >= 2 and self.interaction_succeeded(agent_msg):
            procedure = await self.extract_procedure(user_msg, agent_msg, tool_calls)

        # Step 4: Deduplicate against existing memories (embedding comparison)
        new_facts = await self.deduplicate_facts(facts)
        new_entities = await self.deduplicate_entities(entities)

        # Step 5: Embed and store
        for fact in new_facts:
            embedding = await self.embedder.embed([fact.content])
            memory_id = self.db.insert_memory(fact)
            self.db.insert_embedding(memory_id, embedding[0])

        for entity in new_entities:
            embedding = await self.embedder.embed([entity.name + " " + entity.description])
            entity_id = self.db.upsert_entity(entity)
            self.db.insert_entity_embedding(entity_id, embedding[0])

        for relation in relations:
            self.db.upsert_relation(relation)

        if procedure:
            embedding = await self.embedder.embed(
                [procedure.name + " " + procedure.description])
            proc_id = self.db.insert_procedure(procedure)
            self.db.insert_procedure_embedding(proc_id, embedding[0])

    FACT_EXTRACTION_PROMPT = """Extract new facts from this exchange.
Only extract genuinely NEW information — things not obvious or common knowledge.

User: {user_msg}
Assistant: {agent_msg}

Return JSON array (or empty array if nothing new):
[{{"content": "...", "category": "fact|preference|observation|correction", "importance": 0.0-1.0}}]"""

    GRAPH_EXTRACTION_PROMPT = """Identify entities and relationships in this exchange.

User: {user_msg}
Assistant: {agent_msg}

Return JSON:
{{"entities": [{{"name": "...", "type": "person|tool|project|concept|service|config", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "uses|prefers|part_of|depends_on|configured_with", "detail": "..."}}]}}

Only extract clear, specific entities — not generic words."""
```

**Token cost per interaction:** ~500-800 tokens through local LLM. At Qwen 4B speeds, this is ~1-3 seconds. Runs in background, doesn't block the user.

---

## Memory Read Pipeline

```python
class MemoryReader:
    """Retrieve and assemble context for the LLM."""

    async def retrieve(self, query: str, session_id: str) -> AssembledContext:
        """Main entry point for memory retrieval."""

        # Step 0: Decide whether to retrieve at all
        plan = self.adaptive_retriever.should_retrieve(query)
        if not plan.retrieve:
            return AssembledContext(memories=[], procedure=None, kg_context="")

        # Step 1: Query analysis (LLM call, ~150 tokens)
        analysis = await self.analyze_query(query)
        # Returns: {entities: [...], topics: [...], procedure_type: "..."}

        # Step 2: Parallel candidate generation
        candidates = []

        if plan.search_memories:
            # Dense vector search
            query_embedding = await self.embedder.embed_query(query)
            vector_results = self.db.vector_search(
                query_embedding, table='memory_vectors', limit=50)

            # BM25 keyword search
            bm25_results = self.db.fts_search(query, table='memory_fts', limit=50)

            # Merge via RRF
            candidates = self.reciprocal_rank_fusion(
                [vector_results, bm25_results],
                weights=[0.5, 0.3],
                k=60
            )

        # KG traversal
        kg_context = ""
        if plan.search_kg and analysis.entities:
            kg_context = self.kg.traverse(analysis.entities, max_hops=2)

        # Procedure matching
        procedure = None
        if plan.search_procedures:
            procedure = await self.find_matching_procedure(query, analysis)

        # Step 3: Reranking
        if self.reranker_model:
            # Trained cross-encoder (Phase 3+)
            scored = self.reranker_model.score(query, candidates)
        else:
            # Heuristic reranking (Phase 1-2)
            scored = self.heuristic_rerank(query, candidates)

        # Step 4: Top-k selection with position ordering
        top_k = scored[:plan.top_k]
        ordered = self.strategic_order(top_k)  # Best at edges, worst in middle

        # Step 5: Assemble within budget
        context = self.assemble_context(
            procedure=procedure,
            memories=ordered,
            kg_context=kg_context,
            budget=TOKEN_BUDGET
        )

        # Step 6: Log for reranker training
        self.log_retrieval(session_id, query, scored, context)

        return context

    def reciprocal_rank_fusion(self, result_lists, weights, k=60):
        """Merge ranked lists using RRF."""
        scores = defaultdict(float)
        for results, weight in zip(result_lists, weights):
            for rank, item in enumerate(results):
                scores[item.id] += weight * (1.0 / (k + rank))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def strategic_order(self, memories):
        """Place most relevant at start and end, least relevant in middle."""
        if len(memories) <= 2:
            return memories
        # Interleave: best, 3rd, 5th, ..., 6th, 4th, 2nd
        first_half = memories[0::2]  # Odd positions (0, 2, 4...)
        second_half = memories[1::2][::-1]  # Even positions reversed
        return first_half + second_half

    def heuristic_rerank(self, query, candidates):
        """Multi-signal reranking without ML."""
        scored = []
        now = time.time()
        for mem_id, rrf_score in candidates:
            mem = self.db.get_memory(mem_id)
            recency = 1.0 / (1.0 + (now - mem.last_accessed.timestamp()) / 86400)
            score = (
                0.30 * rrf_score +
                0.20 * mem.importance +
                0.15 * mem.usefulness_score +
                0.15 * recency +
                0.10 * mem.confidence +
                0.10 * (1.0 if self.kg.has_connection(query, mem_id) else 0.0)
            )
            scored.append((mem, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)
```

---

## Context Assembly

The 32K token budget is sacred. Every token must earn its place.

```python
TOKEN_BUDGET = {
    'system_prompt':     500,   # Identity + instructions
    'procedure':        2000,   # Reasoning template (if matched)
    'kg_context':       1500,   # Entity relationships
    'memories':        13000,   # Retrieved facts/observations/chunks
    'chat_history':     6000,   # Recent conversation
    'tool_buffer':      2000,   # Space for tool calls/results
    'output_reserve':   4000,   # For model's response
    'query':             500,   # Current user message
    'overhead':         1268,   # Tokenizer differences, formatting
    # Total:          30768    # Leaves ~2K margin under 32768
}

class ContextAssembler:
    """Pack the 32K context window optimally."""

    def assemble(self, procedure, memories, kg_context,
                 chat_history, query, budget=TOKEN_BUDGET):

        sections = []

        # 1. System prompt (FIXED, never compressed)
        sections.append(("system", self.system_prompt, budget['system_prompt']))

        # 2. Procedure goes RIGHT AFTER system prompt (highest priority)
        if procedure:
            proc_text = self.format_procedure(procedure)
            sections.append(("procedure", proc_text, budget['procedure']))

        # 3. KG context
        if kg_context:
            sections.append(("kg", kg_context, budget['kg_context']))

        # 4. Retrieved memories (fill remaining space)
        used_tokens = sum(s[2] for s in sections) + budget['query'] + \
                      budget['tool_buffer'] + budget['output_reserve']
        memory_budget = 32768 - used_tokens - budget['overhead']

        mem_text = self.pack_memories(memories, memory_budget)
        sections.append(("memories", mem_text, len(mem_text)))  # actual tokens

        # 5. Chat history (compressed if needed)
        history_budget = min(budget['chat_history'],
                            32768 - sum(s[2] for s in sections) -
                            budget['query'] - budget['tool_buffer'] -
                            budget['output_reserve'] - budget['overhead'])
        history = self.compress_history(chat_history, history_budget)
        sections.append(("history", history, len(history)))

        # 6. Current query
        sections.append(("query", query, budget['query']))

        return self.format_sections(sections)

    def compress_history(self, messages, budget):
        """Progressive compression: recent verbatim, older summarized."""
        recent = messages[-5:]
        recent_tokens = self.count_tokens(recent)

        if recent_tokens >= budget:
            return self.truncate(recent, budget)

        remaining = budget - recent_tokens
        older = messages[:-5]

        if not older:
            return self.format_messages(recent)

        # Summarize older messages in batches of 10
        batches = [older[i:i+10] for i in range(0, len(older), 10)]
        summaries = []
        per_batch = remaining // max(len(batches), 1)

        for batch in batches:
            summary = self.llm.generate(
                f"Summarize in {per_batch} tokens. Keep: decisions, facts, "
                f"preferences, unresolved questions.\n\n{self.format_messages(batch)}",
                max_tokens=per_batch
            )
            summaries.append(f"[Earlier context: {summary}]")

        return "\n".join(summaries) + "\n" + self.format_messages(recent)
```

---

## Retrieval Feedback Loop (Learned Reranker)

The system tracks which retrievals were actually useful and trains a model to improve.

```python
class RetrievalFeedbackCollector:
    """Collect signals about retrieval quality."""

    def log_retrieval(self, session_id, query, scored_candidates, final_context):
        """Log what was retrieved and what was used."""
        context_ids = {m.id for m in final_context.memories}

        for rank, (mem, score) in enumerate(scored_candidates[:20]):
            self.db.insert_retrieval_log(
                session_id=session_id,
                query_text=query,
                memory_id=mem.id,
                retrieval_rank=rank,
                retrieval_score=score,
                was_in_context=1 if mem.id in context_ids else 0,
                was_useful=None  # Set later based on outcome
            )

    def record_outcome(self, session_id, user_accepted: bool):
        """After response: did user accept or correct?"""
        # Mark all retrievals in this turn
        self.db.execute('''
            UPDATE retrieval_log
            SET was_useful = ?
            WHERE session_id = ? AND was_in_context = 1
            AND was_useful IS NULL
        ''', (1 if user_accepted else 0, session_id))

        # Also update memory usefulness scores
        if user_accepted:
            self.db.execute('''
                UPDATE memories SET usefulness_score = MIN(1.0,
                    usefulness_score + 0.05)
                WHERE id IN (
                    SELECT memory_id FROM retrieval_log
                    WHERE session_id = ? AND was_in_context = 1
                )
            ''', (session_id,))


class LearnedReranker:
    """Train a reranker from retrieval feedback (Phase 4)."""

    def train(self):
        """Train logistic regression on accumulated feedback."""
        features = self.db.execute('''
            SELECT cosine_similarity, bm25_score, memory_importance,
                   memory_confidence, memory_age_hours, memory_access_count,
                   memory_usefulness, has_kg_connection, kg_hops, label
            FROM reranker_training
            WHERE label IS NOT NULL
        ''').fetchall()

        if len(features) < 200:
            return None  # Not enough data yet

        X = np.array([row[:-1] for row in features])
        y = np.array([row[-1] for row in features])

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Evaluate
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"Reranker accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        return model

    def score(self, query, candidates):
        """Use trained model to rerank candidates."""
        features = [self.extract_features(query, mem) for mem in candidates]
        probabilities = self.model.predict_proba(np.array(features))[:, 1]
        return sorted(zip(candidates, probabilities),
                      key=lambda x: x[1], reverse=True)
```

**Progression:**

- Phase 1-2: Heuristic reranking (works from day 1)
- Phase 3: Logistic regression (after ~200+ labeled interactions)
- Phase 4+: Cross-encoder fine-tuning (after ~1000+ interactions, optional)

---

## Agent Core Loop

```python
class BrainAgent:
    """Main agent loop: query → memory → LLM → response → learn."""

    def __init__(self, config: AgentConfig):
        self.llm = OllamaProvider(model=config.model)
        self.embedder = GeminiEmbeddingProvider(api_key=config.gemini_api_key)
        self.db = MemoryDatabase(config.db_path)
        self.writer = MemoryWriter(self.llm, self.embedder, self.db)
        self.reader = MemoryReader(self.embedder, self.db)
        self.assembler = ContextAssembler(self.llm)
        self.tool_executor = ToolExecutor(config.permissions)
        self.feedback = RetrievalFeedbackCollector(self.db)
        self.session_id = str(uuid.uuid4())

    async def process(self, user_input: str) -> str:
        """Process one user message. Returns agent response."""

        # 1. Store user message
        self.db.store_conversation(self.session_id, 'user', user_input)

        # 2. Retrieve relevant memory
        context = await self.reader.retrieve(user_input, self.session_id)

        # 3. Get chat history
        history = self.db.get_recent_messages(self.session_id, limit=20)

        # 4. Assemble full context
        messages = self.assembler.assemble(
            procedure=context.procedure,
            memories=context.memories,
            kg_context=context.kg_context,
            chat_history=history,
            query=user_input
        )

        # 5. LLM call (potentially with tool loop)
        response = await self.tool_loop(messages, max_iterations=10)

        # 6. Store response
        self.db.store_conversation(self.session_id, 'assistant', response)

        # 7. Async: extract and store new knowledge
        asyncio.create_task(
            self.writer.process_interaction(
                user_input, response, self.current_tool_calls, self.session_id
            )
        )

        # 8. Log retrieval outcome (assume accepted unless corrected)
        self.feedback.record_outcome(self.session_id, user_accepted=True)

        return response

    async def tool_loop(self, messages, max_iterations=10):
        """Execute tool calls until done or max iterations."""
        for i in range(max_iterations):
            response = self.llm.generate(messages)

            # Parse tool calls from response
            tool_calls = self.parse_tool_calls(response)

            if not tool_calls:
                return response  # No tools needed, we're done

            # Execute each tool call
            for call in tool_calls:
                result = await self.tool_executor.execute(
                    call.name, call.params
                )
                self.current_tool_calls.append(call)

                # Add tool result to messages
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    f"<tool_result name=\"{call.name}\">\n{result}\n</tool_result>"})

        return response  # Max iterations reached
```

---

## Tool System

### Tools Available

| Tool         | Purpose                             | Phase |
| ------------ | ----------------------------------- | ----- |
| `bash`       | Run shell commands                  | 1     |
| `read_file`  | Read file contents                  | 1     |
| `write_file` | Create/overwrite file               | 1     |
| `edit_file`  | Search and replace in file          | 2     |
| `web_search` | Search web, store results in memory | 2     |
| `teach`      | Direct memory/procedure insertion   | 1     |
| `recall`     | Explicit memory search              | 1     |
| `ingest`     | Read and memorize a document        | 3     |

### Tool Call Format

```xml
<tool name="bash">
<param name="command">ls -la ~/.ssh/</param>
</tool>
```

Parsing: regex primary → Pydantic validation → execute → error feedback to LLM.

### Safety

```yaml
# ~/.brain_agent/config.yaml
permissions:
  read_allowed: ["~/.brain_agent/**", "${CWD}/**", "~/projects/**"]
  read_blocked: ["~/.ssh/id_*", "~/.aws/**", "**/.env"]
  write_allowed: ["~/.brain_agent/workspace/**", "${CWD}/**"]
  write_blocked: ["~/.bashrc", "~/.zshrc", "~/.brain_agent/config.yaml"]
  bash_blocked_patterns: ["rm -rf /", "sudo rm", "chmod 777", "> /dev/sd"]
  bash_timeout_default: 30
  bash_timeout_max: 300
  network_allowed: ["generativelanguage.googleapis.com", "api.duckduckgo.com"]
  network_default: "prompt_user"
```

---

## `teach` Command

Direct knowledge insertion bypassing extraction:

```
teach: when I say "deploy", run:
  1. git add -A && git commit -m "deploy"
  2. git push origin main
  3. ssh prod 'cd /app && git pull && systemctl restart app'

teach: my SSH passphrase is in my password manager, never ask for it
teach: for nhproject.org, DNS is in Azure, not Google
teach: I prefer concise answers unless I ask for detail
teach: when debugging Docker, always check docker logs first
```

Taught items get `confidence=0.95` and `source='taught'`.

---

## Bootstrap (First Run)

```python
async def bootstrap(self):
    """Scan environment on first run."""
    scans = [
        ("Shell config", "cat ~/.zshrc 2>/dev/null || cat ~/.bashrc 2>/dev/null"),
        ("Git config", "git config --global --list 2>/dev/null"),
        ("SSH keys", "ls ~/.ssh/*.pub 2>/dev/null"),
        ("Python", "python3 --version 2>/dev/null && which python3"),
        ("Node", "node --version 2>/dev/null && which node"),
        ("Docker", "docker --version 2>/dev/null"),
        ("Ollama models", "ollama list 2>/dev/null"),
        ("Recent projects", "find ~/projects -maxdepth 1 -type d 2>/dev/null | head -20"),
        ("Running services", "ss -tlnp 2>/dev/null | head -20"),
        ("OS info", "uname -a"),
        ("Disk usage", "df -h / | tail -1"),
    ]

    for name, cmd in scans:
        result = await self.tool_executor.execute("bash", {"command": cmd, "timeout": 5})
        if result.success and result.output.strip():
            # Extract facts and entities from scan results
            await self.writer.extract_from_scan(name, result.output)
```

---

## Background Consolidation

Runs during idle time (5+ minutes no interaction):

```python
class MemoryConsolidator:
    """Background memory maintenance."""

    async def consolidate(self):
        """Run all maintenance tasks."""
        await self.merge_duplicates()       # Similarity > 0.92
        await self.resolve_contradictions()  # Similar but different content
        await self.strengthen_kg()           # Infer missing relations
        await self.update_usefulness()       # Decay unused, boost used
        await self.compress_old_sessions()   # Summarize sessions > 1 week old
        await self.train_reranker()          # Retrain if enough new data

    async def merge_duplicates(self):
        """Find and merge near-duplicate memories."""
        # Get all memory embeddings
        # For each pair with similarity > 0.92, LLM merges into one
        # Mark original as superseded
        pass

    async def resolve_contradictions(self):
        """Find memories that contradict each other."""
        # High similarity (>0.8) but different content
        # LLM decides which is correct (prefer newer, higher confidence)
        # Mark loser as superseded
        pass

    async def strengthen_kg(self):
        """Find entities that co-occur but lack explicit relations."""
        # Query: entities that appear in same memories but have no relation
        # LLM infers relationship type
        # Add relation with moderate confidence
        pass
```

---

## Document Ingestion

```python
class DocumentIngestor:
    """Read and memorize documents."""

    async def ingest(self, path: str, doc_type: str = 'guide'):
        """Process a document into memory."""

        # 1. Read and parse
        text = self.read_document(path)

        # 2. Chunk (512 tokens, 50 token overlap)
        chunks = self.recursive_split(text, chunk_size=512, overlap=50)

        # 3. Create document record
        doc_id = self.db.insert_document(
            title=Path(path).stem,
            source_path=path,
            doc_type=doc_type,
            total_chunks=len(chunks)
        )

        # 4. Generate document summary (LLM call)
        summary = await self.llm.generate(
            f"Summarize this document in 200 words:\n\n{text[:3000]}")
        self.db.update_document_summary(doc_id, summary)

        # 5. Process each chunk
        for i, chunk in enumerate(chunks):
            # Store as memory
            embedding = await self.embedder.embed([chunk])
            memory_id = self.db.insert_memory(
                content=chunk,
                category='document_chunk',
                source=f'document:{Path(path).stem}',
                importance=0.4,
                metadata=json.dumps({
                    'document_id': doc_id,
                    'chunk_index': i,
                    'position': i / len(chunks)
                })
            )
            self.db.insert_embedding(memory_id, embedding[0])

            # Extract entities and relations (every 5th chunk to save tokens)
            if i % 5 == 0:
                entities, relations = await self.writer.extract_graph(
                    chunk, "")
                for entity in entities:
                    self.db.upsert_entity(entity)
                for relation in relations:
                    self.db.upsert_relation(relation)

        return doc_id
```

---

## TUI Debug View

Built with `textual`:

```
┌─ Brain Agent ─────────────────────────────────────────────────┐
│ ┌─ Conversation ──────────────────────────────────────────── ┐│
│ │ You: Set up SSH keys for the new server                    ││
│ │                                                            ││
│ │ Agent: Found a procedure from our last SSH setup.          ││
│ │ Following the steps with your current server...            ││
│ │                                                            ││
│ │ > ssh-keygen -t ed25519 -C "rei@macbook"                   ││
│ └────────────────────────────────────────────────────────────┘│
│ ┌─ Memory Activity ──────────────────────────────────────── ┐│
│ │ [ADAPT] Full retrieval — trigger: "set up"                 ││
│ │ [QUERY] Entities: SSH, ed25519, server                     ││
│ │ [VECT ] 50 candidates, top sim: 0.91                       ││
│ │ [BM25 ] 23 candidates for "SSH keys server"                ││
│ │ [KG   ] SSH→prefers→ed25519 (conf:0.95)                    ││
│ │ [KG   ] SSH→used_in→Tailscale (conf:0.85)                  ││
│ │ [RRF  ] Merged → 42 unique candidates                      ││
│ │ [RANK ] Heuristic rerank → top 5 selected                  ││
│ │ [PROC ] Match: "ssh_key_setup" (3x used, 100% success)     ││
│ │ [CTX  ] Assembled: proc:1800 + mem:8400 + hist:4200        ││
│ │ [LLM  ] qwen3.5:4b → 312 tokens, 1.9s                     ││
│ │ [WRITE] +2 facts, +1 entity, +1 relation (async)           ││
│ └────────────────────────────────────────────────────────────┘│
│ ┌─ Tokens ──────────┐ ┌─ Knowledge ─────────────────────── ┐│
│ │ █████████░░░ 24K   │ │ Memories: 1,247  Entities: 342    ││
│ │ Sys:500 Proc:1800  │ │ Relations: 891   Procedures: 28   ││
│ │ KG:1200 Mem:8400   │ │ Documents: 12    Reranker: heur.  ││
│ │ Hist:4200 Out:4000 │ │ Feedback: 847 labeled pairs       ││
│ └────────────────────┘ └────────────────────────────────────┘│
│ > _                                                           │
└───────────────────────────────────────────────────────────────┘
```

---

## Model Abstraction

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str: ...
    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

class OllamaProvider(LLMProvider):
    def __init__(self, model="qwen3.5:4b-nothink",
                 base_url="http://localhost:11434"): ...
    def generate(self, messages, temperature=0.3, max_tokens=2000): ...

class EmbeddingProvider:
    """Gemini embedding-001 via free tier."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
```

Start with Qwen 3.5 4B. Swap to any Ollama model or OpenRouter cloud model later without changing any other code.

---

## Project Structure

```
brain_agent/
├── spec.md                     # THIS FILE
├── pyproject.toml
├── core/
│   ├── __init__.py
│   ├── agent.py                # Main loop: query → memory → LLM → response → learn
│   ├── config.py               # AgentConfig dataclass
│   ├── llm/
│   │   ├── provider.py         # LLMProvider + OllamaProvider
│   │   ├── embeddings.py       # GeminiEmbeddingProvider
│   │   └── tool_parser.py      # Parse XML tool calls from LLM output
│   ├── memory/
│   │   ├── database.py         # SQLite + sqlite-vec setup, schema, migrations
│   │   ├── writer.py           # Extract facts, entities, relations, procedures
│   │   ├── reader.py           # Hybrid search, RRF, adaptive retrieval
│   │   ├── kg.py               # Knowledge graph: entity/relation CRUD + traversal
│   │   ├── procedures.py       # Procedure store, matching, Bayesian selection
│   │   ├── reranker.py         # Heuristic → logistic regression → cross-encoder
│   │   ├── consolidation.py    # Background merge, resolve, strengthen, decay
│   │   ├── documents.py        # Document ingestion + hierarchical summarization
│   │   └── feedback.py         # Retrieval outcome logging + feature extraction
│   ├── context/
│   │   ├── assembler.py        # 32K budget packing + strategic ordering
│   │   └── compressor.py       # Progressive history summarization
│   └── tools/
│       ├── executor.py         # Sandboxed execution + Pydantic validation
│       ├── bash.py             # Shell with safety checks
│       ├── file_ops.py         # read/write/edit with permissions
│       ├── web_search.py       # DuckDuckGo + store results in memory
│       ├── teach.py            # Direct memory/procedure insertion
│       └── ingest.py           # Document ingestion trigger
├── tui/
│   ├── app.py                  # Textual main app
│   ├── panels.py               # Conversation, memory activity, stats, budget
│   └── events.py               # Event bus for real-time debug updates
├── benchmarks/
│   ├── recall_test.py          # Tell facts, ask later, measure recall
│   ├── procedure_test.py       # Teach procedures, trigger, measure accuracy
│   ├── retrieval_quality.py    # Measure precision@k on held-out queries
│   ├── vs_baseline.py          # Compare: 4B raw vs 4B+memory vs frontier model
│   └── reranker_eval.py        # A/B test heuristic vs trained reranker
└── tests/
    ├── test_memory_write.py
    ├── test_memory_read.py
    ├── test_kg.py
    ├── test_context_assembly.py
    ├── test_procedures.py
    ├── test_reranker.py
    ├── test_tool_parser.py
    └── test_adaptive_retrieval.py
```

---

## Dependencies

```toml
[project]
name = "brain-agent"
version = "2.0.0"
requires-python = ">=3.11"

dependencies = [
    "requests",              # Ollama API
    "google-genai",          # Gemini embeddings (free tier)
    "sqlite-vec",            # Vector search in SQLite
    "tiktoken",              # Token counting
    "pydantic>=2.0",         # Tool call validation
    "textual>=0.80",         # TUI framework
    "rich>=13.0",            # Rich terminal output
]

[project.optional-dependencies]
ml = [
    "scikit-learn",          # Learned reranker (Phase 3+)
    "numpy",                 # Feature computation
]
reranker = [
    "sentence-transformers", # Cross-encoder reranking (Phase 4+)
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio",
]
```

Zero heavy frameworks. No LangChain, no LangGraph.

---

## Implementation Phases

### Phase 1: Memory Foundation (Weeks 1-2)

Build the core loop with basic memory.

- [ ] SQLite database with full schema + sqlite-vec
- [ ] Gemini embedding-001 integration (embed, embed_query)
- [ ] Ollama provider for Qwen 3.5 4B
- [ ] Memory writer: fact extraction (LLM call → dedup → embed → store)
- [ ] Memory reader: vector search → heuristic rerank → top-k selection
- [ ] Context assembler with token budget enforcement
- [ ] Agent main loop (CLI, not TUI yet)
- [ ] Tools: bash, read_file, write_file
- [ ] `teach` command (direct memory/procedure insertion)
- [ ] `recall` command (explicit memory search)
- [ ] Bootstrap: environment scan on first run
- [ ] Conversation logging

**Benchmark at exit:** Tell agent 10 facts on day 1. Ask about them on day 3. Measure recall %.

### Phase 2: Knowledge Graph + Procedures (Weeks 3-4)

Add relational reasoning and learned workflows.

- [ ] Entity/relation extraction in write pipeline
- [ ] KG traversal in read pipeline
- [ ] BM25 search via FTS5
- [ ] Reciprocal Rank Fusion (merge vector + BM25 + KG)
- [ ] Adaptive retrieval (know when to search)
- [ ] Procedure extraction from successful multi-step interactions
- [ ] Procedure matching and context placement (FIRST in context)
- [ ] Bayesian procedure selection (UCB)
- [ ] Strategic ordering (best at edges)
- [ ] Progressive history compression
- [ ] edit_file tool

**Benchmark:** Teach 5 procedures. Trigger them in new sessions. Measure match + execution accuracy. Compare retrieval precision: vector-only vs hybrid.

### Phase 3: Documents + TUI + Reranker Training (Weeks 5-6)

Learn from books. Full visibility. Start learning from feedback.

- [ ] Document ingestion pipeline (chunk → summarize → embed → store)
- [ ] Hierarchical retrieval (doc summary → section → chunk)
- [ ] `ingest` tool
- [ ] web_search tool (DuckDuckGo + store results as memories)
- [ ] TUI with all debug panels (textual)
- [ ] Retrieval feedback logging
- [ ] Feature extraction for reranker training data
- [ ] Background consolidation (merge, resolve, strengthen, decay)

**Benchmark:** Ingest a Python book. Ask 20 questions. Measure answer quality vs no-book baseline.

### Phase 4: Learned Reranker + Benchmarks (Weeks 7-8)

Memory retrieval improves from usage data.

- [ ] Logistic regression reranker (train on accumulated feedback)
- [ ] A/B test: heuristic vs trained reranker
- [ ] Usefulness score propagation on memories
- [ ] Task decomposer for multi-step tasks exceeding 32K
- [ ] Model swap configuration
- [ ] Full benchmark suite: 4B raw vs 4B+memory vs Claude Sonnet

**Benchmark:** Reranker vs heuristic on precision@5. 4B+memory vs Claude on procedural tasks, recall tasks, multi-session tasks.

### Phase 5 (Future): RAFT Fine-Tuning

After accumulating ~1000 training examples from interaction data, RAFT fine-tune the Qwen 4B model to be exceptional at using retrieved context. This is the highest-impact technique from research (30-76% improvement) but requires enough data first.

---

## Benchmark Suite

The agent is a research instrument. These benchmarks prove or disprove the thesis.

### Recall Test

```python
# Day 1: Tell the agent facts
facts = [
    "My domain is nhproject.org",
    "DNS is managed in Azure",
    "I prefer ed25519 SSH keys",
    "My Python projects use venvs in .venv/",
    "The production server is at 192.168.1.100",
    "I use Tailscale for VPN",
    "My email is rei@gjaci.com",
    "Nginx config is at /etc/nginx/sites-enabled/",
    "I prefer concise answers",
    "The deploy script is at ~/scripts/deploy.sh"
]

# Day 3: Ask about each fact indirectly
questions = [
    "Where is DNS for nhproject.org managed?",
    "What type of SSH keys should I generate?",
    "Where are my Python virtualenvs?",
    "What's the IP of the production server?",
    # ... etc
]

# Measure: % correctly recalled
```

### Procedure Test

```python
# Teach procedures, then trigger them
teach_data = [
    ("deploy", "git push && ssh prod ..."),
    ("debug_import", "check python, check venv, check pip ..."),
    ("setup_ssh", "keygen, copy, test, harden ..."),
]

# Later, trigger with natural language
triggers = [
    "Deploy the app",
    "I'm getting ModuleNotFoundError",
    "Set up keys for the new server",
]

# Measure: % correctly matched AND executed
```

### Versus Baseline

```python
# Same questions, three conditions:
# 1. Qwen 4B raw (no memory)
# 2. Qwen 4B + memory system (this agent)
# 3. Claude Sonnet (no memory, but much smarter)

# Task types:
# A. Procedural (deploy, setup, debug) — memory advantage
# B. Recall (facts about user's environment) — memory advantage
# C. General reasoning (explain concept) — model advantage
# D. Multi-session continuity — memory advantage

# Measure: response quality rated 1-5 by user
```

---

## Success Metrics

| What                                    | Target                      | When         |
| --------------------------------------- | --------------------------- | ------------ |
| Fact recall across sessions             | > 90% after 1 week          | Phase 1 exit |
| Procedure trigger accuracy              | > 80% match rate            | Phase 2 exit |
| Retrieval precision@5                   | > 75% relevant              | Phase 2 exit |
| 4B+memory vs 4B raw on procedural tasks | +40% quality                | Phase 4      |
| 4B+memory vs Claude on recall tasks     | Comparable or better        | Phase 4      |
| 4B+memory vs Claude on procedural tasks | Within 20% quality          | Phase 4      |
| Trained reranker vs heuristic           | +10% precision@5            | Phase 4      |
| Response latency                        | < 5s typical, < 15s complex | Phase 2      |
| Memory extraction accuracy              | > 70% of facts correct      | Phase 1 exit |
| Token budget discipline                 | Never exceeds 32K           | Always       |

---

## Technical Decisions and Rationale

| Decision                                               | Choice                                                                                                    | Why                                                                                       |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| SQLite over Postgres/Chroma/Qdrant                     | Single file, zero deps, SQL for structured queries, sqlite-vec for vectors, FTS5 for BM25 — all in one DB | Simplest possible setup. No server to manage. Backup = copy one file.                     |
| Gemini embedding-001 over local model                  | Best MTEB scores, free tier (10M tokens/min), 768 dims after truncation                                   | Quality > independence. Fallback: queue when offline.                                     |
| No LangChain/LangGraph                                 | Custom code for agent loop, retrieval, context assembly                                                   | Full control, no abstraction leaks, debuggable, adapted to 4B model needs                 |
| Aggressive extraction + aggressive retrieval filtering | Store everything, but surface only the best 3-5                                                           | Broad coverage + high precision. Better to have it and not need it.                       |
| Procedures FIRST in context                            | Before memories, after system prompt                                                                      | Structures the model's thinking. Most impactful context placement.                        |
| Heuristic before ML reranker                           | Start simple, collect data, upgrade when data justifies                                                   | Avoids premature optimization. Heuristics work well enough to collect good training data. |
| 4B nothink as starting model                           | Fastest, cheapest, hardest test of the thesis                                                             | If memory makes 4B work, it'll make any model better. Upgrade path via config.            |

---

## Open Questions

1. **Gemini offline:** Queue embedding requests? Or local nomic-embed-text fallback?
2. **Auto-procedure threshold:** Only extract from interactions with 3+ tool calls? Or lower?
3. **Cross-encoder reranker:** Pre-trained ms-marco-MiniLM-L6 or fine-tune on our data?
4. **Correction detection:** How to reliably detect when user corrects the agent (vs just continuing conversation)?
5. **Memory export:** Git-sync the SQLite file between machines?
6. **RAFT fine-tuning:** When enough data (1000+ examples), fine-tune Qwen 4B on accumulated (query, context, answer) triples?
7. **Cognee integration:** Use Cognee library for KG, or implement patterns ourselves?
