"""Brain Agent v2 — Prompt Registry.

Centralizes all LLM prompt templates used across the agent.  This makes it
easier to iterate on prompts, A/B test variants, and maintain consistency.

Usage::

    from core.prompts import PromptRegistry

    registry = PromptRegistry()
    prompt = registry.get("fact_extraction", user_msg="...", agent_msg="...")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

# Memory Writer prompts
FACT_EXTRACTION_PROMPT = """You are a fact extractor. Your job is to find EVERY specific claim about the user in this exchange.

RULES:
- If the user says their name, location, job, tools, preferences, or anything specific → extract it
- When in doubt, EXTRACT IT. Missing a fact is worse than extracting a redundant one.
- Each fact should be one atomic statement.

User: {user_msg}
Assistant: {agent_msg}

Return a JSON array of facts. Format:
[{{"content": "User [verb] ...", "category": "fact|preference|observation|correction|knowledge|rule", "importance": 0.1-1.0}}]

Categories:
- fact: specific personal info (name, location, job)
- preference: likes, dislikes, tool choices
- observation: things noticed about user's environment or behavior
- correction: corrects a previous wrong assumption
- knowledge: domain knowledge or technical information
- rule: learned heuristics or guidelines ("always do X when Y")

Example output for "I'm Sarah and I use Neovim":
[{{"content": "User's name is Sarah", "category": "fact", "importance": 0.9}}, {{"content": "User uses Neovim", "category": "preference", "importance": 0.7}}]"""

GRAPH_EXTRACTION_PROMPT = """Identify entities and relationships in this exchange.
Only extract SPECIFIC, named entities — not generic words.

User: {user_msg}
Assistant: {agent_msg}

Return JSON:
{{"entities": [
    {{"name": "...", "type": "person|tool|project|concept|service|config|language|file", "description": "one line description"}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "uses|prefers|part_of|depends_on|configured_with|works_with", "detail": "..."}}
  ]
}}

Examples:
- Entity: {{"name": "Tailscale", "type": "service", "description": "VPN mesh network service"}}
- Relation: {{"source": "User", "target": "ed25519", "type": "prefers", "detail": "for SSH authentication"}}"""

PROCEDURE_EXTRACTION_PROMPT = """A successful multi-step operation just completed. Extract a reusable procedure.

User request: {user_msg}
Agent response: {agent_msg}
Tools used: {tool_summary}

Return JSON describing the procedure:
{{
  "name": "short_snake_case_name",
  "description": "One sentence describing what this procedure accomplishes.",
  "trigger_pattern": "Natural language pattern that would trigger this procedure",
  "preconditions": ["condition1", "condition2"],
  "steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "warnings": ["Warning or gotcha to watch out for"],
  "context": "Any important contextual notes"
}}

Focus on the GENERAL pattern, not the specific values used this time.
If no clear reusable procedure exists, return {{}}."""

# Dream Engine prompts
ABSTRACTION_PROMPT = """You are a memory analyst. Given a cluster of related facts, create a single higher-level insight that summarises the key information.

Facts:
{facts}

Return a JSON object:
{{"insight": "A concise higher-level summary that captures the essence of these facts", "importance": 0.1-1.0}}

Rules:
- The insight should be MORE GENERAL than the individual facts
- It should capture the common theme or pattern
- Importance should reflect how useful this insight is"""

CONTRADICTION_PROMPT = """Analyse these two memories for contradictions:

Memory A: {memory_a}
Memory B: {memory_b}

Return a JSON object:
{{"contradicts": true/false, "resolution": "keep_a"|"keep_b"|"merge", "merged_content": "merged fact if resolution is merge, else empty string", "reasoning": "brief explanation"}}

Rules:
- Only mark as contradicting if they truly conflict (not just different aspects of the same topic)
- If one is more recent or specific, prefer it
- If they can be combined, merge them"""

PATTERN_PROMPT = """Analyse these recent memories and identify patterns, themes, or recurring topics:

{memories}

Return a JSON array of patterns found:
[{{"pattern": "description of the pattern", "importance": 0.1-1.0, "evidence_count": N}}]

Rules:
- Only report genuine patterns (appearing 2+ times)
- Rank by importance and frequency
- Be specific, not generic"""

CONNECTION_PROMPT = """Given this entity and its current relations, suggest missing connections:

Entity: {entity_name} (type: {entity_type})
Current relations: {relations}
Available entities: {other_entities}

Return a JSON array of suggested relations:
[{{"source": "entity_name", "target": "other_entity_name", "type": "uses|prefers|part_of|depends_on|works_with", "confidence": 0.1-1.0, "reasoning": "why this connection exists"}}]

Rules:
- Only suggest confident connections (>0.5)
- Base suggestions on the context, not guesses
- Limit to 3 most confident suggestions"""

QUESTION_PROMPT = """Based on these memories and knowledge gaps, generate questions that would help fill gaps:

Memories:
{memories}

Return a JSON array of questions:
[{{"question": "What is ...?", "importance": 0.1-1.0, "topic": "related topic"}}]

Rules:
- Questions should be naturally answerable from future conversations
- Focus on practical gaps (things that would help serve the user better)
- Limit to 5 most important questions"""

# Reasoning Engine prompts
GAP_ANALYSIS_PROMPT = """You are analysing a knowledge base about a user. Given these memories about the topic "{topic}", identify knowledge gaps.

Memories:
{memories}

Return a JSON object:
{{"gaps": ["specific gap 1", "specific gap 2"], "questions": ["question to fill gap 1", "question to fill gap 2"], "importance": 0.1-1.0}}

Rules:
- Focus on PRACTICAL gaps (things that would help the agent serve the user better)
- Questions should be naturally answerable from future conversations
- Limit to 3 most important gaps"""

CROSS_DOMAIN_PROMPT = """Given these two sets of memories about different topics, identify connections:

Topic A ({topic_a}):
{memories_a}

Topic B ({topic_b}):
{memories_b}

Return a JSON object:
{{"connections": [{{"insight": "how A relates to B", "importance": 0.1-1.0}}], "new_relations": [{{"source": "entity_a", "target": "entity_b", "type": "uses|depends_on|works_with|part_of", "reasoning": "why"}}]}}

Rules:
- Only report genuine, meaningful connections
- Be specific, not generic"""

RULE_INFERENCE_PROMPT = """Analyse these observations and derive general rules or heuristics:

Observations:
{observations}

Return a JSON array of rules:
[{{"rule": "When X, then Y", "confidence": 0.1-1.0, "evidence_count": N, "category": "rule"}}]

Rules:
- Only derive rules with strong evidence (2+ supporting observations)
- Rules should be actionable and specific
- Limit to 3 most confident rules"""

CONTRADICTION_CHECK_PROMPT = """Check these memories for inconsistencies or contradictions:

{memories}

Return a JSON object:
{{"contradictions": [{{"memory_a": "content of first", "memory_b": "content of second", "explanation": "why they contradict", "resolution": "which is more likely correct and why"}}]}}

Rules:
- Only flag genuine contradictions (not just different aspects of the same topic)
- Prefer newer, more specific, or higher-confidence memories
- Explain your reasoning clearly"""

PROCEDURE_IMPROVEMENT_PROMPT = """Analyse this procedure and suggest improvements:

Procedure: {procedure_name}
Description: {procedure_description}
Steps: {steps}
Success rate: {success_rate}%

Return a JSON object:
{{"suggestions": ["suggestion 1", "suggestion 2"], "missing_steps": ["step that should be added"], "warnings": ["things to watch out for"]}}

Rules:
- Focus on practical improvements
- Suggest missing error handling or verification steps
- Limit to 3 most important suggestions"""

QUESTION_ANSWERING_PROMPT = """Try to answer this question using the available memories:

Question: {question}

Available memories:
{memories}

Return a JSON object:
{{"answer": "best answer based on memories", "confidence": 0.1-1.0, "sources": ["memory contents used"], "gaps": ["what's still unknown"]}}

Rules:
- Only answer if you have sufficient information
- Be honest about uncertainty
- Cite specific memories when making claims"""

# Conversation summarization prompt
SESSION_SUMMARY_PROMPT = """Summarize this conversation in 2-3 sentences.
Focus on: decisions made, facts learned, preferences expressed, and unresolved questions. Be concise.

Conversation:
{conversation}"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """A single prompt template with metadata.

    Attributes:
        name:         Unique identifier for the prompt.
        template:     The prompt template string with {placeholders}.
        version:      Version number for tracking changes.
        description:  Human-readable description of what the prompt does.
        category:     Category label (e.g. "extraction", "reasoning", "dream").
    """
    name: str
    template: str
    version: str = "1.0.0"
    description: str = ""
    category: str = "general"


class PromptRegistry:
    """Central registry for all LLM prompt templates.

    Provides:
    - Centralized prompt management
    - Version tracking for prompt changes
    - Easy A/B testing support
    - Validation of required placeholders

    Example::

        registry = PromptRegistry()
        prompt = registry.format("fact_extraction", user_msg="...", agent_msg="...")
    """

    def __init__(self) -> None:
        self._prompts: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default prompt templates."""
        defaults = [
            PromptTemplate("fact_extraction", FACT_EXTRACTION_PROMPT,
                          description="Extract facts from user-agent exchanges",
                          category="extraction"),
            PromptTemplate("graph_extraction", GRAPH_EXTRACTION_PROMPT,
                          description="Extract entities and relationships",
                          category="extraction"),
            PromptTemplate("procedure_extraction", PROCEDURE_EXTRACTION_PROMPT,
                          description="Extract reusable procedures",
                          category="extraction"),
            PromptTemplate("abstraction", ABSTRACTION_PROMPT,
                          description="Create higher-level insights from fact clusters",
                          category="dream"),
            PromptTemplate("contradiction", CONTRADICTION_PROMPT,
                          description="Detect and resolve contradictions",
                          category="dream"),
            PromptTemplate("pattern", PATTERN_PROMPT,
                          description="Identify patterns in memories",
                          category="dream"),
            PromptTemplate("connection", CONNECTION_PROMPT,
                          description="Suggest missing KG connections",
                          category="dream"),
            PromptTemplate("question", QUESTION_PROMPT,
                          description="Generate questions for knowledge gaps",
                          category="dream"),
            PromptTemplate("gap_analysis", GAP_ANALYSIS_PROMPT,
                          description="Identify knowledge gaps",
                          category="reasoning"),
            PromptTemplate("cross_domain", CROSS_DOMAIN_PROMPT,
                          description="Find cross-domain connections",
                          category="reasoning"),
            PromptTemplate("rule_inference", RULE_INFERENCE_PROMPT,
                          description="Derive general rules from observations",
                          category="reasoning"),
            PromptTemplate("contradiction_check", CONTRADICTION_CHECK_PROMPT,
                          description="Check for memory contradictions",
                          category="reasoning"),
            PromptTemplate("procedure_improvement", PROCEDURE_IMPROVEMENT_PROMPT,
                          description="Suggest procedure improvements",
                          category="reasoning"),
            PromptTemplate("question_answering", QUESTION_ANSWERING_PROMPT,
                          description="Answer questions from memories",
                          category="reasoning"),
            PromptTemplate("session_summary", SESSION_SUMMARY_PROMPT,
                          description="Summarize old conversation sessions",
                          category="consolidation"),
        ]
        for prompt in defaults:
            self.register(prompt)

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template.

        Args:
            template: The prompt template to register.
        """
        self._prompts[template.name] = template
        logger.debug("Registered prompt: %s (v%s)", template.name, template.version)

    def get(self, name: str) -> PromptTemplate | None:
        """Get a prompt template by name.

        Args:
            name: The prompt name.

        Returns:
            The PromptTemplate, or None if not found.
        """
        return self._prompts.get(name)

    def format(self, name: str, **kwargs: Any) -> str:
        """Get and format a prompt template.

        Args:
            name: The prompt name.
            **kwargs: Values to substitute into the template.

        Returns:
            The formatted prompt string.

        Raises:
            KeyError: If the prompt name is not found.
        """
        template = self._prompts.get(name)
        if template is None:
            raise KeyError(f"Prompt '{name}' not found in registry")
        return template.template.format(**kwargs)

    def list_prompts(self, category: str | None = None) -> list[PromptTemplate]:
        """List all registered prompts, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of PromptTemplate objects.
        """
        if category:
            return [p for p in self._prompts.values() if p.category == category]
        return list(self._prompts.values())

    def update(self, name: str, template: str, version: str = "1.0.0") -> None:
        """Update an existing prompt template.

        Args:
            name: The prompt name to update.
            template: The new template string.
            version: New version number.

        Raises:
            KeyError: If the prompt name is not found.
        """
        existing = self._prompts.get(name)
        if existing is None:
            raise KeyError(f"Prompt '{name}' not found in registry")
        existing.template = template
        existing.version = version
        logger.info("Updated prompt: %s to v%s", name, version)
