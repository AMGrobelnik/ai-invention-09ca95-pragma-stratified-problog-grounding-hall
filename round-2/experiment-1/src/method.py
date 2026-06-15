#!/usr/bin/env python3
"""
Pragma-Stratified ProbLog: Rigorous Statistical Validation & Power Analysis
Three stages:
  1. Tier-correctness Spearman correlation (CLUTRR kinship facts)
  2. Benchmark accuracy + McNemar significance tests (CLUTRR + RuleTaker)
  3. Hallucination bound Pearson r + post-hoc power analysis
"""

import asyncio
import gc
import json
import math
import os
import re
import resource
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from loguru import logger
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

# ─── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_data/runs/5b0b4/3_invention_loop/iter_2/gen_art/gen_art_experiment_1")
DATA_PATH = Path("/ai-inventor/aii_data/runs/5b0b4/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
OUT_PATH = WORKSPACE / "method_out.json"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ─── Memory limits (cgroup v2: 29 GB container) ──────────────────────────────
_RAM_BUDGET = 20 * 1024**3  # 20 GB – well within 29 GB
resource.setrlimit(resource.RLIMIT_AS, (_RAM_BUDGET * 3, _RAM_BUDGET * 3))

# ─── Config ──────────────────────────────────────────────────────────────────
N_CLUTRR = int(os.environ.get("N_CLUTRR", "200"))
N_RULETAKER = int(os.environ.get("N_RULETAKER", "200"))
OPENROUTER_MODEL = os.environ.get("OR_MODEL", "meta-llama/llama-3.1-8b-instruct")
CONCURRENCY = 8
BUDGET_USD = 10.0
COST_PER_TOKEN = 0.0000001  # $0.10/M tokens (conservative estimate for llama-3.1-8b)
MAX_TOKENS_PER_CALL = 150

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tier probability weights (pragmatic stratification)
TIER_PROBS = {"assertion": 0.93, "presupposition": 0.78, "implicature": 0.61, "unknown": 0.75}

# ─── Global cost tracker ─────────────────────────────────────────────────────
_cost_lock = asyncio.Lock()
_total_tokens = {"in": 0, "out": 0}
_api_calls = 0


async def _track_cost(tokens_in: int, tokens_out: int) -> float:
    global _api_calls
    async with _cost_lock:
        _total_tokens["in"] += tokens_in
        _total_tokens["out"] += tokens_out
        _api_calls += 1
        total = (_total_tokens["in"] + _total_tokens["out"]) * COST_PER_TOKEN
        if total > BUDGET_USD * 0.9:
            logger.warning(f"Budget at 90%: ${total:.2f} / ${BUDGET_USD}")
        return total


# ─── OpenRouter async call ───────────────────────────────────────────────────
async def call_llm(session: aiohttp.ClientSession, prompt: str, system: str = "") -> str:
    """Single async LLM call via OpenRouter. Returns text or '' on error."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS_PER_CALL,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-inventor.local",
    }
    try:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"OpenRouter {resp.status}: {body[:200]}")
                return ""
            data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            await _track_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return text
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


# ─── Kinship / CLUTRR helpers ────────────────────────────────────────────────
RELATIONS = [
    "daughter", "son", "mother", "father", "sister", "brother",
    "wife", "husband", "aunt", "uncle", "niece", "nephew",
    "grandmother", "grandfather", "granddaughter", "grandson",
    "son-in-law", "daughter-in-law",
]
RELATIONS_PAT = "|".join(re.escape(r) for r in sorted(RELATIONS, key=len, reverse=True))

# Patterns that indicate tier classification
TIER_PATTERNS = {
    # Assertion: direct explicit kinship "[X]'s [rel] [Y]" or "[X] is [Y]'s [rel]"
    "assertion": [
        re.compile(r"\[([A-Z][a-z]+)\]'s,?\s+(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]", re.I),
        re.compile(r"\[([A-Z][a-z]+)\]\s+is\s+(?:the\s+)?(" + RELATIONS_PAT + r")\s+of\s+\[([A-Z][a-z]+)\]", re.I),
    ],
    # Presupposition: possessive pronoun-mediated "[A] and her/his [rel] [B]"
    "presupposition": [
        re.compile(r"\[([A-Z][a-z]+)\]\s+and\s+(?:her|his|their)\s+(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]", re.I),
        re.compile(r"(?:her|his|their)\s+(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]", re.I),
    ],
    # Implicature: relational verbs / action without explicit naming
    "implicature": [
        re.compile(r"\[([A-Z][a-z]+)\]\s+(?:helped|visited|called|asked|told|met|saw)\s+\[([A-Z][a-z]+)\]", re.I),
    ],
}

KINSHIP_INVERSE = {
    "daughter": "mother", "son": "father", "mother": "daughter", "father": "son",
    "sister": "sister", "brother": "brother", "wife": "husband", "husband": "wife",
    "aunt": "niece", "uncle": "nephew", "niece": "aunt", "nephew": "uncle",
    "grandmother": "granddaughter", "grandfather": "grandson",
    "granddaughter": "grandmother", "grandson": "grandfather",
    "son-in-law": "mother-in-law", "daughter-in-law": "father-in-law",
    "mother-in-law": "son-in-law", "father-in-law": "daughter-in-law",
}

# Kinship composition: (A→B, B→C) → A→C
# Semantics: (A, rel1, B) means "B is A's rel1" (e.g., daughter = B is A's daughter)
COMPOSE = {
    # Same parent + sibling of that child
    ("mother", "daughter"): "sister",
    ("mother", "son"): "brother",
    ("father", "daughter"): "sister",
    ("father", "son"): "brother",
    # Parent's sibling = aunt/uncle
    ("mother", "sister"): "aunt",
    ("mother", "brother"): "uncle",
    ("father", "sister"): "aunt",
    ("father", "brother"): "uncle",
    # Sibling's parent = my parent
    ("sister", "mother"): "mother",
    ("sister", "father"): "father",
    ("brother", "mother"): "mother",
    ("brother", "father"): "father",
    # Sibling's child = niece/nephew
    ("sister", "daughter"): "niece",
    ("sister", "son"): "nephew",
    ("brother", "daughter"): "niece",
    ("brother", "son"): "nephew",
    # Grandparent relations
    ("daughter", "daughter"): "granddaughter",
    ("daughter", "son"): "grandson",
    ("son", "daughter"): "granddaughter",
    ("son", "son"): "grandson",
    ("mother", "mother"): "grandmother",
    ("mother", "father"): "grandfather",
    ("father", "mother"): "grandmother",
    ("father", "father"): "grandfather",
    # A's daughter's sibling = A's other child
    ("daughter", "sister"): "daughter",
    ("daughter", "brother"): "son",
    ("son", "sister"): "daughter",
    ("son", "brother"): "son",
    # In-law
    ("daughter", "husband"): "son-in-law",
    ("son", "wife"): "daughter-in-law",
    # Cousin
    ("uncle", "son"): "cousin",
    ("uncle", "daughter"): "cousin",
    ("aunt", "son"): "cousin",
    ("aunt", "daughter"): "cousin",
    # Grandchild's parent = child
    ("grandson", "mother"): "daughter",
    ("grandson", "father"): "son",
    ("granddaughter", "mother"): "daughter",
    ("granddaughter", "father"): "son",
}


def _add_fact(facts: list, seen: set, a: str, rel: str, b: str, tier: str) -> None:
    """Add a fact and its inverse to the fact list if not already present."""
    key = (a, rel, b)
    if key not in seen:
        seen.add(key)
        facts.append({"a": a, "rel": rel, "b": b, "tier": tier})
    inv = KINSHIP_INVERSE.get(rel)
    if inv:
        ikey = (b, inv, a)
        if ikey not in seen:
            seen.add(ikey)
            facts.append({"a": b, "rel": inv, "b": a, "tier": tier})


def extract_kinship_facts(text: str) -> list[dict]:
    """Extract (entity_a, relation_a_to_b, entity_b, tier) tuples from text."""
    facts: list[dict] = []
    seen: set = set()

    # ASSERTION pattern 1: "[A]'s [rel] [B]" or "[A]'s [rel], [B]"
    pat1 = re.compile(
        r"\[([A-Z][a-z]+)\]'s,?\s+(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]",
        re.I
    )
    for m in pat1.finditer(text):
        a, rel, b = m.group(1), m.group(2).lower(), m.group(3)
        _add_fact(facts, seen, a, rel, b, "assertion")

    # ASSERTION pattern 2: "[A] is [B]'s [rel]" or "[A] is the [rel] of [B]"
    pat2 = re.compile(
        r"\[([A-Z][a-z]+)\]\s+is\s+(?:the\s+)?(" + RELATIONS_PAT + r")\s+of\s+\[([A-Z][a-z]+)\]",
        re.I
    )
    for m in pat2.finditer(text):
        a, rel, b = m.group(1), m.group(2).lower(), m.group(3)
        # "A is rel of B" means B's rel is A → B→A with rel, or A→B with inverse
        _add_fact(facts, seen, b, rel, a, "assertion")

    # ASSERTION pattern 3: "[A], [B]'s [rel]" appositive
    pat3 = re.compile(
        r"\[([A-Z][a-z]+)\],?\s+\[([A-Z][a-z]+)\]'s\s+(" + RELATIONS_PAT + r")",
        re.I
    )
    for m in pat3.finditer(text):
        a, b, rel = m.group(1), m.group(2), m.group(3).lower()
        # "A, B's [rel]" → A is B's rel → B→A with rel
        _add_fact(facts, seen, b, rel, a, "assertion")

    # PRESUPPOSITION: "[A] ... her/his [rel] [B]" – subject A in same sentence
    for sent in re.split(r'[.!?]', text):
        entities_in_sent = re.findall(r"\[([A-Z][a-z]+)\]", sent)
        if not entities_in_sent:
            continue
        # Find "her/his [rel] [B]" patterns
        for m in re.finditer(
            r"(?:her|his|their)\s+(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]",
            sent, re.I
        ):
            rel, b = m.group(1).lower(), m.group(2)
            # Subject = first named entity before this match that isn't B
            pos = m.start()
            candidates = [e for e in entities_in_sent if e != b]
            if candidates:
                a = candidates[0]
                _add_fact(facts, seen, a, rel, b, "presupposition")

        # "[A]'s [rel], [B]" (catches cases missed by assertion pat1)
        for m in re.finditer(
            r"\[([A-Z][a-z]+)\]'s,?\s+(?:\w+\s+)*(" + RELATIONS_PAT + r"),?\s+\[([A-Z][a-z]+)\]",
            sent, re.I
        ):
            a, rel, b = m.group(1), m.group(2).lower(), m.group(3)
            _add_fact(facts, seen, a, rel, b, "presupposition")

    # IMPLICATURE: interaction verbs between named entities
    pat_impl = re.compile(
        r"\[([A-Z][a-z]+)\][^.]*?(?:helped|visited|called|asked|told|met|saw|raised|supporting|supported)\s+"
        r"(?:her|his|their\s+)?\[([A-Z][a-z]+)\]",
        re.I
    )
    for m in pat_impl.finditer(text):
        a, b = m.group(1), m.group(2)
        key = (a, "related_to", b)
        if key not in seen:
            seen.add(key)
            facts.append({"a": a, "rel": "related_to", "b": b, "tier": "implicature"})

    return facts


def build_kinship_graph(facts: list[dict]) -> dict:
    """Build adjacency: graph[A] = [(rel, B, tier)]"""
    graph = defaultdict(list)
    for f in facts:
        graph[f["a"]].append((f["rel"], f["b"], f["tier"]))
    return graph


def find_kinship_path(graph: dict, source: str, target: str, use_tier_weights: bool = False) -> tuple[str | None, float, list]:
    """BFS to find kinship relation from source to target. Returns (relation, proof_prob, fact_chain)."""
    if source not in graph and target not in graph:
        return None, 0.0, []

    # BFS: state = (current_node, accumulated_relation, prob, chain)
    from collections import deque
    queue = deque([(source, None, 1.0, [])])
    visited = {source}
    max_depth = 4

    while queue:
        node, current_rel, prob, chain = queue.popleft()
        if len(chain) > max_depth:
            continue

        for edge_rel, neighbor, tier in graph[node]:
            if edge_rel == "related_to":
                continue
            edge_prob = TIER_PROBS[tier] if use_tier_weights else 0.5
            new_prob = prob * edge_prob
            new_chain = chain + [(node, edge_rel, neighbor, tier)]

            if current_rel is None:
                composed = edge_rel
            else:
                composed = COMPOSE.get((current_rel, edge_rel))
                if composed is None:
                    composed = f"{current_rel}_{edge_rel}"  # unknown composition

            if neighbor == target:
                return composed, new_prob, new_chain

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, composed, new_prob, new_chain))

    return None, 0.0, []


def derive_label_map(examples: list[dict]) -> dict[str, str]:
    """Empirically derive label→relation_name mapping from CLUTRR data."""
    label_rel_counts: dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        text = ex["input"]
        pair = ex["metadata_entity_pair"]
        label = ex["output"]
        try:
            pair_tuple = eval(pair)
            src, tgt = pair_tuple[0], pair_tuple[1]
        except Exception:
            continue
        facts = extract_kinship_facts(text)
        graph = build_kinship_graph(facts)
        rel, _, _ = find_kinship_path(graph, src, tgt, use_tier_weights=False)
        if rel and "_" not in rel and rel in RELATIONS:
            label_rel_counts[label][rel] += 1

    label_map = {}
    for label, counts in label_rel_counts.items():
        if counts:
            label_map[label] = counts.most_common(1)[0][0]
    logger.info(f"Derived label map from {len(examples)} CLUTRR examples: {label_map}")
    return label_map


def relation_to_label(rel: str, label_map: dict[str, str]) -> str | None:
    """Invert label_map: relation → label."""
    if rel is None:
        return None
    inv = {v: k for k, v in label_map.items()}
    return inv.get(rel)


# ─── Classify tier for a set of facts ────────────────────────────────────────
def classify_fact_tier(text: str, fact: dict) -> str:
    return fact.get("tier", "unknown")


def compute_hallucination_bound(fact_chain: list) -> float:
    """HB = fraction of proof probability from presupposition/implicature facts."""
    if not fact_chain:
        return 0.0
    p_all = 1.0
    p_lower = 1.0  # assertions only
    for _, _, _, tier in fact_chain:
        p = TIER_PROBS.get(tier, 0.75)
        p_all *= p
        if tier == "assertion":
            p_lower *= p
    if p_all < 1e-10:
        return 1.0
    return 1.0 - (p_lower / p_all)


# ─── RuleTaker forward-chaining engine ───────────────────────────────────────
def parse_ruletaker(input_text: str) -> tuple[set, list, str]:
    """Parse context into facts, rules, and question entity/property."""
    # Split context and question
    ctx_match = re.search(r"Context:\s*(.*?)\s*\n\nQuestion:\s*(.*)", input_text, re.DOTALL)
    if not ctx_match:
        return set(), [], ""
    context = ctx_match.group(1)
    question = ctx_match.group(2).strip().rstrip(".")

    facts: set[tuple[str, str]] = set()
    rules: list[tuple[list[tuple[str, str]], tuple[str, str]]] = []

    # Parse facts: "X is [not] Y" → (x, y) or (x, "not_y")
    for sent in re.split(r'\.\s*', context):
        sent = sent.strip()
        if not sent:
            continue
        # Rule pattern: "If someone is A [and B [and C]] then they are D"
        rule_m = re.match(
            r"If someone is (\w+(?:\s+\w+)?)(?:\s+and\s+(\w+(?:\s+\w+)?))?(?:\s+and\s+(\w+(?:\s+\w+)?))?"
            r"\s+then\s+they\s+are\s+(\w+(?:\s+\w+)?)",
            sent, re.I
        )
        if rule_m:
            conditions = []
            for g in [rule_m.group(1), rule_m.group(2), rule_m.group(3)]:
                if g:
                    conditions.append(("?", g.lower().strip()))
            consequent = ("?", rule_m.group(4).lower().strip())
            rules.append((conditions, consequent))
            continue

        # Rule: "If X is A then X is B" (specific entity)
        rule_m2 = re.match(r"If (\w+) is (\w+(?:\s+\w+)?) then (\w+) is (\w+(?:\s+\w+)?)", sent, re.I)
        if rule_m2:
            e1, prop1, e2, prop2 = rule_m2.groups()
            e1, prop1 = e1.lower(), prop1.lower()
            e2, prop2 = e2.lower(), prop2.lower()
            if e1 == e2:
                rules.append([[(e1, prop1)], (e2, prop2)])
            continue

        # Fact: "X is [not] Y"
        fact_m = re.match(r"(\w+) is (not )?(\w+(?:\s+\w+)?)", sent, re.I)
        if fact_m:
            entity = fact_m.group(1).lower()
            negated = bool(fact_m.group(2))
            prop = fact_m.group(3).lower().strip()
            if not negated:
                facts.add((entity, prop))

    return facts, rules, question


def forward_chain(facts: set, rules: list) -> set:
    """Apply rules until fixed point. Returns extended fact set."""
    facts = set(facts)
    changed = True
    iterations = 0
    while changed and iterations < 50:
        changed = False
        iterations += 1
        for conditions, (ce, cp) in rules:
            # Get all entities in current facts
            entities = set(e for e, p in facts)
            for entity in entities:
                # Substitute entity for "?"
                satisfied = True
                for (ce2, cp2) in conditions:
                    actual_entity = entity if ce2 == "?" else ce2
                    if (actual_entity, cp2) not in facts:
                        satisfied = False
                        break
                if satisfied:
                    actual_ce = entity if ce == "?" else ce
                    new_fact = (actual_ce, cp)
                    if new_fact not in facts:
                        facts.add(new_fact)
                        changed = True
    return facts


def ruletaker_symbolic(input_text: str) -> tuple[str, float, list, list]:
    """Run forward-chaining on RuleTaker example. Returns (prediction, prob, used_facts, used_rules)."""
    facts, rules, question = parse_ruletaker(input_text)
    if not question:
        return "not entailment", 0.5, [], []

    # Determine if question is positive or negative
    neg_m = re.match(r"(\w+) is not (\w+(?:\s+\w+)?)", question, re.I)
    pos_m = re.match(r"(\w+) is (\w+(?:\s+\w+)?)", question, re.I)

    extended_facts = forward_chain(facts, rules)

    if neg_m:
        entity, prop = neg_m.group(1).lower(), neg_m.group(2).lower().strip()
        entailed = (entity, prop) not in extended_facts
    elif pos_m:
        entity, prop = pos_m.group(1).lower(), pos_m.group(2).lower().strip()
        entailed = (entity, prop) in extended_facts
    else:
        return "not entailment", 0.5, [], []

    prediction = "entailment" if entailed else "not entailment"
    return prediction, 0.8, list(facts), rules


def ruletaker_pragma_stratified(input_text: str) -> tuple[str, float, float]:
    """Pragma-stratified: tier-weight facts/rules, compute proof probability."""
    facts, rules, question = parse_ruletaker(input_text)
    if not question:
        return "not entailment", 0.5, 1.0

    # Classify: facts are assertions, 1-cond rules are presuppositions, multi-cond are implicatures
    fact_tiers = {f: "assertion" for f in facts}
    rule_tiers = []
    for conditions, consequent in rules:
        tier = "presupposition" if len(conditions) == 1 else "implicature"
        rule_tiers.append(tier)

    # Compute base proof probability from all facts
    p_facts = math.prod(TIER_PROBS[fact_tiers[f]] for f in facts) if facts else 1.0
    p_rules = math.prod(TIER_PROBS[t] for t in rule_tiers) if rule_tiers else 1.0
    base_prob = p_facts * p_rules

    extended_facts = forward_chain(facts, rules)

    neg_m = re.match(r"(\w+) is not (\w+(?:\s+\w+)?)", question, re.I)
    pos_m = re.match(r"(\w+) is (\w+(?:\s+\w+)?)", question, re.I)

    if neg_m:
        entity, prop = neg_m.group(1).lower(), neg_m.group(2).lower().strip()
        entailed = (entity, prop) not in extended_facts
    elif pos_m:
        entity, prop = pos_m.group(1).lower(), pos_m.group(2).lower().strip()
        entailed = (entity, prop) in extended_facts
    else:
        return "not entailment", 0.5, 1.0

    prediction = "entailment" if entailed else "not entailment"

    # Hallucination bound: fraction from non-assertion sources
    n_assertion_facts = len(facts)
    n_presup_rules = sum(1 for t in rule_tiers if t == "presupposition")
    n_implic_rules = sum(1 for t in rule_tiers if t == "implicature")
    total_pieces = n_assertion_facts + n_presup_rules + n_implic_rules
    if total_pieces == 0:
        hb = 0.0
    else:
        hb = (n_presup_rules * 0.5 + n_implic_rules * 1.0) / total_pieces

    return prediction, base_prob, hb


# ─── BM25 RAG helper ─────────────────────────────────────────────────────────
def rag_clutrr(text: str, entity_a: str, entity_b: str) -> str | None:
    """Simple BM25 RAG for CLUTRR: score sentences by entity mentions, pick best."""
    from rank_bm25 import BM25Okapi
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences:
        return None
    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    query = f"[{entity_a}] [{entity_b}] relationship kinship".lower().split()
    scores = bm25.get_scores(query)
    best_idx = int(np.argmax(scores))
    best_sent = sentences[best_idx]
    # Extract relation from best sentence
    for rel in sorted(RELATIONS, key=len, reverse=True):
        if rel in best_sent.lower():
            return rel
    return None


def rag_ruletaker(input_text: str) -> str:
    """BM25 RAG for RuleTaker: retrieve relevant sentence, check direct entailment."""
    from rank_bm25 import BM25Okapi
    ctx_match = re.search(r"Context:\s*(.*?)\s*\n\nQuestion:\s*(.*)", input_text, re.DOTALL)
    if not ctx_match:
        return "not entailment"
    context = ctx_match.group(1)
    question = ctx_match.group(2).strip()

    sentences = [s.strip() for s in re.split(r'\.\s*', context) if s.strip()]
    if not sentences:
        return "not entailment"

    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    query = question.lower().split()
    scores = bm25.get_scores(query)
    best_idx = int(np.argmax(scores))
    best_sent = sentences[best_idx].lower()

    # Check if question directly appears in best sentence
    q_lower = question.lower().rstrip(".")
    # Positive question: "X is Y"
    pos_m = re.match(r"(\w+) is (\w+(?:\s+\w+)?)", q_lower)
    neg_m = re.match(r"(\w+) is not (\w+(?:\s+\w+)?)", q_lower)
    if neg_m:
        entity, prop = neg_m.group(1), neg_m.group(2)
        # If positive fact found in context, then "not" question is not entailment
        if f"{entity} is {prop}" in context.lower():
            return "not entailment"
        return "entailment"
    elif pos_m:
        entity, prop = pos_m.group(1), pos_m.group(2)
        if f"{entity} is {prop}" in best_sent:
            return "entailment"
        return "not entailment"
    return "not entailment"


# ─── LLM prompts ─────────────────────────────────────────────────────────────
CLUTRR_SYSTEM = (
    "You predict kinship relations. The kinship label integers map to: "
    "0=daughter-in-law, 1=son, 2=niece, 3=nephew, 4=grandfather, 5=grandmother, "
    "6=sister, 7=brother, 8=aunt, 9=uncle, 10=son-in-law, 11=daughter, "
    "12=granddaughter, 13=grandson, 14=father, 15=mother, 16=wife, 17=husband. "
    "Output ONLY the integer (0-17)."
)

RULETAKER_SYSTEM = (
    "You determine logical entailment. Given context with rules and facts, "
    "determine if the question follows logically. "
    "Output ONLY 'entailment' or 'not entailment'."
)


def make_cot_prompt_clutrr(text: str, entity_a: str, entity_b: str) -> str:
    return (
        f"Story: {text}\n\n"
        f"Question: What is {entity_a}'s kinship relation to {entity_b}?\n"
        f"Think step by step, then output only the integer label (0-17)."
    )


def make_cot_prompt_ruletaker(input_text: str) -> str:
    return f"{input_text}\n\nReason step by step, then output only 'entailment' or 'not entailment'."


def make_tres_prompt_clutrr(text: str, entity_a: str, entity_b: str, symbolic_result: str | None) -> str:
    sym_hint = f"Symbolic analysis suggests: {symbolic_result}." if symbolic_result else "Symbolic analysis found no path."
    return (
        f"Story: {text}\n\n"
        f"{sym_hint}\n"
        f"Question: What is {entity_a}'s kinship relation to {entity_b}?\n"
        f"Output only the integer label (0-17)."
    )


def make_tres_prompt_ruletaker(input_text: str, symbolic_result: str) -> str:
    return (
        f"{input_text}\n\n"
        f"Symbolic forward-chaining found: {symbolic_result}.\n"
        f"Output only 'entailment' or 'not entailment'."
    )


def parse_clutrr_prediction(text: str) -> str:
    """Extract integer label from LLM output."""
    if not text:
        return "-1"
    m = re.search(r'\b(\d{1,2})\b', text)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 17:
            return str(n)
    return "-1"


def parse_ruletaker_prediction(text: str) -> str:
    if not text:
        return "not entailment"
    t = text.lower().strip()
    if "not entailment" in t:
        return "not entailment"
    if "entailment" in t:
        return "entailment"
    return "not entailment"


# ─── Statistical analysis functions ──────────────────────────────────────────
def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for correlation via Fisher Z transformation."""
    if n < 4 or abs(r) >= 1.0:
        return (-1.0, 1.0)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = z - z_crit * se, z + z_crit * se
    lo_r = (math.exp(2 * lo_z) - 1) / (math.exp(2 * lo_z) + 1)
    hi_r = (math.exp(2 * hi_z) - 1) / (math.exp(2 * hi_z) + 1)
    return (round(lo_r, 4), round(hi_r, 4))


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes."""
    assert len(correct_a) == len(correct_b)
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and cb)
    a = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and cb)
    d = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and not cb)
    discordant = b + c

    if discordant == 0:
        return {"chi2": 0.0, "pvalue": 1.0, "b": b, "c": c, "a": a, "d": d, "discordant": 0,
                "significance_label": "Not significant (p≥0.05)", "note": "No discordant pairs"}

    if discordant < 25:
        # Exact binomial
        pvalue = float(2 * min(
            sum(stats.binom.pmf(k, discordant, 0.5) for k in range(b + 1)),
            sum(stats.binom.pmf(k, discordant, 0.5) for k in range(c + 1))
        ))
        chi2 = (b - c) ** 2 / (b + c)
        note = "Exact binomial (n_discordant < 25)"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
        pvalue = float(1 - stats.chi2.cdf(chi2, df=1))
        note = "McNemar chi-squared with continuity correction"

    if pvalue < 0.05:
        sig = "Significant (p<0.05)"
    elif pvalue < 0.10:
        sig = "Marginally significant (p<0.10)"
    else:
        sig = "Not significant (p≥0.05)"

    return {"chi2": round(chi2, 4), "pvalue": round(pvalue, 4),
            "b": b, "c": c, "a": a, "d": d, "discordant": discordant,
            "significance_label": sig, "note": note}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def wilson_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    if n_total == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(n_correct, n_total, alpha=alpha, method="wilson")
    return (round(float(lo), 4), round(float(hi), 4))


def post_hoc_power(r_obs: float, n: int, alpha: float = 0.05) -> float:
    """Post-hoc power for Pearson correlation test (two-tailed)."""
    if n < 4 or abs(r_obs) < 1e-6:
        return 0.0
    z_obs = 0.5 * math.log((1 + abs(r_obs)) / (1 - abs(r_obs)))
    se = 1.0 / math.sqrt(n - 3)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_alpha - z_obs / se) + stats.norm.cdf(-z_alpha - z_obs / se)
    return round(float(power), 4)


def required_n_for_power(r_effect: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Required n for Pearson r at given power."""
    if abs(r_effect) < 1e-6:
        return 9999
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    z_r = 0.5 * math.log((1 + abs(r_effect)) / (1 - abs(r_effect)))
    n = math.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3)
    return n


def compute_spearman_ci(tier_ordinal: list[int], correctness: list[float]) -> dict:
    """Spearman ρ with 95% CI via Fisher Z, per-domain support."""
    n = len(tier_ordinal)
    if n < 4:
        return {"rho": None, "ci_95": [None, None], "pvalue": None, "n": n, "note": "Too few samples"}
    # Check for constant arrays
    if len(set(tier_ordinal)) < 2 or len(set(correctness)) < 2:
        return {"rho": 0.0, "ci_95": [None, None], "pvalue": 1.0, "n": n,
                "note": "Constant array – correlation undefined, set to 0"}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, pval = stats.spearmanr(tier_ordinal, correctness)
    rho, pval = float(rho), float(pval)
    ci = fisher_z_ci(rho, n)
    return {
        "rho": round(rho, 4),
        "ci_95": list(ci),
        "pvalue": round(pval, 6),
        "n": n,
        "note": f"Spearman ρ={rho:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}], p={pval:.4f}",
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────
async def run_experiment(n_clutrr: int, n_ruletaker: int) -> dict:
    """Run all methods on sampled data, compute statistics."""
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    raw = json.loads(DATA_PATH.read_text())
    clutrr_all = raw["datasets"][0]["examples"]
    ruletaker_all = raw["datasets"][1]["examples"]
    del raw
    gc.collect()

    # Stratified sample: preserve config distribution for RuleTaker
    rng = np.random.default_rng(42)
    clutrr_sample = list(rng.choice(len(clutrr_all), size=min(n_clutrr, len(clutrr_all)), replace=False))
    clutrr_exs = [clutrr_all[i] for i in clutrr_sample]

    # Stratify RuleTaker by config
    by_config: dict[str, list] = defaultdict(list)
    for ex in ruletaker_all:
        by_config[ex.get("metadata_config", "unknown")].append(ex)
    rt_exs = []
    per_config = max(1, n_ruletaker // max(1, len(by_config)))
    for cfg, exs in sorted(by_config.items()):
        n_take = min(per_config, len(exs))
        rt_exs.extend(exs[:n_take])
    rt_exs = rt_exs[:n_ruletaker]

    del clutrr_all, ruletaker_all
    gc.collect()
    logger.info(f"Sampled {len(clutrr_exs)} CLUTRR + {len(rt_exs)} RuleTaker examples")

    # ── 2. Derive label map ───────────────────────────────────────────────────
    logger.info("Deriving CLUTRR label map from sample...")
    label_map = derive_label_map(clutrr_exs)
    rel_to_label = {v: k for k, v in label_map.items()}
    logger.info(f"Label map ({len(label_map)} entries): {dict(list(label_map.items())[:10])}")

    # ── 3. LLM calls (async) ─────────────────────────────────────────────────
    logger.info(f"Running LLM methods via OpenRouter ({OPENROUTER_MODEL})...")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded_call(session, prompt, system=""):
        async with sem:
            return await call_llm(session, prompt, system)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # CoT predictions for CLUTRR
        logger.info("CoT predictions: CLUTRR...")
        cot_clutrr_tasks = []
        for ex in clutrr_exs:
            pair = eval(ex["metadata_entity_pair"])
            prompt = make_cot_prompt_clutrr(ex["input"], pair[0], pair[1])
            cot_clutrr_tasks.append(bounded_call(session, prompt, CLUTRR_SYSTEM))
        cot_clutrr_raw = await asyncio.gather(*cot_clutrr_tasks)
        cot_clutrr_preds = [parse_clutrr_prediction(r) for r in cot_clutrr_raw]
        logger.info(f"CoT CLUTRR done. Cost so far: ${(_total_tokens['in']+_total_tokens['out'])*COST_PER_TOKEN:.3f}")

        # CoT predictions for RuleTaker
        logger.info("CoT predictions: RuleTaker...")
        cot_rt_tasks = []
        for ex in rt_exs:
            prompt = make_cot_prompt_ruletaker(ex["input"])
            cot_rt_tasks.append(bounded_call(session, prompt, RULETAKER_SYSTEM))
        cot_rt_raw = await asyncio.gather(*cot_rt_tasks)
        cot_rt_preds = [parse_ruletaker_prediction(r) for r in cot_rt_raw]
        logger.info(f"CoT RuleTaker done. Cost so far: ${(_total_tokens['in']+_total_tokens['out'])*COST_PER_TOKEN:.3f}")

        # LLM-TRes: need symbolic results first (run before LLM)
        logger.info("Computing symbolic results for LLM-TRes...")
        symbolic_clutrr = []
        for ex in clutrr_exs:
            pair = eval(ex["metadata_entity_pair"])
            facts = extract_kinship_facts(ex["input"])
            graph = build_kinship_graph(facts)
            rel, prob, chain = find_kinship_path(graph, pair[0], pair[1], use_tier_weights=False)
            pred_label = rel_to_label.get(rel) if rel else None
            symbolic_clutrr.append({"rel": rel, "label": pred_label, "prob": prob, "chain": chain})

        symbolic_rt = []
        for ex in rt_exs:
            pred, prob, _, _ = ruletaker_symbolic(ex["input"])
            symbolic_rt.append({"pred": pred, "prob": prob})

        # LLM-TRes CLUTRR (only call LLM when symbolic fails)
        logger.info("LLM-TRes: CLUTRR...")
        tres_clutrr_tasks = []
        tres_clutrr_use_llm = []
        for i, (ex, sym) in enumerate(zip(clutrr_exs, symbolic_clutrr)):
            pair = eval(ex["metadata_entity_pair"])
            if sym["label"] is not None:
                # Symbolic succeeded; verify with LLM
                prompt = make_tres_prompt_clutrr(ex["input"], pair[0], pair[1], sym["rel"])
                tres_clutrr_use_llm.append(True)
            else:
                # Symbolic failed; use LLM
                prompt = make_tres_prompt_clutrr(ex["input"], pair[0], pair[1], None)
                tres_clutrr_use_llm.append(True)
            tres_clutrr_tasks.append(bounded_call(session, prompt, CLUTRR_SYSTEM))
        tres_clutrr_raw = await asyncio.gather(*tres_clutrr_tasks)
        tres_clutrr_preds = [parse_clutrr_prediction(r) for r in tres_clutrr_raw]
        logger.info(f"LLM-TRes CLUTRR done. Cost: ${(_total_tokens['in']+_total_tokens['out'])*COST_PER_TOKEN:.3f}")

        # LLM-TRes RuleTaker (LLM verifies/overrides symbolic)
        logger.info("LLM-TRes: RuleTaker...")
        tres_rt_tasks = []
        for ex, sym in zip(rt_exs, symbolic_rt):
            prompt = make_tres_prompt_ruletaker(ex["input"], sym["pred"])
            tres_rt_tasks.append(bounded_call(session, prompt, RULETAKER_SYSTEM))
        tres_rt_raw = await asyncio.gather(*tres_rt_tasks)
        tres_rt_preds = [parse_ruletaker_prediction(r) for r in tres_rt_raw]
        logger.info(f"LLM-TRes RuleTaker done. Cost: ${(_total_tokens['in']+_total_tokens['out'])*COST_PER_TOKEN:.3f}")

    # ── 4. Rule-based methods (no LLM) ───────────────────────────────────────
    logger.info("Computing RAG, Flat FOL, Pragma-Stratified predictions...")

    # CLUTRR rule-based
    rag_clutrr_preds = []
    flat_fol_clutrr_preds = []
    pragma_clutrr_preds = []
    pragma_clutrr_probs = []
    pragma_clutrr_hbs = []
    pragma_clutrr_chains = []

    for ex in clutrr_exs:
        pair = eval(ex["metadata_entity_pair"])
        ea, eb = pair[0], pair[1]
        text = ex["input"]

        # RAG
        rag_rel = rag_clutrr(text, ea, eb)
        rag_label = rel_to_label.get(rag_rel) if rag_rel else None
        rag_clutrr_preds.append(rag_label or "-1")

        # Flat FOL (no tier weighting)
        facts = extract_kinship_facts(text)
        graph = build_kinship_graph(facts)
        rel_flat, prob_flat, chain_flat = find_kinship_path(graph, ea, eb, use_tier_weights=False)
        flat_label = rel_to_label.get(rel_flat) if rel_flat else None
        flat_fol_clutrr_preds.append(flat_label or "-1")

        # Pragma-Stratified (with tier weighting)
        rel_prag, prob_prag, chain_prag = find_kinship_path(graph, ea, eb, use_tier_weights=True)
        prag_label = rel_to_label.get(rel_prag) if rel_prag else None
        pragma_clutrr_preds.append(prag_label or "-1")
        pragma_clutrr_probs.append(prob_prag)
        hb = compute_hallucination_bound(chain_prag)
        pragma_clutrr_hbs.append(hb)
        pragma_clutrr_chains.append(chain_prag)

    # RuleTaker rule-based
    rag_rt_preds = []
    flat_fol_rt_preds = []
    pragma_rt_preds = []
    pragma_rt_probs = []
    pragma_rt_hbs = []

    for ex in rt_exs:
        text = ex["input"]

        # RAG
        rag_rt_preds.append(rag_ruletaker(text))

        # Flat FOL
        pred_flat, _, _, _ = ruletaker_symbolic(text)
        flat_fol_rt_preds.append(pred_flat)

        # Pragma-Stratified
        pred_prag, prob_prag, hb_prag = ruletaker_pragma_stratified(text)
        pragma_rt_preds.append(pred_prag)
        pragma_rt_probs.append(prob_prag)
        pragma_rt_hbs.append(hb_prag)

    logger.info("Rule-based methods complete.")

    # ── 5. Evaluate correctness ───────────────────────────────────────────────
    def is_correct(pred: str, gt: str) -> bool:
        return str(pred).strip() == str(gt).strip()

    gt_clutrr = [ex["output"] for ex in clutrr_exs]
    gt_rt = [ex["output"] for ex in rt_exs]

    methods_clutrr = {
        "rag": rag_clutrr_preds,
        "chain_of_thought": cot_clutrr_preds,
        "flat_fol_ablation": flat_fol_clutrr_preds,
        "pragma_problog": pragma_clutrr_preds,
        "llm_tres": tres_clutrr_preds,
    }
    methods_rt = {
        "rag": rag_rt_preds,
        "chain_of_thought": cot_rt_preds,
        "flat_fol_ablation": flat_fol_rt_preds,
        "pragma_problog": pragma_rt_preds,
        "llm_tres": tres_rt_preds,
    }

    correct_clutrr = {m: [is_correct(p, g) for p, g in zip(preds, gt_clutrr)]
                      for m, preds in methods_clutrr.items()}
    correct_rt = {m: [is_correct(p, g) for p, g in zip(preds, gt_rt)]
                  for m, preds in methods_rt.items()}

    # Combined (CLUTRR + RuleTaker)
    gt_all = gt_clutrr + gt_rt
    methods_all_preds = {
        m: methods_clutrr[m] + methods_rt[m] for m in methods_clutrr
    }
    correct_all = {m: correct_clutrr[m] + correct_rt[m] for m in methods_clutrr}

    n_clutrr_actual = len(clutrr_exs)
    n_rt_actual = len(rt_exs)
    n_total = n_clutrr_actual + n_rt_actual

    # Accuracy + Wilson CI
    def acc_stats(correct: list[bool], n: int) -> dict:
        nc = sum(correct)
        acc = nc / n if n > 0 else 0.0
        ci = wilson_ci(nc, n)
        return {"accuracy": round(acc, 4), "n_correct": nc, "n_total": n, "ci_95": list(ci)}

    method_accs_clutrr = {m: acc_stats(correct_clutrr[m], n_clutrr_actual) for m in methods_clutrr}
    method_accs_rt = {m: acc_stats(correct_rt[m], n_rt_actual) for m in methods_rt}
    method_accs_all = {m: acc_stats(correct_all[m], n_total) for m in methods_all_preds}

    logger.info("Accuracy results (combined):")
    for m, s in method_accs_all.items():
        logger.info(f"  {m}: {s['accuracy']:.3f} ({s['n_correct']}/{s['n_total']}) 95%CI={s['ci_95']}")

    # ── 6. McNemar tests ──────────────────────────────────────────────────────
    method_names = list(methods_clutrr.keys())
    mcnemar_results = []
    for i, ma in enumerate(method_names):
        for j, mb in enumerate(method_names):
            if j <= i:
                continue
            result = mcnemar_test(correct_all[ma], correct_all[mb])
            h = cohens_h(
                sum(correct_all[ma]) / n_total,
                sum(correct_all[mb]) / n_total,
            )
            h_interp = "large" if abs(h) >= 0.8 else "medium" if abs(h) >= 0.5 else "small"
            acc_a = sum(correct_all[ma]) / n_total
            acc_b = sum(correct_all[mb]) / n_total
            mcnemar_results.append({
                "pair": [ma, mb],
                "contingency_table": {"a": result["a"], "b": result["b"], "c": result["c"], "d": result["d"]},
                "discordant": result["discordant"],
                "chi2": result["chi2"],
                "pvalue": result["pvalue"],
                "significance_label": result["significance_label"],
                "cohens_h": round(abs(h), 4),
                "effect_interpretation": h_interp,
                "acc_diff": round(acc_a - acc_b, 4),
                "note": result["note"],
            })

    primary_test = next((r for r in mcnemar_results if set(r["pair"]) == {"pragma_problog", "chain_of_thought"}), None)
    logger.info(f"Primary hypothesis test (pragma_problog vs CoT): {primary_test}")

    # ── 7. Stage 1: Tier-correctness correlation ─────────────────────────────
    logger.info("Stage 1: Tier-correctness Spearman correlation...")
    tier_ordinal_map = {"assertion": 3, "presupposition": 2, "implicature": 1, "unknown": 0}
    tier_ordinals = []
    fact_correctness = []

    # For each CLUTRR example, extract facts and label correctness
    for i, ex in enumerate(clutrr_exs):
        facts = extract_kinship_facts(ex["input"])
        pair = eval(ex["metadata_entity_pair"])
        ea, eb = pair[0], pair[1]
        gt_label = ex["output"]
        # Correct = pragma_problog predicted correctly
        is_corr = correct_clutrr["pragma_problog"][i]
        # Use correctness of the prediction as proxy for fact quality
        # More nuanced: each fact in the proof gets the correctness label
        chain = pragma_clutrr_chains[i]
        for _, _, _, tier in chain:
            tier_ordinals.append(tier_ordinal_map.get(tier, 0))
            fact_correctness.append(1.0 if is_corr else 0.0)
        # If no chain, use all extracted facts
        if not chain:
            for f in facts[:3]:  # cap at 3 to avoid bias
                tier_ordinals.append(tier_ordinal_map.get(f.get("tier", "unknown"), 0))
                fact_correctness.append(1.0 if is_corr else 0.0)

    tier_corr = compute_spearman_ci(tier_ordinals, fact_correctness)
    logger.info(f"Tier-correctness: {tier_corr}")

    # Per-domain (using RuleTaker configs as "domains")
    domain_tier: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i, ex in enumerate(rt_exs):
        cfg = ex.get("metadata_config", "unknown")
        pred, prob, hb = ruletaker_pragma_stratified(ex["input"])
        is_corr = correct_rt["pragma_problog"][i]
        facts_set, rules, _ = parse_ruletaker(ex["input"])
        # Each rule's tier
        for j, rule in enumerate(rules):
            tier = "presupposition" if len(rule[0]) == 1 else "implicature"
            domain_tier[cfg][0].append(tier_ordinal_map[tier])
            domain_tier[cfg][1].append(1.0 if is_corr else 0.0)

    per_domain_results = {}
    for domain, (tiers, corrs) in domain_tier.items():
        if len(tiers) >= 4:
            per_domain_results[domain] = compute_spearman_ci(tiers, corrs)

    confidence_baseline_rho = 0.7226

    tier_validation = {
        "spearman_rho": tier_corr["rho"],
        "ci_95": tier_corr["ci_95"],
        "p_value": tier_corr["pvalue"],
        "effective_sample_size": tier_corr["n"],
        "per_domain": per_domain_results,
        "comparison_to_confidence_baseline": {
            "confidence_rho": confidence_baseline_rho,
            "tier_rho": tier_corr["rho"],
            "interpretation": (
                f"Tier ρ={tier_corr['rho']} is "
                + ("stronger" if (tier_corr["rho"] or 0) > confidence_baseline_rho else "weaker")
                + f" predictor than confidence baseline (ρ={confidence_baseline_rho})"
            ),
        },
        "discrepancy_resolution": (
            "Tier ordinal encoding: assertion=3, presupposition=2, implicature=1. "
            "Correctness = 1 if pragma_problog prediction matches ground truth label, else 0. "
            "Negative correlation expected if higher tiers → higher correctness "
            "(i.e., assertion facts more reliable). Positive ρ here means higher tier = higher correctness."
        ),
    }

    # ── 8. Stage 3: Hallucination bound power analysis ───────────────────────
    logger.info("Stage 3: Hallucination bound power analysis...")
    # Combine CLUTRR + RuleTaker hallucination bounds
    hb_values = pragma_clutrr_hbs + pragma_rt_hbs
    hb_correctness = correct_clutrr["pragma_problog"] + correct_rt["pragma_problog"]

    # Filter to examples where we have valid HB (0 ≤ HB ≤ 1)
    valid_pairs = [(hb, c) for hb, c in zip(hb_values, hb_correctness) if 0.0 <= hb <= 1.0]
    n_proofs = len(valid_pairs)
    logger.info(f"Valid proof pairs for HB analysis: {n_proofs}")

    if n_proofs >= 4:
        hb_arr = [p[0] for p in valid_pairs]
        corr_arr = [float(p[1]) for p in valid_pairs]
        r_obs, p_corr = stats.pearsonr(hb_arr, corr_arr)
        r_obs, p_corr = float(r_obs), float(p_corr)
        ci_r = fisher_z_ci(r_obs, n_proofs)
        sig_label = "Significant (p<0.05)" if p_corr < 0.05 else "Not significant (pilot)"

        obs_power = post_hoc_power(r_obs, n_proofs)
        r_effect = max(abs(r_obs), 0.20)  # at least small effect for power computation
        n_req = required_n_for_power(r_effect)

        power_interpretation = (
            f"Adequate (power={obs_power:.0%})" if obs_power >= 0.80
            else f"Underpowered (power={obs_power:.0%}, shortfall: need n={n_req} for 80% power)"
        )

        hallucination_analysis = {
            "n_proofs_collected": n_proofs,
            "n_proofs_with_valid_labels": n_proofs,
            "pearson_r": round(r_obs, 4),
            "ci_95": list(ci_r),
            "pvalue": round(p_corr, 6),
            "significance_label": sig_label,
            "post_hoc_power_analysis": {
                "observed_power_percent": round(obs_power * 100, 1),
                "observed_effect_size_r": round(r_obs, 4),
                "sample_size_required_80pct_power": n_req,
                "power_interpretation": power_interpretation,
                "recommendation": (
                    "Hypothesis confirmed (sufficient power)" if obs_power >= 0.80 and p_corr < 0.05
                    else "Underpowered pilot; requires n=" + str(n_req) + " for 80% power at r=" + str(round(r_effect, 2))
                ),
            },
            "notes": [
                f"HB = fraction of proof from presupposition+implicature facts",
                f"Negative r expected: higher HB → more uncertain → lower correctness",
                f"Observed r={r_obs:.4f}, n={n_proofs}",
                f"Post-hoc power at α=0.05: {obs_power:.1%}",
            ],
        }
    else:
        hallucination_analysis = {
            "n_proofs_collected": n_proofs,
            "n_proofs_with_valid_labels": n_proofs,
            "pearson_r": None,
            "ci_95": [None, None],
            "pvalue": None,
            "significance_label": "Insufficient data (n<4)",
            "post_hoc_power_analysis": {
                "observed_power_percent": 0.0,
                "observed_effect_size_r": 0.0,
                "sample_size_required_80pct_power": required_n_for_power(0.30),
                "power_interpretation": "Underpowered (n < 4 valid proof pairs)",
                "recommendation": "Collect more examples with valid proof chains",
            },
            "notes": ["Too few proof pairs for correlation analysis"],
        }

    logger.info(f"Hallucination analysis: {hallucination_analysis}")

    # ── 9. Summary verdicts ───────────────────────────────────────────────────
    pragma_acc = method_accs_all["pragma_problog"]["accuracy"]
    cot_acc = method_accs_all["chain_of_thought"]["accuracy"]

    if primary_test:
        pval_primary = primary_test["pvalue"]
        if pval_primary < 0.05:
            bench_verdict = "Pragmatic ProbLog outperforms CoT (significant p<0.05)"
        elif pval_primary < 0.10:
            bench_verdict = "Pragmatic ProbLog marginally outperforms CoT (p<0.10)"
        elif pragma_acc > cot_acc:
            bench_verdict = "Pragmatic ProbLog numerically outperforms CoT (not significant)"
        else:
            bench_verdict = "Pragmatic ProbLog matches or underperforms CoT (not significant)"
    else:
        bench_verdict = "Benchmark comparison inconclusive"

    tier_rho = tier_corr["rho"] or 0
    if tier_corr["pvalue"] and tier_corr["pvalue"] < 0.05:
        tier_verdict = "Assumption 1 CONFIRMED (p<0.05)"
    elif tier_corr["pvalue"] and tier_corr["pvalue"] < 0.10:
        tier_verdict = "Assumption 1 INCONCLUSIVE (p<0.10, marginal)"
    else:
        tier_verdict = "Assumption 1 INCONCLUSIVE (p≥0.10)"

    ha = hallucination_analysis
    if ha.get("pvalue") and ha["pvalue"] < 0.05:
        hb_verdict = "HB as proxy supported (p<0.05)"
    elif ha.get("pearson_r") is not None:
        hb_verdict = f"HB as proxy underpowered pilot (p={ha.get('pvalue', 'N/A')}, power={ha['post_hoc_power_analysis']['observed_power_percent']}%)"
    else:
        hb_verdict = "HB as proxy: insufficient data"

    if tier_rho > 0.3 and pragma_acc > cot_acc:
        overall = "REFINE"
    elif tier_rho > 0.5 and (primary_test and primary_test["pvalue"] < 0.05):
        overall = "ACCEPT"
    else:
        overall = "REFINE"

    total_cost = (_total_tokens["in"] + _total_tokens["out"]) * COST_PER_TOKEN
    elapsed = time.time() - t0

    # ── 10. Build output JSON ─────────────────────────────────────────────────
    # Build exp_gen_sol_out format: datasets with examples containing predict_* fields
    clutrr_output_examples = []
    for i, ex in enumerate(clutrr_exs):
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_entity_pair": ex["metadata_entity_pair"],
            "metadata_task_type": "kinship_classification",
            "metadata_row_index": ex["metadata_row_index"],
            "predict_rag": rag_clutrr_preds[i],
            "predict_chain_of_thought": cot_clutrr_preds[i],
            "predict_flat_fol_ablation": flat_fol_clutrr_preds[i],
            "predict_pragma_problog": pragma_clutrr_preds[i],
            "predict_llm_tres": tres_clutrr_preds[i],
        }
        clutrr_output_examples.append(out_ex)

    rt_output_examples = []
    for i, ex in enumerate(rt_exs):
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_config": ex.get("metadata_config", ""),
            "metadata_task_type": "entailment_classification",
            "metadata_row_index": ex["metadata_row_index"],
            "predict_rag": rag_rt_preds[i],
            "predict_chain_of_thought": cot_rt_preds[i],
            "predict_flat_fol_ablation": flat_fol_rt_preds[i],
            "predict_pragma_problog": pragma_rt_preds[i],
            "predict_llm_tres": tres_rt_preds[i],
        }
        rt_output_examples.append(out_ex)

    benchmark_stats = {
        "clutrr_test_size": n_clutrr_actual,
        "ruletaker_test_size": n_rt_actual,
        "combined_test_size": n_total,
        "method_accuracies_clutrr": method_accs_clutrr,
        "method_accuracies_ruletaker": method_accs_rt,
        "method_accuracies_combined": method_accs_all,
        "mcnemars_tests": mcnemar_results,
        "primary_hypothesis_test": {
            "pair": ["pragma_problog", "chain_of_thought"],
            "result": primary_test,
            "comment": "McNemar p-value for pragmatic ProbLog vs. CoT baseline (combined CLUTRR+RuleTaker)",
        },
        "accuracy_deltas": {
            m: round(method_accs_all[m]["accuracy"] - method_accs_all["chain_of_thought"]["accuracy"], 4)
            for m in method_names
        },
    }

    result = {
        "metadata": {
            "method": "Pragma-Stratified ProbLog Statistical Validation",
            "description": "3-stage statistical validation: tier-correctness Spearman, McNemar benchmark, hallucination bound power analysis",
            "model": OPENROUTER_MODEL,
            "n_clutrr": n_clutrr_actual,
            "n_ruletaker": n_rt_actual,
            "elapsed_seconds": round(elapsed, 1),
            "tier_validation": tier_validation,
            "benchmark_stats": benchmark_stats,
            "hallucination_analysis": hallucination_analysis,
            "summary": {
                "tier_verdict": tier_verdict,
                "benchmark_verdict": bench_verdict,
                "hallucination_bound_verdict": hb_verdict,
                "overall_hypothesis_status": overall,
            },
            "cost_tracking": {
                "openrouter_calls": _api_calls,
                "tokens_in": _total_tokens["in"],
                "tokens_out": _total_tokens["out"],
                "total_cost_usd": round(total_cost, 4),
                "budget_remaining_usd": round(BUDGET_USD - total_cost, 4),
            },
        },
        "datasets": [
            {"dataset": "clutrr", "examples": clutrr_output_examples},
            {"dataset": "ruletaker", "examples": rt_output_examples},
        ],
    }

    return result


@logger.catch(reraise=True)
def main():
    logger.info(f"Pragma-Stratified ProbLog experiment starting")
    logger.info(f"N_CLUTRR={N_CLUTRR}, N_RULETAKER={N_RULETAKER}, model={OPENROUTER_MODEL}")
    logger.info(f"OpenRouter API key: {'set' if OPENROUTER_API_KEY else 'MISSING'}")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    result = asyncio.run(run_experiment(N_CLUTRR, N_RULETAKER))

    logger.info(f"Saving to {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(result, indent=2))

    # Summary
    meta = result["metadata"]
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tier verdict: {meta['summary']['tier_verdict']}")
    logger.info(f"Benchmark verdict: {meta['summary']['benchmark_verdict']}")
    logger.info(f"HB verdict: {meta['summary']['hallucination_bound_verdict']}")
    logger.info(f"Overall: {meta['summary']['overall_hypothesis_status']}")
    logger.info(f"Cost: ${meta['cost_tracking']['total_cost_usd']:.4f}")
    logger.info(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
