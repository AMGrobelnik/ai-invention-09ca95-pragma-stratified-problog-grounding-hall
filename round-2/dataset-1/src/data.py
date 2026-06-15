#!/usr/bin/env python3
"""Load ContractNLI and NYT News datasets, convert to exp_sel_data_out.json schema.

Each example = one source document (contract clause or news article) that can
serve as input for downstream pragma-stratified fact extraction.
"""

import json
import sys
import hashlib
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path("/ai-inventor/aii_data/runs/5b0b4/3_invention_loop/iter_2/gen_art/gen_art_dataset_1")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

MIN_CHARS = 200   # minimum text length to include
MAX_CHARS = 3500  # truncate at this length


def load_contract_nli() -> list[dict]:
    """Load ContractNLI train split. Each row = (premise, hypothesis, label)."""
    path = DATASETS_DIR / "full_kiddothe2b_contract-nli_contractnli_a_train.json"
    logger.info(f"Loading ContractNLI from {path}")
    rows = json.loads(path.read_text())
    logger.info(f"Loaded {len(rows)} rows from ContractNLI")

    examples = []
    seen_premises = set()

    for i, row in enumerate(rows):
        premise = row.get("premise", "") or ""
        hypothesis = row.get("hypothesis", "") or ""
        label_int = row.get("label", -1)
        label_str = LABEL_MAP.get(label_int, "unknown")

        if len(premise) < MIN_CHARS:
            continue

        # Truncate if too long
        premise_trunc = premise[:MAX_CHARS]

        # Use premise hash to track unique contract clauses
        premise_hash = hashlib.md5(premise.encode()).hexdigest()[:8]

        examples.append({
            "input": premise_trunc,
            "output": label_str,
            "metadata_document_type": "legal",
            "metadata_source_dataset": "contract-nli",
            "metadata_hypothesis": hypothesis[:500],
            "metadata_label_int": label_int,
            "metadata_premise_hash": premise_hash,
            "metadata_row_index": i,
            "metadata_text_length": len(premise),
        })

    logger.info(f"ContractNLI: {len(examples)} examples after filtering")
    return examples


def load_nyt_news() -> list[dict]:
    """Load NYT News test split. Each row = one news article."""
    path = DATASETS_DIR / "full_ErikCikalleshi_new_york_times_news_2000_2007_default_test.json"
    logger.info(f"Loading NYT News from {path}")
    rows = json.loads(path.read_text())
    logger.info(f"Loaded {len(rows)} rows from NYT News")

    examples = []
    for i, row in enumerate(rows):
        content = row.get("content", "") or ""
        title = row.get("title", "") or ""
        date = row.get("date", "")

        if len(content) < MIN_CHARS:
            continue

        content_trunc = content[:MAX_CHARS]

        examples.append({
            "input": content_trunc,
            "output": title[:200] if title else "untitled",
            "metadata_document_type": "news",
            "metadata_source_dataset": "nyt-news-2000-2007",
            "metadata_title": title[:200],
            "metadata_date": str(date),
            "metadata_row_index": i,
            "metadata_text_length": len(content),
        })

    logger.info(f"NYT News: {len(examples)} examples after filtering")
    return examples


@logger.catch(reraise=True)
def main():
    Path("logs").mkdir(exist_ok=True)

    contract_examples = load_contract_nli()
    nyt_examples = load_nyt_news()

    # NYT News selected as best dataset: full articles (2000-11000 chars) support
    # extracting 8-10 facts per doc across all pragmatic tiers (assertion/presupposition/implicature).
    # ContractNLI clauses are too short (200-750 chars) for diverse pragmatic-tier extraction.
    output = {
        "metadata": {
            "description": "Source documents for pragma-stratified fact extraction",
            "selected_dataset": "nyt-news-2000-2007",
            "selection_rationale": (
                "NYT News articles are long (2000-11000 chars), factually dense, and cover "
                "diverse topics — ideal for extracting 8-10 facts per document across all "
                "pragmatic tiers (assertion, presupposition, implicature)."
            ),
            "sources": ["NYT News 2000-2007 (ErikCikalleshi/new_york_times_news_2000_2007)"],
            "document_types": ["news"],
            "total_examples": len(nyt_examples),
        },
        "datasets": [
            {
                "dataset": "nyt-news-2000-2007",
                "examples": nyt_examples,
            },
        ],
    }

    # Split into parts under 100MB (~27625 examples each)
    import glob
    examples_all = nyt_examples
    part_size = 27625
    parts = [examples_all[i:i+part_size] for i in range(0, len(examples_all), part_size)]
    out_dir = WORKSPACE / "full_data_out"
    out_dir.mkdir(exist_ok=True)
    for j, part in enumerate(parts, 1):
        part_data = {**output, "datasets": [{"dataset": output["datasets"][0]["dataset"], "examples": part}]}
        part_path = out_dir / f"full_data_out_{j}.json"
        part_path.write_text(json.dumps(part_data, indent=2))
        logger.info(f"Saved part {j}: {len(part)} examples -> {part_path} ({part_path.stat().st_size/1e6:.1f}MB)")
    logger.info(f"Total: {len(examples_all)} examples across {len(parts)} files")


if __name__ == "__main__":
    main()
