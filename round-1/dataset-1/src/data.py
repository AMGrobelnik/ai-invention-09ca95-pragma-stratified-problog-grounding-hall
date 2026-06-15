#!/usr/bin/env python3
"""Load 4 datasets and standardize to exp_sel_data_out.json schema."""

from loguru import logger
from pathlib import Path
import json
import re
import sys
from collections import defaultdict

WORKSPACE = Path("/ai-inventor/aii_data/runs/5b0b4/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
DATASETS_DIR = WORKSPACE / "temp/datasets"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs/data.log"), rotation="30 MB", level="DEBUG")

MAX_RULETAKER = 50000  # stratified cap to keep file manageable


def load_clutrr() -> list[dict]:
    path = DATASETS_DIR / "full_tasksource_clutrr_default_train.json"
    logger.info(f"Loading CLUTRR from {path}")
    data = json.loads(path.read_text())
    examples = []
    for i, row in enumerate(data):
        sentence1 = str(row.get("sentence1", ""))
        sentence2 = str(row.get("sentence2", ""))
        label = str(row.get("labels", ""))
        if not sentence1 or not label:
            continue
        examples.append({
            "input": sentence1,
            "output": label,
            "metadata_entity_pair": sentence2,
            "metadata_task_type": "kinship_classification",
            "metadata_row_index": i,
        })
    logger.info(f"CLUTRR: {len(examples)} examples loaded")
    return examples


def load_ruletaker() -> list[dict]:
    path = DATASETS_DIR / "full_tasksource_ruletaker_default_train.json"
    logger.info(f"Loading RuleTaker from {path} (will cap at {MAX_RULETAKER})")
    data = json.loads(path.read_text())
    total = len(data)
    logger.info(f"RuleTaker raw rows: {total}")

    # Group by depth config for stratified sampling
    by_config: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(data):
        by_config[row.get("config", "unknown")].append((i, row))

    examples = []
    for cfg in sorted(by_config.keys()):
        rows = by_config[cfg]
        n = max(1, round(len(rows) / total * MAX_RULETAKER))
        for orig_i, row in rows[:n]:
            context = row.get("context", "")
            question = row.get("question", "")
            label = str(row.get("label", ""))
            inp = f"Context: {context}\n\nQuestion: {question}"
            examples.append({
                "input": inp,
                "output": label,
                "metadata_config": str(cfg),
                "metadata_task_type": "entailment_classification",
                "metadata_row_index": orig_i,
            })
    logger.info(f"RuleTaker: {len(examples)} examples (stratified from {total})")
    return examples


def load_proofwriter() -> list[dict]:
    path = DATASETS_DIR / "full_D3xter1922_proofwriter-dataset_default_train.json"
    logger.info(f"Loading ProofWriter from {path}")
    data = json.loads(path.read_text())
    examples = []
    for i, row in enumerate(data):
        translation = row.get("translation", {})
        en_text = str(translation.get("en", ""))
        ro_text = str(translation.get("ro", ""))

        ans_match = re.search(r'\$answer\$\s*=\s*(\w+)', ro_text)
        answer = ans_match.group(1) if ans_match else ""

        proof_match = re.search(r'\$proof\$\s*=\s*([^;$]+)', ro_text)
        proof = proof_match.group(1).strip() if proof_match else ""

        if not en_text or not answer:
            continue

        examples.append({
            "input": en_text,
            "output": answer,
            "metadata_proof": proof,
            "metadata_task_type": "proof_verification",
            "metadata_row_index": i,
        })
    logger.info(f"ProofWriter: {len(examples)} examples loaded")
    return examples


def load_entailment_bank() -> list[dict]:
    path = DATASETS_DIR / "full_nguyen-brat_entailment_bank_default_train.json"
    logger.info(f"Loading EntailmentBank from {path}")
    data = json.loads(path.read_text())
    examples = []
    for i, row in enumerate(data):
        question = str(row.get("question", ""))
        answer = row.get("answer", [])
        cot = row.get("cot", [])
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)
        if not question or not answer:
            continue
        examples.append({
            "input": question,
            "output": answer,
            "metadata_cot": json.dumps(cot),
            "metadata_ref_id": str(row.get("ref_id", "")),
            "metadata_task_type": "science_qa_entailment",
            "metadata_row_index": i,
        })
    logger.info(f"EntailmentBank: {len(examples)} examples loaded")
    return examples


@logger.catch(reraise=True)
def main():
    (WORKSPACE / "logs").mkdir(exist_ok=True)

    datasets = []

    datasets.append({"dataset": "clutrr", "examples": load_clutrr()})
    datasets.append({"dataset": "ruletaker", "examples": load_ruletaker()})

    output = {
        "metadata": {
            "description": "Pragmatic tier annotation source datasets: kinship reasoning and rule-based entailment benchmarks",
            "source_hf_ids": [
                "tasksource/clutrr",
                "tasksource/ruletaker",
            ],
        },
        "datasets": datasets,
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    total = sum(len(d["examples"]) for d in datasets)
    logger.info(f"Saved {total} total examples across {len(datasets)} datasets → {out_path}")
    logger.info("Sizes per dataset: " + ", ".join(
        f"{d['dataset']}={len(d['examples'])}" for d in datasets
    ))


if __name__ == "__main__":
    main()
