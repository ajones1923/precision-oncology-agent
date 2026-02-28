#!/usr/bin/env python3
"""Seed the onco_trials collection with curated clinical trial data.

Loads trial_seed_data.json and uses ClinicalTrialsIngestPipeline to
embed each record into the onco_trials Milvus collection.

Usage: python3 scripts/seed_trials.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from src.collections import OncoCollectionManager
from src.ingest.clinical_trials_parser import ClinicalTrialsIngestPipeline


class SimpleEmbedder:
    """Thin wrapper around SentenceTransformer for pipeline compatibility."""

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode(self, texts):
        return self.model.encode(texts).tolist()


def main():
    seed_file = PROJECT_ROOT / "data" / "reference" / "trial_seed_data.json"
    if not seed_file.exists():
        print(f"ERROR: Seed file not found: {seed_file}")
        return 1

    print("=" * 60)
    print("Precision Oncology -- Clinical Trial Data Seeder")
    print("=" * 60)

    with open(seed_file, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    print(f"\n  Loaded {len(seed_data)} trial records from seed file")

    print("\n[1/3] Connecting to Milvus...")
    manager = OncoCollectionManager()
    manager.connect()
    try:
        stats = manager.get_collection_stats("onco_trials")
        existing = stats.get("num_entities", 0)
    except Exception:
        existing = 0
        manager.create_collection("onco_trials")
    print(f"  onco_trials currently has {existing} records")

    print("\n[2/3] Loading BGE-small-en-v1.5 embedder...")
    embedder = SimpleEmbedder()

    print("\n[3/3] Embedding and inserting trial seed data...")
    # Transform seed records into onco_trials schema format
    pipeline = ClinicalTrialsIngestPipeline(
        collection_manager=manager, embedder=embedder
    )

    trial_records = []
    for rec in seed_data:
        text_summary = f"{rec['title']}. {rec['text_summary']}"
        trial_records.append({
            "id": rec["nct_id"],
            "title": rec["title"],
            "text_summary": text_summary,
            "text": text_summary,
            "phase": rec.get("phase", ""),
            "status": rec.get("status", ""),
            "sponsor": rec.get("sponsor", ""),
            "cancer_types": rec.get("cancer_types", ""),
            "biomarker_criteria": rec.get("biomarker_criteria", ""),
            "enrollment": str(rec.get("enrollment", "")),
            "start_year": "",
            "interventions": rec.get("interventions", ""),
            "source_type": "seed",
        })

    count = pipeline.embed_and_store(trial_records)

    try:
        stats = manager.get_collection_stats("onco_trials")
        final = stats.get("num_entities", 0)
    except Exception:
        final = count

    print(f"\n{'=' * 60}")
    print(f"DONE: Inserted {count} trial records")
    print(f"  onco_trials now has {final} records")
    print(f"{'=' * 60}")

    manager.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
