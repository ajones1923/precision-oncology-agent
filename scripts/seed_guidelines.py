#!/usr/bin/env python3
"""Seed the onco_guidelines collection with curated guideline data.

Loads guideline_seed_data.json and uses GuidelineIngestPipeline to parse
and embed each record into the onco_guidelines Milvus collection.

Usage: python3 scripts/seed_guidelines.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from src.collections import OncoCollectionManager
from src.ingest.guideline_parser import GuidelineIngestPipeline


class SimpleEmbedder:
    """Thin wrapper around SentenceTransformer for pipeline compatibility."""

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode(self, texts):
        return self.model.encode(texts).tolist()


def main():
    seed_file = PROJECT_ROOT / "data" / "reference" / "guideline_seed_data.json"
    if not seed_file.exists():
        print(f"ERROR: Seed file not found: {seed_file}")
        return 1

    print("=" * 60)
    print("Precision Oncology -- Guideline Data Seeder")
    print("=" * 60)

    print("\n[1/3] Connecting to Milvus...")
    manager = OncoCollectionManager()
    manager.connect()
    try:
        stats = manager.get_collection_stats("onco_guidelines")
        existing = stats.get("num_entities", 0)
    except Exception:
        existing = 0
        manager.create_collection("onco_guidelines")
    print(f"  onco_guidelines currently has {existing} records")

    print("\n[2/3] Loading BGE-small-en-v1.5 embedder...")
    embedder = SimpleEmbedder()

    print("\n[3/3] Ingesting guideline seed data...")
    pipeline = GuidelineIngestPipeline(
        collection_manager=manager,
        embedder=embedder,
        seed_path=str(seed_file),
    )
    count = pipeline.run()

    try:
        stats = manager.get_collection_stats("onco_guidelines")
        final = stats.get("num_entities", 0)
    except Exception:
        final = count

    print(f"\n{'=' * 60}")
    print(f"DONE: Inserted {count} guideline records")
    print(f"  onco_guidelines now has {final} records")
    print(f"{'=' * 60}")

    manager.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
