"""
Build and query a local Faiss vector index over Neo4j nodes (Entity, Pattern, Rule,
Template, NerRule) for Graph RAG retrieval.

Environment (same as load_json_to_neo4j):
  NEO4J_URI       default bolt://127.0.0.1:7687
  NEO4J_USER      default neo4j
  NEO4J_PASSWORD  required for build

Optional:
  EMBEDDING_MODEL  default sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU-friendly)

Usage:
  python vector_index.py build --out ../vector-index
  python vector_index.py search --out ../vector-index --q "Chase payment line"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Neo4j imported inside get_driver() so `--help` works without a full env.

# Lazy imports for sentence_transformers/faiss (only when running build/search)
_faiss = None


def _lazy_embedding_deps():
    global _st_model, _faiss
    if _faiss is None:
        import faiss

        _faiss = faiss
    return _faiss


def get_driver():
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("NEO4J_PASSWORD is required for build.", file=sys.stderr)
        sys.exit(1)
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_entities(session) -> list[dict[str, Any]]:
    q = """
    MATCH (e:Entity)
    RETURN e._id AS id, e.name AS name, e.entityType AS entityType,
           e.dataType AS dataType, e.compoundEntityScope AS compoundEntityScope
    """
    return [dict(r) for r in session.run(q)]


def fetch_patterns(session) -> list[dict[str, Any]]:
    q = """
    MATCH (p:Pattern)
    OPTIONAL MATCH (p)-[:EXTRACTS_ENTITY]->(e:Entity)
    RETURN p._id AS id, p.name AS name, p.type AS type,
           p.regexPattern AS regexPattern, p.stringPattern AS stringPattern,
           p.stringPatternRegex AS stringPatternRegex, p.spacyPatternJson AS spacyPatternJson,
           p.source AS source, collect(DISTINCT e._id) AS entity_ids
    """
    return [dict(r) for r in session.run(q)]


def fetch_rules(session) -> list[dict[str, Any]]:
    q = """
    MATCH (r:Rule)
    RETURN r._id AS id, r.name AS name, r.ruleType AS ruleType,
           r.filtersJson AS filtersJson, r.entityJson AS entityJson,
           r.valueJson AS valueJson, r.keyJson AS keyJson
    """
    return [dict(r) for r in session.run(q)]


def fetch_templates(session) -> list[dict[str, Any]]:
    q = """
    MATCH (t:Template)
    OPTIONAL MATCH (t)-[:HAS_ENTITY_SETTING]->(tes)-[:HAS_NER_RULE]->(nr:NerRule)
    RETURN t.nerTemplateId AS id, t.name AS name,
           collect(DISTINCT tes.name) AS setting_names,
           collect(DISTINCT nr.name) AS ner_rule_names,
           collect(DISTINCT nr.nerRuleType) AS ner_rule_types
    """
    return [dict(r) for r in session.run(q)]


def fetch_ner_rules(session) -> list[dict[str, Any]]:
    q = """
    MATCH (nr:NerRule)
    RETURN nr.stableId AS id, nr.name AS name, nr.nerRuleType AS nerRuleType,
           nr.nerTemplateId AS nerTemplateId,
           nr.valueMatchSettingJson AS valueMatchSettingJson,
           nr.keyMatchSettingJson AS keyMatchSettingJson
    """
    return [dict(r) for r in session.run(q)]


def _truncate(s: str | None, max_chars: int = 6000) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def entity_embed_text(row: dict[str, Any]) -> str:
    parts = [
        "kind: entity",
        f"id: {row.get('id')}",
        f"name: {row.get('name')}",
        f"entityType: {row.get('entityType')}",
        f"dataType: {row.get('dataType')}",
    ]
    if row.get("compoundEntityScope"):
        parts.append(f"compoundEntityScope: {row['compoundEntityScope']}")
    return "\n".join(p for p in parts if p.split(": ")[-1] not in ("None", ""))


def pattern_embed_text(row: dict[str, Any]) -> str:
    eids = [x for x in (row.get("entity_ids") or []) if x]
    parts = [
        "kind: pattern",
        f"id: {row.get('id')}",
        f"name: {row.get('name')}",
        f"type: {row.get('type')}",
        f"source: {row.get('source')}",
        f"regexPattern: {row.get('regexPattern')}",
        f"stringPattern: {row.get('stringPattern')}",
        f"stringPatternRegex: {row.get('stringPatternRegex')}",
        f"spacyPatternJson: {_truncate(row.get('spacyPatternJson'), 4000)}",
        f"extracts_entities: {', '.join(eids)}",
    ]
    return "\n".join(str(p) for p in parts)


def rule_embed_text(row: dict[str, Any]) -> str:
    parts = [
        "kind: rule",
        f"id: {row.get('id')}",
        f"name: {row.get('name')}",
        f"ruleType: {row.get('ruleType')}",
        f"filtersJson: {_truncate(row.get('filtersJson'))}",
        f"entityJson: {_truncate(row.get('entityJson'))}",
        f"valueJson: {_truncate(row.get('valueJson'))}",
        f"keyJson: {_truncate(row.get('keyJson'))}",
    ]
    return "\n".join(str(p) for p in parts)


def template_embed_text(row: dict[str, Any]) -> str:
    sn = [x for x in (row.get("setting_names") or []) if x]
    rn = [x for x in (row.get("ner_rule_names") or []) if x]
    rt = [x for x in (row.get("ner_rule_types") or []) if x]
    parts = [
        "kind: template",
        f"nerTemplateId: {row.get('id')}",
        f"name: {row.get('name')}",
        f"entity_settings: {', '.join(sn)}",
        f"ner_rules: {', '.join(rn)}",
        f"ner_rule_types: {', '.join(rt)}",
    ]
    return "\n".join(str(p) for p in parts)


def ner_rule_embed_text(row: dict[str, Any]) -> str:
    parts = [
        "kind: ner_rule",
        f"stableId: {row.get('id')}",
        f"name: {row.get('name')}",
        f"nerRuleType: {row.get('nerRuleType')}",
        f"nerTemplateId: {row.get('nerTemplateId')}",
        f"valueMatchSettingJson: {_truncate(row.get('valueMatchSettingJson'))}",
        f"keyMatchSettingJson: {_truncate(row.get('keyMatchSettingJson'))}",
    ]
    return "\n".join(str(p) for p in parts)


@dataclass
class IndexRecord:
    kind: str
    primary_id: str
    text: str
    extra: dict[str, Any]


def collect_records(session) -> list[IndexRecord]:
    out: list[IndexRecord] = []

    for row in fetch_entities(session):
        rid = row.get("id")
        if not rid:
            continue
        out.append(
            IndexRecord(
                kind="entity",
                primary_id=str(rid),
                text=entity_embed_text(row),
                extra={"neo4j_label": "Entity", "id": rid},
            )
        )

    for row in fetch_patterns(session):
        rid = row.get("id")
        if not rid:
            continue
        out.append(
            IndexRecord(
                kind="pattern",
                primary_id=str(rid),
                text=pattern_embed_text(row),
                extra={"neo4j_label": "Pattern", "id": rid, "source": row.get("source")},
            )
        )

    for row in fetch_rules(session):
        rid = row.get("id")
        if not rid:
            continue
        out.append(
            IndexRecord(
                kind="rule",
                primary_id=str(rid),
                text=rule_embed_text(row),
                extra={"neo4j_label": "Rule", "id": rid},
            )
        )

    for row in fetch_templates(session):
        tid = row.get("id")
        if tid is None:
            continue
        out.append(
            IndexRecord(
                kind="template",
                primary_id=str(tid),
                text=template_embed_text(row),
                extra={"neo4j_label": "Template", "nerTemplateId": tid},
            )
        )

    for row in fetch_ner_rules(session):
        sid = row.get("id")
        if not sid:
            continue
        out.append(
            IndexRecord(
                kind="ner_rule",
                primary_id=str(sid),
                text=ner_rule_embed_text(row),
                extra={"neo4j_label": "NerRule", "stableId": sid},
            )
        )

    return out


def cmd_build(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    driver = get_driver()
    records: list[IndexRecord] = []
    with driver.session() as session:
        records = collect_records(session)
    driver.close()

    if not records:
        print("No records from Neo4j. Check data and connection.", file=sys.stderr)
        sys.exit(1)

    texts = [r.text for r in records]
    print(f"Embedding {len(texts)} records...")
    vectors = model.encode(
        texts,
        batch_size=int(args.batch_size),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = vectors.astype("float32")

    faiss = _lazy_embedding_deps()
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = out_dir / "vectors.faiss"
    meta_path = out_dir / "metadata.jsonl"
    cfg_path = out_dir / "config.json"

    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            line = {
                "index": i,
                "kind": rec.kind,
                "primary_id": rec.primary_id,
                "text": rec.text,
                "extra": rec.extra,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    cfg = {
        "embedding_model": model_name,
        "embedding_dim": dim,
        "metric": "inner_product_on_normalized_vectors_equals_cosine_similarity",
        "record_count": len(records),
        "files": {"faiss": index_path.name, "metadata": meta_path.name},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"Wrote {index_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {cfg_path}")


def load_metadata(meta_path: Path) -> list[dict[str, Any]]:
    rows = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cmd_search(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    out_dir = Path(args.out).resolve()
    cfg_path = out_dir / "config.json"
    if not cfg_path.is_file():
        print(f"Missing {cfg_path}. Run build first.", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    index_path = out_dir / cfg["files"]["faiss"]
    meta_path = out_dir / cfg["files"]["metadata"]

    model = SentenceTransformer(model_name)
    faiss = _lazy_embedding_deps()
    index = faiss.read_index(str(index_path))
    metadata = load_metadata(meta_path)

    qvec = model.encode(
        [args.q], show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")
    k = min(int(args.k), index.ntotal)
    scores, indices = index.search(qvec, k)

    print(f"Query: {args.q!r}\nTop {k}:\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        print(f"{rank}. score={score:.4f}  kind={row['kind']}  id={row['primary_id']}")
        preview = row["text"].replace("\n", " ")[:200]
        print(f"   {preview}...")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Faiss vector index over Neo4j KB")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Pull from Neo4j and build Faiss index")
    b.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "vector-index"),
        help="Output directory (default: graph-db/vector-index)",
    )
    b.add_argument("--batch-size", type=int, default=32)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="Query existing index")
    s.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "vector-index"),
        help="Index directory",
    )
    s.add_argument("--q", required=True, help="Search text")
    s.add_argument("-k", type=int, default=8, help="Top-k")
    s.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
