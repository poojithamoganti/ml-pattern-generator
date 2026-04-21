"""
Graph RAG: hybrid retrieval over the KB vector index + Neo4j expansion.

**Previous / baseline (GRAPH_RAG_HYBRID=0):** single dense pass — Faiss ``IndexFlatIP``
on L2-normalized MiniLM embeddings (inner product = cosine similarity). One query
string embeds the whole prompt (entities + hints + long OCR excerpt), which often
*dilutes* the vector and misses regex-heavy KB rows.

**Default (GRAPH_RAG_HYBRID=1):** *Reciprocal Rank Fusion* (RRF) of three ranked lists:
  1. **BM25** (``rank_bm25``) over the same per-row ``text`` stored in ``metadata.jsonl``
     — strong lexical match on ids, rule names, and literal phrases.
  2. **Dense (entity-focused)** — embeds only the block *before* ``Document excerpt:``
     (entities + hints), so semantics align with *field* intent, not raw OCR noise.
  3. **Dense (full query)** — embeds the full retrieval string including a *short*
     doc snippet (``GRAPH_RAG_DOC_SNIPPET_CHARS``), for layout/context alignment.

RRF merges ranks with score ``sum 1/(RRF_K + rank)`` (classic fusion; ``RRF_K``≈60).
Final top-``k`` rows are expanded in Neo4j when credentials are set.

Re-indexing: changing the embedding model in ``vector_index.py`` requires rebuilding
``vectors.faiss``; hybrid BM25 uses only ``metadata.jsonl`` text.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from app.config import (
    GRAPH_RAG_BM25_BRANCH_K,
    GRAPH_RAG_DENSE_BRANCH_K,
    GRAPH_RAG_DOC_SNIPPET_CHARS,
    GRAPH_RAG_HYBRID,
    GRAPH_RAG_RRF_K,
)

logger = logging.getLogger(__name__)

_faiss = None
_ST = None


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25."""
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _reciprocal_rank_fusion(
    rankings: list[list[int]],
    k_const: int,
) -> dict[int, float]:
    """Cormack et al.-style RRF: fused_score(d) = sum_i 1/(k + rank_i(d))."""
    scores: dict[int, float] = {}
    for rks in rankings:
        if not rks:
            continue
        for rank, doc_id in enumerate(rks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_const + rank)
    return scores


def _faiss_mod():
    global _faiss
    if _faiss is None:
        import faiss

        _faiss = faiss
    return _faiss


def _sentence_transformer():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer

        _ST = SentenceTransformer
    return _ST


class GraphRagStore:
    """Loads Faiss + metadata once; encodes queries with the same model as build."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.cfg: dict[str, Any] = {}
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._model = None
        self._dim = 0

    def available(self) -> bool:
        cfg_path = self.index_dir / "config.json"
        return cfg_path.is_file() and (self.index_dir / "vectors.faiss").is_file()

    def load(self) -> None:
        cfg_path = self.index_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Graph RAG: missing {cfg_path}")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        meta_name = self.cfg.get("files", {}).get("metadata", "metadata.jsonl")
        meta_path = self.index_dir / meta_name
        self._metadata = []
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._metadata.append(json.loads(line))
        self._bm25 = None  # lazy: BM25Okapi over tokenized metadata text
        faiss = _faiss_mod()
        idx_name = self.cfg.get("files", {}).get("faiss", "vectors.faiss")
        self._index = faiss.read_index(str(self.index_dir / idx_name))
        model_name = self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        ST = _sentence_transformer()
        self._model = ST(model_name)
        self._dim = int(self.cfg.get("embedding_dim", self._model.get_sentence_embedding_dimension()))

    def ensure_loaded(self) -> None:
        if self._index is None:
            self.load()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Single-query dense retrieval (cosine via normalized IP)."""
        self.ensure_loaded()
        assert self._model is not None and self._index is not None
        k = min(k, int(self._index.ntotal))
        if k <= 0:
            return []
        qv = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self._index.search(qv, k)
        out: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            out.append((int(idx), float(score)))
        return out

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        corpus_tokens: list[list[str]] = []
        for row in self._metadata:
            toks = _tokenize(str(row.get("text", "")))
            corpus_tokens.append(toks if toks else ["emptydoc"])
        self._bm25 = BM25Okapi(corpus_tokens)

    def search_hybrid(
        self,
        query_full: str,
        query_entity_focus: str,
        final_k: int,
    ) -> list[tuple[int, float]]:
        """
        BM25 + dense(entity) + dense(full) fused with RRF; returns (index, rrf_score).
        """
        self.ensure_loaded()
        assert self._model is not None and self._index is not None
        ntotal = int(self._index.ntotal)
        if ntotal == 0 or final_k <= 0:
            return []

        branch = min(GRAPH_RAG_DENSE_BRANCH_K, ntotal)
        bm25_take = min(GRAPH_RAG_BM25_BRANCH_K, ntotal)

        self._ensure_bm25()
        assert self._bm25 is not None

        q_tok = _tokenize(query_full)
        if len(q_tok) < 2:
            q_tok = _tokenize(query_entity_focus) or ["query"]
        bm25_scores = self._bm25.get_scores(q_tok)
        bm25_order = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:bm25_take]

        dense_full_order = [i for i, _ in self.search(query_full, branch)]
        ent_q = (query_entity_focus or "").strip()
        full_q = (query_full or "").strip()
        rankings: list[list[int]] = [bm25_order]
        if ent_q and ent_q != full_q:
            rankings.append([i for i, _ in self.search(ent_q, branch)])
        rankings.append(dense_full_order)

        rrf_scores = _reciprocal_rank_fusion(rankings, GRAPH_RAG_RRF_K)
        ranked_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[
            :final_k
        ]
        if not ranked_docs:
            return self.search(query_full, final_k)
        return [(idx, rrf_scores[idx]) for idx in ranked_docs]


def _truncate(s: str, n: int = 4000) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def expand_neo4j(driver, kind: str, primary_id: str) -> str:
    """Return a compact markdown-ish block for one hit; empty if query fails."""
    lines: list[str] = []

    def run(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]

    try:
        if kind == "entity":
            rows = run(
                """
                MATCH (e:Entity {_id: $id})
                OPTIONAL MATCH (e)-[lr:LINKED_ENTITY]->(c:Entity)
                RETURN properties(e) AS props,
                  collect(DISTINCT coalesce(lr.slot, '') + ':' + coalesce(c._id, '')) AS linked_pairs
                """,
                {"id": primary_id},
            )
            if not rows:
                return ""
            r0 = rows[0]
            props = r0.get("props") or {}
            lines.append(f"**Entity** `{primary_id}`: {props.get('name', '')}  type={props.get('entityType')}  dataType={props.get('dataType')}")
            for lp in r0.get("linked_pairs") or []:
                if lp and isinstance(lp, str) and ":" in lp:
                    slot, cid = lp.split(":", 1)
                    if cid:
                        lines.append(f"  - linked {slot or '?'}: `{cid}`")

        elif kind == "pattern":
            rows = run(
                """
                MATCH (p:Pattern {_id: $id})
                OPTIONAL MATCH (p)-[:EXTRACTS_ENTITY]->(e:Entity)
                RETURN properties(p) AS props, collect(DISTINCT e._id) AS entities
                """,
                {"id": primary_id},
            )
            if not rows:
                return ""
            p = rows[0].get("props") or {}
            lines.append(
                f"**Pattern** `{primary_id}` ({p.get('source', 'kb')}): {p.get('name', '')}  type={p.get('type')}"
            )
            if p.get("regexPattern"):
                lines.append(f"  - regex: `{_truncate(str(p.get('regexPattern')), 500)}`")
            if p.get("stringPattern"):
                lines.append(f"  - string: `{_truncate(str(p.get('stringPattern')), 300)}`")
            ents = rows[0].get("entities") or []
            if ents:
                lines.append(f"  - extracts entities: {', '.join(str(x) for x in ents if x)}")

        elif kind == "rule":
            rows = run(
                """
                MATCH (r:Rule {_id: $id})
                OPTIONAL MATCH (r)-[:TARGETS_ENTITY]->(e:Entity)
                OPTIONAL MATCH (r)-[:USES_VALUE_PATTERN]->(vp:Pattern)
                OPTIONAL MATCH (r)-[:USES_KEY_PATTERN]->(kp:Pattern)
                OPTIONAL MATCH (r)-[:USES_KEY_ENTITY]->(ke:Entity)
                RETURN properties(r) AS props, e._id AS target_entity,
                  collect(DISTINCT vp._id) AS value_patterns, collect(DISTINCT kp._id) AS key_patterns,
                  collect(DISTINCT ke._id) AS key_entities
                """,
                {"id": primary_id},
            )
            if not rows:
                return ""
            r0 = rows[0]
            props = r0.get("props") or {}
            lines.append(
                f"**Rule** `{primary_id}`: {props.get('name', '')}  ruleType={props.get('ruleType')}  target_entity=`{r0.get('target_entity')}`"
            )
            lines.append(f"  - filters: {_truncate(props.get('filtersJson', ''), 800)}")
            lines.append(f"  - entity: {_truncate(props.get('entityJson', ''), 600)}")
            lines.append(f"  - value: {_truncate(props.get('valueJson', ''), 800)}")
            lines.append(f"  - key: {_truncate(props.get('keyJson', ''), 800)}")
            vp = [x for x in (r0.get("value_patterns") or []) if x]
            kp = [x for x in (r0.get("key_patterns") or []) if x]
            ke = [x for x in (r0.get("key_entities") or []) if x]
            if vp:
                lines.append(f"  - value patterns: {', '.join(vp)}")
            if kp:
                lines.append(f"  - key patterns: {', '.join(kp)}")
            if ke:
                lines.append(f"  - key entities: {', '.join(ke)}")

        elif kind == "template":
            tid = int(primary_id) if str(primary_id).isdigit() else primary_id
            rows = run(
                """
                MATCH (t:Template)
                WHERE t.nerTemplateId = $tid
                OPTIONAL MATCH (t)-[:HAS_ENTITY_SETTING]->(tes)
                OPTIONAL MATCH (tes)-[:HAS_NER_RULE]->(nr:NerRule)
                RETURN properties(t) AS props,
                  collect(DISTINCT tes.name) AS setting_names,
                  collect(DISTINCT nr.name) AS rule_names,
                  collect(DISTINCT nr.nerRuleType) AS rule_types
                """,
                {"tid": tid},
            )
            if not rows:
                return ""
            r0 = rows[0]
            props = r0.get("props") or {}
            lines.append(f"**Template** nerTemplateId={props.get('nerTemplateId')}: {props.get('name', '')}")
            sn = [x for x in (r0.get("setting_names") or []) if x]
            rn = [x for x in (r0.get("rule_names") or []) if x]
            rt = [x for x in (r0.get("rule_types") or []) if x]
            if sn:
                lines.append(f"  - settings: {', '.join(sn[:25])}")
            if rn:
                lines.append(f"  - ner rules: {', '.join(rn[:25])}")
            if rt:
                lines.append(f"  - rule types: {', '.join(rt[:15])}")

        elif kind == "ner_rule":
            rows = run(
                """
                MATCH (nr:NerRule {stableId: $sid})
                OPTIONAL MATCH (nr)-[:VALUE_MATCH_SETTING]->(vms:NerMatchSetting)
                OPTIONAL MATCH (nr)-[:KEY_MATCH_SETTING]->(kms:NerMatchSetting)
                RETURN properties(nr) AS props,
                  collect(DISTINCT vms.stableId) AS vms,
                  collect(DISTINCT kms.stableId) AS kms
                """,
                {"sid": primary_id},
            )
            if not rows:
                return ""
            r0 = rows[0]
            p = r0.get("props") or {}
            lines.append(
                f"**NerRule** `{primary_id}`: {p.get('name', '')}  type={p.get('nerRuleType')}  template={p.get('nerTemplateId')}"
            )
            if p.get("valueMatchSettingJson"):
                lines.append(f"  - value match: {_truncate(p.get('valueMatchSettingJson', ''), 1200)}")
            if p.get("keyMatchSettingJson"):
                lines.append(f"  - key match: {_truncate(p.get('keyMatchSettingJson', ''), 1200)}")

        else:
            return ""
    except Neo4jError as e:
        logger.warning("Graph RAG expand failed for %s %s: %s", kind, primary_id, e)
        return ""

    return "\n".join(lines)


def split_retrieval_query(retrieval_query: str) -> tuple[str, str]:
    """
    Split a query from build_retrieval_query into (full, entity_focused_head).
    The head excludes the long OCR block so dense retrieval can emphasize entities/hints.
    """
    marker = "Document excerpt:"
    q = (retrieval_query or "").strip()
    if marker in q:
        head = q.split(marker, 1)[0].strip()
        return q, head if head else q
    return q, q


def build_retrieval_query(
    entity_names: list[str],
    hints: str,
    doc_snippet: str,
    *,
    doc_max_chars: int | None = None,
) -> str:
    max_chars = doc_max_chars if doc_max_chars is not None else GRAPH_RAG_DOC_SNIPPET_CHARS
    parts = [f"Entities: {', '.join(entity_names)}."]
    if hints.strip():
        parts.append(f"Hints: {hints.strip()[:2000]}")
    sn = re.sub(r"\s+", " ", doc_snippet)[:max_chars]
    if sn:
        parts.append(f"Document excerpt: {sn}")
    return "\n".join(parts)


def run_graph_rag(
    *,
    index_dir: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    retrieval_query: str,
    vector_k: int,
    max_context_chars: int,
    expand_neo4j_graph: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns (markdown context string, hit dicts for API/debug).
    If Neo4j password is empty, uses vector metadata text only (no expansion).
    """
    store = GraphRagStore(index_dir)
    if not store.available():
        raise FileNotFoundError(f"No vector index at {index_dir} (run graph-db/scripts/vector_index.py build).")

    store.ensure_loaded()
    q_full, q_entity_focus = split_retrieval_query(retrieval_query)

    if GRAPH_RAG_HYBRID:
        hits_raw = store.search_hybrid(
            query_full=q_full,
            query_entity_focus=q_entity_focus,
            final_k=vector_k,
        )
        logger.debug(
            "Graph RAG hybrid: RRF fusion (BM25 + dense entity + dense full), k=%s",
            vector_k,
        )
    else:
        hits_raw = store.search(q_full, vector_k)
        logger.debug("Graph RAG dense-only: single cosine query, k=%s", vector_k)

    driver = None
    if expand_neo4j_graph and neo4j_password:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    hit_rows: list[dict[str, Any]] = []
    blocks: list[str] = []

    for rank, (idx, score) in enumerate(hits_raw, start=1):
        if idx >= len(store._metadata):
            continue
        row = store._metadata[idx]
        kind = row.get("kind", "")
        pid = str(row.get("primary_id", ""))
        embed_text = str(row.get("text", ""))

        expanded = ""
        if driver:
            expanded = expand_neo4j(driver, kind, pid)
        if not expanded.strip():
            expanded = _truncate(embed_text, 3500)

        metric = "rrf" if GRAPH_RAG_HYBRID else "cos_ip"
        block = f"### Hit {rank} ({metric}={score:.4f}, kind={kind}, id=`{pid}`)\n{expanded}"
        blocks.append(block)
        hit_rows.append(
            {
                "vector_index": idx,
                "score": round(score, 6),
                "score_metric": metric,
                "kind": kind,
                "primary_id": pid,
            }
        )

    if driver:
        driver.close()

    text = "\n\n".join(blocks)
    if len(text) > max_context_chars:
        text = text[: max_context_chars - 20] + "\n…(truncated)"

    return text, hit_rows


def run_graph_rag_safe(
    *,
    index_dir: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    retrieval_query: str,
    vector_k: int,
    max_context_chars: int,
) -> tuple[str | None, list[dict[str, Any]], str]:
    """
    Never raises: returns (context or None, hits, error_message).
    Neo4j expansion is skipped if password is empty (metadata fallback still runs).
    """
    try:
        ctx, hits = run_graph_rag(
            index_dir=index_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            retrieval_query=retrieval_query,
            vector_k=vector_k,
            max_context_chars=max_context_chars,
            expand_neo4j_graph=bool(neo4j_password),
        )
        return ctx, hits, ""
    except Exception as e:
        logger.exception("Graph RAG retrieval failed")
        return None, [], str(e)
