"""
Load entities.json, patterns.json, rules.json, templates.json into Neo4j.

Uses the graph model documented in ../README.md. Handles JSON that is either
a single object or an array at the root.

Environment:
  NEO4J_URI      default bolt://127.0.0.1:7687
  NEO4J_USER     default neo4j
  NEO4J_PASSWORD required
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


def _as_list(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"Expected object or array, got {type(data)}")


def load_entities(tx, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {_id: row._id})
        SET e.name = row.name,
            e.entityType = row.entityType,
            e.dataType = row.dataType
        """,
        rows=rows,
    )


def load_patterns(tx, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (p:Pattern {_id: row._id})
        SET p.name = row.name,
            p.type = row.type,
            p.regexPattern = row.regexPattern,
            p.spacyPattern = coalesce(row.spacyPattern, [])
        """,
        rows=rows,
    )
    tx.run(
        """
        UNWIND $rows AS row
        UNWIND coalesce(row.entities, []) AS entityId
        MERGE (p:Pattern {_id: row._id})
        MERGE (e:Entity {_id: entityId})
        MERGE (p)-[:EXTRACTS_ENTITY]->(e)
        """,
        rows=rows,
    )


def _str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def load_rules(tx, rows: list[dict[str, Any]]) -> None:
    payload: list[dict[str, Any]] = []
    for r in rows:
        ent = r.get("entity") or {}
        val = r.get("value") or {}
        key = r.get("key") or {}
        payload.append(
            {
                "_id": r.get("_id"),
                "name": r.get("name"),
                "ruleType": r.get("ruleType"),
                "filters": json.dumps(r.get("filters"), ensure_ascii=False),
                "entityJson": json.dumps(r.get("entity"), ensure_ascii=False),
                "valueJson": json.dumps(r.get("value"), ensure_ascii=False),
                "keyJson": json.dumps(r.get("key"), ensure_ascii=False),
                "entityId": ent.get("entityId"),
                "valuePatterns": _str_list(val.get("valuePattern")),
                "keyPatterns": _str_list(key.get("keyPattern")),
            }
        )
    payload = [p for p in payload if p.get("_id") and p.get("entityId")]
    if not payload:
        return
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (rule:Rule {_id: row._id})
        SET rule.name = row.name,
            rule.ruleType = row.ruleType,
            rule.filtersJson = row.filters,
            rule.entityJson = row.entityJson,
            rule.valueJson = row.valueJson,
            rule.keyJson = row.keyJson
        WITH rule, row
        MERGE (e:Entity {_id: row.entityId})
        MERGE (rule)-[:TARGETS_ENTITY]->(e)
        """,
        rows=payload,
    )
    tx.run(
        """
        UNWIND $rows AS row
        UNWIND row.valuePatterns AS pid
        MERGE (rule:Rule {_id: row._id})
        MERGE (p:Pattern {_id: pid})
        MERGE (rule)-[:USES_VALUE_PATTERN]->(p)
        """,
        rows=payload,
    )
    tx.run(
        """
        UNWIND $rows AS row
        UNWIND row.keyPatterns AS pid
        MERGE (rule:Rule {_id: row._id})
        MERGE (p:Pattern {_id: pid})
        MERGE (rule)-[:USES_KEY_PATTERN]->(p)
        """,
        rows=payload,
    )


def load_templates(tx, templates: list[dict[str, Any]]) -> None:
    for t in templates:
        tid = t.get("nerTemplateId")
        if tid is None:
            continue
        tx.run(
            """
            MERGE (tpl:Template {nerTemplateId: $tid})
            SET tpl.name = $name
            """,
            tid=tid,
            name=t.get("name"),
        )
        for setting in t.get("nerTemplateEntitySettings") or []:
            sid = setting.get("nerTemplateEntitySettingId")
            if sid is None:
                continue
            stable_tes = f"{tid}:{sid}"
            tx.run(
                """
                MERGE (tpl:Template {nerTemplateId: $tid})
                MERGE (tes:TemplateEntitySetting {stableId: $stable})
                SET tes.nerTemplateEntitySettingId = $sid,
                    tes.name = $name,
                    tes.nerTemplateId = $tid
                MERGE (tpl)-[:HAS_ENTITY_SETTING]->(tes)
                """,
                tid=tid,
                stable=stable_tes,
                sid=sid,
                name=setting.get("name"),
            )
            for nr in setting.get("nerRules") or []:
                rid = nr.get("nerRuleId")
                if rid is None:
                    continue
                stable_rule = f"{tid}:{rid}"
                tx.run(
                    """
                    MERGE (tes:TemplateEntitySetting {stableId: $stableTes})
                    MERGE (rule:NerRule {stableId: $stableRule})
                    SET rule.nerRuleId = $rid,
                        rule.nerRuleType = $nrt,
                        rule.name = $rname,
                        rule.connectionType = $ct,
                        rule.direction = $dir,
                        rule.keyMatchSettingJson = $kms,
                        rule.valueMatchSettingJson = $vms
                    MERGE (tes)-[:HAS_NER_RULE]->(rule)
                    """,
                    stableTes=stable_tes,
                    stableRule=stable_rule,
                    rid=rid,
                    nrt=nr.get("nerRuleType"),
                    rname=nr.get("name"),
                    ct=nr.get("connectionType"),
                    dir=nr.get("direction"),
                    kms=json.dumps(nr.get("keyMatchSetting"), ensure_ascii=False),
                    vms=json.dumps(nr.get("valueMatchSetting"), ensure_ascii=False),
                )
                # value match setting + inline patterns
                vms = nr.get("valueMatchSetting")
                if isinstance(vms, dict):
                    ms_stable = f"{tid}:{rid}:value"
                    tx.run(
                        """
                        MERGE (rule:NerRule {stableId: $stableRule})
                        MERGE (ms:NerMatchSetting {stableId: $msStable})
                        SET ms.slot = 'value',
                            ms.nerMatchSettingType = $mst,
                            ms.dataField = $df,
                            ms.rawJson = $raw
                        MERGE (rule)-[:VALUE_MATCH_SETTING]->(ms)
                        """,
                        stableRule=stable_rule,
                        msStable=ms_stable,
                        mst=vms.get("nerMatchSettingType"),
                        df=vms.get("dataField"),
                        raw=json.dumps(vms, ensure_ascii=False),
                    )
                    for idx, pat in enumerate(vms.get("patterns") or []):
                        if not isinstance(pat, dict):
                            continue
                        pid = f"inline:{tid}:{rid}:v:{idx}"
                        tx.run(
                            """
                            MERGE (ms:NerMatchSetting {stableId: $msStable})
                            MERGE (p:Pattern {_id: $pid})
                            SET p.name = $pname,
                                p.type = coalesce($ptype, 'regex_pattern'),
                                p.regexPattern = $preg,
                                p.source = 'template_inline'
                            MERGE (ms)-[:INCLUDES_PATTERN]->(p)
                            """,
                            msStable=ms_stable,
                            pid=pid,
                            pname=pat.get("name"),
                            ptype=pat.get("type"),
                            preg=pat.get("pattern"),
                        )
                kms = nr.get("keyMatchSetting")
                if isinstance(kms, dict):
                    ms_stable = f"{tid}:{rid}:key"
                    tx.run(
                        """
                        MERGE (rule:NerRule {stableId: $stableRule})
                        MERGE (ms:NerMatchSetting {stableId: $msStable})
                        SET ms.slot = 'key',
                            ms.nerMatchSettingType = $mst,
                            ms.dataField = $df,
                            ms.rawJson = $raw
                        MERGE (rule)-[:KEY_MATCH_SETTING]->(ms)
                        """,
                        stableRule=stable_rule,
                        msStable=ms_stable,
                        mst=kms.get("nerMatchSettingType"),
                        df=kms.get("dataField"),
                        raw=json.dumps(kms, ensure_ascii=False),
                    )
                    for idx, pat in enumerate(kms.get("patterns") or []):
                        if not isinstance(pat, dict):
                            continue
                        pid = f"inline:{tid}:{rid}:k:{idx}"
                        tx.run(
                            """
                            MERGE (ms:NerMatchSetting {stableId: $msStable})
                            MERGE (p:Pattern {_id: $pid})
                            SET p.name = $pname,
                                p.type = coalesce($ptype, 'regex_pattern'),
                                p.regexPattern = $preg,
                                p.source = 'template_inline'
                            MERGE (ms)-[:INCLUDES_PATTERN]->(p)
                            """,
                            msStable=ms_stable,
                            pid=pid,
                            pname=pat.get("name"),
                            ptype=pat.get("type"),
                            preg=pat.get("pattern"),
                        )


def main() -> int:
    p = argparse.ArgumentParser(description="Load extraction KB JSON into Neo4j")
    p.add_argument("--entities", type=Path, help="entities.json path")
    p.add_argument("--patterns", type=Path, help="patterns.json path")
    p.add_argument("--rules", type=Path, help="rules.json path")
    p.add_argument("--templates", type=Path, help="templates.json path")
    args = p.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("NEO4J_PASSWORD is required", file=sys.stderr)
        return 1

    paths = [args.entities, args.patterns, args.rules, args.templates]
    if not any(paths):
        print("Provide at least one of --entities --patterns --rules --templates", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            if args.entities:
                data = json.loads(Path(args.entities).read_text(encoding="utf-8"))
                rows = _as_list(data)
                session.execute_write(load_entities, rows)
                print(f"entities: {len(rows)}")
            if args.patterns:
                data = json.loads(Path(args.patterns).read_text(encoding="utf-8"))
                rows = _as_list(data)
                session.execute_write(load_patterns, rows)
                print(f"patterns: {len(rows)}")
            if args.rules:
                data = json.loads(Path(args.rules).read_text(encoding="utf-8"))
                rows = _as_list(data)
                session.execute_write(load_rules, rows)
                print(f"rules: {len(rows)}")
            if args.templates:
                data = json.loads(Path(args.templates).read_text(encoding="utf-8"))
                rows = _as_list(data)
                session.execute_write(load_templates, rows)
                print(f"templates: {len(rows)}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
