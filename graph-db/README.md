# Graph DB migration (Neo4j)

Loads `entities.json`, `patterns.json`, `rules.json`, and `templates.json` into Neo4j with explicit relationships.

## Prerequisites

- Neo4j 5.x
- **APOC** plugin enabled (`apoc.load.json`, `apoc.convert.toJson`, `apoc.convert.fromJsonMap` for templates). Optional if you use the Python loader only.

## Graph model

| Node | Key property | Notes |
|------|----------------|------|
| `Entity` | `_id` | From `entities.json` |
| `Pattern` | `_id` | From `patterns.json` **or** synthetic `inline::…` from `04_load_templates.cypher` for template-embedded regex |
| `Rule` | `_id` | From `rules.json` (standalone rules) |
| `Template` | `nerTemplateId` | From `templates.json` |
| `TemplateEntitySetting` | `stableId` | `"{nerTemplateId}-tes-{nerTemplateEntitySettingId}"` |
| `NerRule` | `stableId` | `"{nerTemplateId}-nr-{nerRuleId}"` — template-embedded rules (separate from `Rule`) |
| `NerMatchSetting` | `stableId` | `"{nerTemplateId}-vms-{nerMatchSettingId}-nr-{nerRuleId}"` or `-kms-` for key side |

## Relationships

- `Entity -[:LINKED_ENTITY {slot}]-> Entity` — from compound `linkedEntity1` / `linkedEntity2` (see `01_load_entities.cypher`)
- `Pattern -[:EXTRACTS_ENTITY]-> Entity` — from `pattern.entities[]`
- `Rule -[:TARGETS_ENTITY]-> Entity` — from `rule.entity.entityId`
- `Rule -[:USES_VALUE_PATTERN]-> Pattern` — from `rule.value.valuePattern[]` when `value.valueType` is `pattern`
- `Rule -[:USES_KEY_PATTERN]-> Pattern` — from `rule.key.keyPattern[]` when `key.keyType` is `pattern`
- `Rule -[:USES_KEY_ENTITY]-> Entity` — from `rule.key.keyEntity[]` when present
- `Template -[:HAS_ENTITY_SETTING]-> TemplateEntitySetting`
- `TemplateEntitySetting -[:HAS_NER_RULE]-> NerRule`
- `NerRule -[:KEY_MATCH_SETTING]-> NerMatchSetting` / `NerRule -[:VALUE_MATCH_SETTING]-> NerMatchSetting` (when present)
- `NerMatchSetting -[:INCLUDES_PATTERN]-> Pattern` — each entry in `patterns[]` (inline regex becomes a `Pattern` with synthetic `_id`)

Standalone `Rule` nodes and template `NerRule` nodes are **different labels** because IDs live in different namespaces (`payment_date_webbank` vs numeric `nerRuleId`).

## Usage

### Option A — Cypher + APOC (entities, patterns, rules, templates)

1. Copy JSON files into Neo4j **import** directory and fix `file:///` paths in each script.
2. Run `cypher/00_constraints.cypher` once.
3. Run in order: `01_load_entities.cypher`, `02_load_patterns.cypher`, `03_load_rules.cypher`, `04_load_templates.cypher`.

JSON files must be **arrays** at the root (`[{...}, {...}]`). If you have a single object, wrap it in `[...]`.

**Templates:** `04_load_templates.cypher` loads in three phases (structure + JSON on `NerRule`, then value/key `NerMatchSetting` and inline `Pattern` nodes) so conditional subqueries do not drop rows.

### Option B — Python (recommended; handles templates + single-object JSON)

```bash
cd graph-db
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, then:

```bash
python scripts/load_json_to_neo4j.py --entities path\to\entities.json --patterns path\to\patterns.json --rules path\to\rules.json --templates path\to\templates.json
```

Paths can be omitted for any file you do not import yet.

## Linking standalone Rule to Template NerRule

If your data uses a shared key between `rules.json` and template `nerRules`, add a property on both and `MATCH` + `MERGE` that relationship in a follow-up migration. The samples you shared used different ID shapes; this repo does not assume a join key.
