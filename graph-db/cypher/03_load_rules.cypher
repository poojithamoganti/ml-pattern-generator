// rules.json MUST be a JSON array of rule documents.

CALL apoc.load.json('file:///rules.json') YIELD value AS rows
UNWIND rows AS row
MERGE (r:Rule {_id: row._id})
SET r.name = row.name,
    r.ruleType = row.ruleType,
    r.filtersJson = apoc.convert.toJson(row.filters),
    r.entityJson = apoc.convert.toJson(row.entity),
    r.valueJson = apoc.convert.toJson(row.value),
    r.keyJson = apoc.convert.toJson(row.key)
WITH r, row
MERGE (e:Entity {_id: row.entity.entityId})
MERGE (r)-[:TARGETS_ENTITY]->(e)
WITH r, row
FOREACH (pid IN CASE WHEN row.value IS NULL OR row.value.valuePattern IS NULL THEN [] ELSE row.value.valuePattern END |
  MERGE (p:Pattern {_id: pid})
  MERGE (r)-[:USES_VALUE_PATTERN]->(p)
)
WITH r, row
FOREACH (pid IN CASE WHEN row.key IS NULL OR row.key.keyPattern IS NULL THEN [] ELSE row.key.keyPattern END |
  MERGE (p:Pattern {_id: pid})
  MERGE (r)-[:USES_KEY_PATTERN]->(p)
)
WITH r, row
FOREACH (eid IN CASE WHEN row.key IS NULL OR row.key.keyEntity IS NULL THEN [] ELSE row.key.keyEntity END |
  MERGE (ke:Entity {_id: eid})
  MERGE (r)-[:USES_KEY_ENTITY]->(ke)
);
