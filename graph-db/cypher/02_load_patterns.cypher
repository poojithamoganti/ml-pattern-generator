// patterns.json MUST be a JSON array of pattern documents.

CALL apoc.load.json('file:///patterns.json') YIELD value AS rows
UNWIND rows AS row
MERGE (p:Pattern {_id: row._id})
SET p.name = row.name,
    p.type = row.type,
    p.regexPattern = row.regexPattern,
    p.stringPattern = row.stringPattern,
    p.stringPatternRegex = row.stringPatternRegex,
    p.spacyPattern = coalesce(row.spacyPattern, []),
    p.source = coalesce(p.source, 'kb')
WITH p, row
FOREACH (entityId IN CASE WHEN row.entities IS NULL THEN [] ELSE row.entities END |
  MERGE (e:Entity {_id: entityId})
  MERGE (p)-[:EXTRACTS_ENTITY]->(e)
);
