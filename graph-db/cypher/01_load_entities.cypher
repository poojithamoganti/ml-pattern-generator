// entities.json MUST be a JSON array: [ { "_id": "...", ... }, ... ]
// APOC: place file under Neo4j import directory and allow file URLs.
// Phase 1: all Entity nodes. Phase 2: compound → LINKED_ENTITY edges (no CALL-subquery row loss).

// ----- Phase 1: properties on every entity row -----
CALL apoc.load.json('file:///entities.json') YIELD value AS rows
UNWIND rows AS row
MERGE (e:Entity {_id: row._id})
SET e.name = row.name,
    e.entityType = row.entityType,
    e.dataType = row.dataType,
    e.compoundEntityScope = row.compoundEntityScope;

// ----- Phase 2: compound linkedEntity1 / linkedEntity2 -----
CALL apoc.load.json('file:///entities.json') YIELD value AS rows
UNWIND rows AS row
WITH row WHERE row.linkedEntity1 IS NOT NULL
MATCH (e:Entity {_id: row._id})
MERGE (le1:Entity {_id: row.linkedEntity1._id})
SET le1.name = row.linkedEntity1.name,
    le1.dataType = row.linkedEntity1.dataType,
    le1.entityType = coalesce(row.linkedEntity1.entityType, 'single')
MERGE (e)-[:LINKED_ENTITY {slot: 'linkedEntity1'}]->(le1);

CALL apoc.load.json('file:///entities.json') YIELD value AS rows
UNWIND rows AS row
WITH row WHERE row.linkedEntity2 IS NOT NULL
MATCH (e:Entity {_id: row._id})
MERGE (le2:Entity {_id: row.linkedEntity2._id})
SET le2.name = row.linkedEntity2.name,
    le2.dataType = row.linkedEntity2.dataType,
    le2.entityType = coalesce(row.linkedEntity2.entityType, 'single')
MERGE (e)-[:LINKED_ENTITY {slot: 'linkedEntity2'}]->(le2);
