// Example read queries after import (run in Neo4j Browser).

// All pattern → entity links from patterns.json
// MATCH (p:Pattern)-[:EXTRACTS_ENTITY]->(e:Entity) RETURN p._id, e._id LIMIT 50;

// Standalone rules: rule → entity + value/key pattern refs
// MATCH (r:Rule)-[:TARGETS_ENTITY]->(e:Entity)
// OPTIONAL MATCH (r)-[:USES_VALUE_PATTERN]->(pv:Pattern)
// OPTIONAL MATCH (r)-[:USES_KEY_PATTERN]->(pk:Pattern)
// RETURN r._id, e._id, collect(DISTINCT pv._id), collect(DISTINCT pk._id) LIMIT 25;

// Template tree (one template id)
// MATCH (t:Template {nerTemplateId: 1091})-[:HAS_ENTITY_SETTING]->(tes)-[:HAS_NER_RULE]->(nr)
// OPTIONAL MATCH (nr)-[:VALUE_MATCH_SETTING|KEY_MATCH_SETTING]->(ms)-[:INCLUDES_PATTERN]->(pat)
// RETURN t, tes, nr, ms, pat LIMIT 100;

// Inline template patterns (synthetic ids)
// MATCH (p:Pattern) WHERE p.source = 'template_inline' RETURN p._id, p.name LIMIT 50;
