// Run once per database. Neo4j 5 syntax.

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e._id IS UNIQUE;

CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS
FOR (p:Pattern) REQUIRE p._id IS UNIQUE;

CREATE CONSTRAINT rule_id_unique IF NOT EXISTS
FOR (r:Rule) REQUIRE r._id IS UNIQUE;

CREATE CONSTRAINT template_id_unique IF NOT EXISTS
FOR (t:Template) REQUIRE t.nerTemplateId IS UNIQUE;

CREATE CONSTRAINT template_entity_setting_stable IF NOT EXISTS
FOR (s:TemplateEntitySetting) REQUIRE s.stableId IS UNIQUE;

CREATE CONSTRAINT ner_rule_stable IF NOT EXISTS
FOR (r:NerRule) REQUIRE r.stableId IS UNIQUE;

CREATE CONSTRAINT ner_match_setting_stable IF NOT EXISTS
FOR (m:NerMatchSetting) REQUIRE m.stableId IS UNIQUE;
