// templates.json MUST be a JSON array of template documents (same shape as template.json).
// APOC: place file under Neo4j import directory and allow file URLs.
// Phase 1: Template → TemplateEntitySetting → NerRule (JSON for match settings on the rule node).
// Phase 2–3: NerMatchSetting + inline Pattern from stored JSON (avoids CALL-subquery row-loss).

// ----- Phase 1: structure + JSON blobs on NerRule -----
CALL apoc.load.json('file:///templates.json') YIELD value AS rows
UNWIND rows AS trow
MERGE (t:Template {nerTemplateId: trow.nerTemplateId})
SET t.name = trow.name
WITH t, trow
UNWIND coalesce(trow.nerTemplateEntitySettings, []) AS tesRow
MERGE (tes:TemplateEntitySetting {stableId: toString(trow.nerTemplateId) + '-tes-' + toString(tesRow.nerTemplateEntitySettingId)})
SET tes.nerTemplateEntitySettingId = tesRow.nerTemplateEntitySettingId,
    tes.name = tesRow.name,
    tes.nerTemplateId = tesRow.nerTemplateId
MERGE (t)-[:HAS_ENTITY_SETTING]->(tes)
WITH t, trow, tes, tesRow
UNWIND coalesce(tesRow.nerRules, []) AS nrRow
MERGE (nr:NerRule {stableId: toString(trow.nerTemplateId) + '-nr-' + toString(nrRow.nerRuleId)})
SET nr.nerRuleId = nrRow.nerRuleId,
    nr.nerRuleType = nrRow.nerRuleType,
    nr.name = nrRow.name,
    nr.connectionType = nrRow.connectionType,
    nr.direction = nrRow.direction,
    nr.nerTemplateEntitySettingId = nrRow.nerTemplateEntitySettingId,
    nr.nerTemplateId = trow.nerTemplateId,
    nr.keyMatchSettingJson = CASE WHEN nrRow.keyMatchSetting IS NULL THEN NULL ELSE apoc.convert.toJson(nrRow.keyMatchSetting) END,
    nr.valueMatchSettingJson = CASE WHEN nrRow.valueMatchSetting IS NULL THEN NULL ELSE apoc.convert.toJson(nrRow.valueMatchSetting) END
MERGE (tes)-[:HAS_NER_RULE]->(nr);

// ----- Phase 2: value match settings + inline patterns -----
MATCH (nr:NerRule)
WHERE nr.valueMatchSettingJson IS NOT NULL
WITH nr,
     apoc.convert.fromJsonMap(nr.valueMatchSettingJson) AS vms,
     nr.nerTemplateId AS tid,
     nr.nerRuleId AS nrid
MERGE (ms:NerMatchSetting {stableId: toString(tid) + '-vms-' + toString(vms.nerMatchSettingId) + '-nr-' + toString(nrid)})
SET ms.nerMatchSettingId = vms.nerMatchSettingId,
    ms.nerMatchSettingType = vms.nerMatchSettingType,
    ms.dataField = vms.dataField,
    ms.matchSide = 'value',
    ms.json = nr.valueMatchSettingJson
MERGE (nr)-[:VALUE_MATCH_SETTING]->(ms)
WITH ms, vms, tid, nrid
UNWIND CASE WHEN vms.patterns IS NULL OR size(vms.patterns) = 0
  THEN []
  ELSE range(0, size(vms.patterns) - 1) END AS pidx
WITH ms, vms, tid, nrid, pidx, vms.patterns[pidx] AS patRow
MERGE (ip:Pattern {_id: 'inline::' + toString(tid) + '::vms::' + toString(vms.nerMatchSettingId) + '::nr::' + toString(nrid) + '::p::' + toString(pidx)})
SET ip.name = patRow.name,
    ip.pattern = patRow.pattern,
    ip.regexPattern = CASE WHEN toLower(coalesce(patRow.type, '')) = 'regex_pattern' THEN patRow.pattern ELSE NULL END,
    ip.type = 'regex',
    ip.source = 'template_inline'
MERGE (ms)-[:INCLUDES_PATTERN]->(ip);

// ----- Phase 3: key match settings + inline patterns -----
MATCH (nr:NerRule)
WHERE nr.keyMatchSettingJson IS NOT NULL
WITH nr,
     apoc.convert.fromJsonMap(nr.keyMatchSettingJson) AS kms,
     nr.nerTemplateId AS tid,
     nr.nerRuleId AS nrid
MERGE (ms:NerMatchSetting {stableId: toString(tid) + '-kms-' + toString(kms.nerMatchSettingId) + '-nr-' + toString(nrid)})
SET ms.nerMatchSettingId = kms.nerMatchSettingId,
    ms.nerMatchSettingType = kms.nerMatchSettingType,
    ms.dataField = kms.dataField,
    ms.matchSide = 'key',
    ms.json = nr.keyMatchSettingJson
MERGE (nr)-[:KEY_MATCH_SETTING]->(ms)
WITH ms, kms, tid, nrid
UNWIND CASE WHEN kms.patterns IS NULL OR size(kms.patterns) = 0
  THEN []
  ELSE range(0, size(kms.patterns) - 1) END AS pidx
WITH ms, kms, tid, nrid, pidx, kms.patterns[pidx] AS patRow
MERGE (ip:Pattern {_id: 'inline::' + toString(tid) + '-kms-' + toString(kms.nerMatchSettingId) + '::nr::' + toString(nrid) + '::p::' + toString(pidx)})
SET ip.name = patRow.name,
    ip.pattern = patRow.pattern,
    ip.regexPattern = CASE WHEN toLower(coalesce(patRow.type, '')) = 'regex_pattern' THEN patRow.pattern ELSE NULL END,
    ip.type = 'regex',
    ip.source = 'template_inline'
MERGE (ms)-[:INCLUDES_PATTERN]->(ip);
