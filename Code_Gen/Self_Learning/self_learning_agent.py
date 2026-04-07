from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Self-learning compares mapping rules to the template in one direction only:
# every rule-required message/IE must be satisfied; extra template content is allowed.
VALIDATION_POLICY: Dict[str, str] = {
    "direction": "rule_to_template_minimum",
    "messages": (
        "Each message in the matched mapping rule must appear in Call_Flow. "
        "Additional messages in the template are not flagged."
    ),
    "information_elements": (
        "Each IE listed in the mapping rule for a step must exist in Information_Elements and "
        "match the rule's ASN.1 structure (nested optional fields may be skipped). "
        "Additional IEs in the template are not flagged."
    ),
    "type_fields": (
        "Mapping 'type' entries are ASN.1 constructors and type references (structure), "
        "not concrete F1AP/RRC instance values."
    ),
    "human_resolution": (
        "Ambiguities are shown to a human. Template updates and prompt regeneration happen only "
        "after the user supplies resolutions (e.g. via resolve_self_learning_ambiguities). "
        "Automated checks do not apply template edits."
    ),
}


def _norm(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_ie_entries(node: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        if "IE_Name" in node:
            out.append(node)
        for v in node.values():
            out.extend(_iter_ie_entries(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_iter_ie_entries(item))
    return out


def _extract_template_message_map(template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    call_flow = template.get("Call_Flow", [])
    if not isinstance(call_flow, list):
        return out
    for msg in call_flow:
        if not isinstance(msg, dict):
            continue
        # Your filled templates use: { "Message": "...", "From": "...", "To": "..." }
        # Older variants might use Message_Name / name.
        name = str(msg.get("Message") or msg.get("Message_Name") or msg.get("name") or "").strip()
        if not name:
            continue
        out[_norm_token(name)] = msg
    return out


def _extract_template_ie_map(template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ie in _iter_ie_entries(template):
        name = str(ie.get("IE_Name") or "").strip()
        if not name:
            continue
        out[_norm_token(name)] = ie
    return out


def _norm_asn_token(text: str) -> str:
    """
    ASN.1-ish token normalization for fuzzy matching:
    - keep alphanumerics
    - drop punctuation/hyphens/underscores/spaces
    """
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _extract_ie_definition_text(ie_entry: Dict[str, Any]) -> str:
    return str((ie_entry or {}).get("IE_Definition") or "").strip()


def _definition_has_token(def_text: str, token: str) -> bool:
    if not def_text or not token:
        return False
    return _norm_asn_token(token) in _norm_asn_token(def_text)


_MAX_ASN_REF_DEPTH = 5


def _sequence_of_item_type_name(def_text: str) -> Optional[str]:
    """Extract Foo from 'SEQUENCE (SIZE(...)) OF Foo' or 'SEQUENCE OF Foo'."""
    if not def_text:
        return None
    m = re.search(
        r"SEQUENCE\s*(?:\([^)]*\))?\s*OF\s+([A-Za-z][A-Za-z0-9-]*)",
        def_text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _asn_type_after_field(def_text: str, field_name: str) -> Optional[str]:
    """Type following an ASN.1 component name (OCTET STRING, SEQUENCE OF T, or referenced type)."""
    if not def_text or not field_name:
        return None
    esc = re.escape(field_name)
    m = re.search(
        rf"\b{esc}\s+((?:OCTET\s+STRING|BIT\s+STRING|SEQUENCE\s+OF\s+[A-Za-z][A-Za-z0-9-]*|[A-Za-z][A-Za-z0-9-]*))",
        def_text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _lookup_ie_definition_text(ie_map: Dict[str, Dict[str, Any]], type_name: str) -> str:
    entry = ie_map.get(_norm_token(type_name))
    if isinstance(entry, dict):
        return _extract_ie_definition_text(entry)
    return ""


def _follow_typedef_chain_for_token(
    def_text: str, tok: str, ie_map: Dict[str, Dict[str, Any]], depth: int
) -> bool:
    """If ::= aliases another named type, follow until tok appears or depth limit."""
    if depth > _MAX_ASN_REF_DEPTH or not def_text:
        return False
    if _definition_has_token(def_text, tok):
        return True
    m = re.search(r"::=\s*([A-Za-z][A-Za-z0-9-]*)", def_text)
    if not m:
        return False
    alias = m.group(1)
    if alias.upper() in {
        "SEQUENCE",
        "CHOICE",
        "ENUMERATED",
        "INTEGER",
        "OCTET",
        "BIT",
        "NULL",
    }:
        return _definition_has_token(def_text, tok)
    sub = _lookup_ie_definition_text(ie_map, alias)
    if not sub:
        return False
    return _follow_typedef_chain_for_token(sub, tok, ie_map, depth + 1)


def _token_satisfied_for_mapping(
    def_text: str,
    field_name: str,
    tok: str,
    ie_map: Dict[str, Dict[str, Any]],
    depth: int,
) -> bool:
    """
    Mapping token (e.g. ENUMERATED, NRCGI) may appear in the container SEQUENCE, on the field's
    referenced type definition, or in a typedef chain — not always on the same line as the field.
    """
    if _definition_has_token(def_text, tok):
        return True
    ref = _asn_type_after_field(def_text, field_name)
    if not ref:
        return False
    ru = ref.upper()
    if tok.upper() == "OCTET STRING" and "OCTET" in ru and "STRING" in ru:
        return True
    if tok.upper() == "BIT STRING" and "BIT" in ru and "STRING" in ru:
        return True
    if tok.upper() == "SEQUENCE" and ru.startswith("SEQUENCE"):
        return True
    if tok.upper() == "CHOICE" and ru.startswith("CHOICE"):
        return True
    if tok.upper() == "ENUMERATED" and ru.startswith("ENUMERATED"):
        return True
    if tok.upper() == "INTEGER" and ru.startswith("INTEGER"):
        return True
    type_name = ref
    if ref.upper().startswith("SEQUENCE OF"):
        m_of = re.search(r"OF\s+([A-Za-z][A-Za-z0-9-]*)", ref, re.I)
        type_name = m_of.group(1).strip() if m_of else ref.split()[-1]
    sub = _lookup_ie_definition_text(ie_map, type_name)
    if sub:
        if _definition_has_token(sub, tok):
            return True
        if depth < _MAX_ASN_REF_DEPTH and _follow_typedef_chain_for_token(sub, tok, ie_map, depth + 1):
            return True
    return False


def _expected_type_tokens(expected: str) -> List[str]:
    """
    Convert mapping type strings into one or more tokens we should see in ASN.1.
    Examples:
      "ENUMERATED (complete)" -> ["ENUMERATED", "complete"]
      "CHOICE (...)" -> ["CHOICE"]
      "RRC IE (TS 38.331)" -> [] (spec notes, not ASN.1 tokens in F1AP)
    """
    s = str(expected or "").strip()
    if not s:
        return []
    s_u = s.upper()
    if "RRC IE" in s_u:
        # This is a cross-spec reference; don't require it in F1AP ASN.1.
        return []
    # Extract leading ASN.1 keywords/types and referenced type names.
    tokens: List[str] = []
    for t in ["BOOLEAN", "CHOICE", "SEQUENCE", "ENUMERATED", "INTEGER", "OCTET STRING", "BIT STRING", "OPTIONAL", "LIST"]:
        if t in s_u:
            # Keep canonical token
            tokens.append(t)
    # Also capture any explicit type identifiers (e.g., NRCGI, LTMConfigurationID)
    for m in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", s):
        # Skip generic words
        if m.upper() in {"LIST", "OF", "OPTIONAL", "CHOICE", "SEQUENCE", "ENUMERATED", "BOOLEAN", "INTEGER", "OCTET", "STRING", "BIT"}:
            continue
        tokens.append(m)
    # De-dup, prefer longer first
    seen = set()
    out = []
    for tok in sorted(tokens, key=len, reverse=True):
        k = tok.upper()
        if k in seen:
            continue
        seen.add(k)
        out.append(tok)
    return out


def _definition_has_field_and_type(
    def_text: str,
    field_name: str,
    expected_type: Any,
    ie_map: Optional[Dict[str, Dict[str, Any]]] = None,
    _depth: int = 0,
) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    - Field must appear in this IE_Definition or, for SEQUENCE OF T, in the item type T if T is in ie_map.
    - Type tokens may be satisfied in a referenced ASN.1 type (e.g. lTMIndicator LTMIndicator -> LTMIndicator ::= ENUMERATED).
    """
    if _depth > _MAX_ASN_REF_DEPTH:
        return False, "ASN.1 reference chain too deep"
    if not def_text:
        return False, "IE_Definition is empty"

    ie_map = ie_map or {}
    field_ok = bool(field_name and _definition_has_token(def_text, field_name))
    if not field_ok:
        item_t = _sequence_of_item_type_name(def_text)
        if item_t:
            nested = _lookup_ie_definition_text(ie_map, item_t)
            if nested:
                return _definition_has_field_and_type(
                    nested, field_name, expected_type, ie_map, _depth + 1
                )
        return False, f"field '{field_name}' not found in IE_Definition"

    if isinstance(expected_type, dict):
        return True, "ok"

    if not isinstance(expected_type, str):
        return True, "ok"

    tokens = _expected_type_tokens(expected_type)
    for tok in tokens:
        if _token_satisfied_for_mapping(def_text, field_name, tok, ie_map, _depth):
            continue
        return False, f"expected type/token '{tok}' not found for field '{field_name}'"
    return True, "ok"


def _flatten_expected_ie_fields(expected_ie: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """
    Flatten mapping IE structures into (field_path, expected_type) pairs.
    Example:
      { "cSIResourceConfiguration": { "ltm-CSIResource...": "RRC IE ..." } }
    becomes:
      ("cSIResourceConfiguration", dict(...)) and ("ltm-CSIResource...", "RRC IE ...")
    """
    out: List[Tuple[str, Any]] = []
    if isinstance(expected_ie, dict):
        for k, v in expected_ie.items():
            key = str(k)
            if prefix:
                key = f"{prefix}.{key}"
            out.append((key, v))
            out.extend(_flatten_expected_ie_fields(v, prefix=key))
    return out


def _normalize_field_spec(fspec: Any) -> Any:
    """Map one field spec from the rich rules schema to string or nested dict for validation."""
    if not isinstance(fspec, dict):
        return fspec
    nested = fspec.get("fields")
    if nested and isinstance(nested, dict):
        inner: Dict[str, Any] = {}
        for nk, nv in nested.items():
            inner[nk] = _normalize_field_spec(nv)
        return inner
    typ = str(fspec.get("type", "")).strip()
    rrc = str(fspec.get("rrc_ie", "")).strip()
    if rrc:
        return f"{typ} ({rrc})" if typ else rrc
    if typ:
        return typ
    stripped = {k: _normalize_field_spec(v) for k, v in fspec.items() if k not in {"optional"}}
    return stripped if stripped else fspec


def _normalize_ie_spec(spec: Any) -> Any:
    """
    Normalize IE expectation: { optional, fields } or { type, optional } -> flat dict or type string.
    """
    if not isinstance(spec, dict):
        return spec
    if "fields" in spec and isinstance(spec["fields"], dict):
        out: Dict[str, Any] = {}
        for fname, fspec in spec["fields"].items():
            out[fname] = _normalize_field_spec(fspec)
        return out
    if "type" in spec:
        return str(spec.get("type", ""))
    return {k: _normalize_ie_spec(v) for k, v in spec.items() if k != "optional"}


def _normalize_ies_dict(ies: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _normalize_ie_spec(v) for k, v in ies.items()}


def _optional_paths_under_field(fspec: Any, path_prefix: str) -> Set[str]:
    """Collect dot-paths of nested fields marked optional: true under a field spec."""
    paths: Set[str] = set()
    if not isinstance(fspec, dict):
        return paths
    nested = fspec.get("fields")
    if not isinstance(nested, dict):
        return paths
    for nk, nv in nested.items():
        sub = f"{path_prefix}.{nk}" if path_prefix else str(nk)
        if isinstance(nv, dict):
            if nv.get("optional") is True:
                paths.add(sub)
            paths |= _optional_paths_under_field(nv, sub)
    return paths


def _optional_nested_field_paths_for_raw_ie(raw_ie_spec: Any) -> Set[str]:
    """Dot-paths (relative to the IE) for optional nested fields from rich mapping schema."""
    paths: Set[str] = set()
    if not isinstance(raw_ie_spec, dict):
        return paths
    fields = raw_ie_spec.get("fields")
    if not isinstance(fields, dict):
        return paths
    for fname, fspec in fields.items():
        if not isinstance(fspec, dict):
            continue
        if fspec.get("optional") is True:
            paths.add(str(fname))
        paths |= _optional_paths_under_field(fspec, str(fname))
    return paths


def _optional_fields_by_ie_from_raw(raw_ies: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ie_name, raw_spec in raw_ies.items():
        p = _optional_nested_field_paths_for_raw_ie(raw_spec)
        if p:
            out[str(ie_name)] = sorted(p)
    return out


def _field_path_under_optional_prefix(field_path: str, optional_rel_paths: Set[str]) -> bool:
    """True if this path or any of its parent path prefixes is marked optional in the mapping."""
    if not field_path or not optional_rel_paths:
        return False
    if field_path in optional_rel_paths:
        return True
    parts = field_path.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix in optional_rel_paths:
            return True
    return False


def _prepare_rule_for_validation(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy rule and convert messages/description + rich IE schema for ambiguity checks."""
    r: Dict[str, Any] = json.loads(json.dumps(rule))
    if not r.get("intent_description") and r.get("description"):
        r["intent_description"] = r["description"]
    flow = r.get("message_flow") or r.get("messages", []) or []
    r["message_flow"] = flow
    for mf in flow:
        if not isinstance(mf, dict):
            continue
        raw_ies = mf.get("ies")
        if isinstance(raw_ies, dict):
            # IE presence is mandatory; only nested-field optionality is honored.
            opt_fields = _optional_fields_by_ie_from_raw(raw_ies)
            if opt_fields:
                mf["_optional_fields_by_ie"] = opt_fields
            mf["ies"] = _normalize_ies_dict(raw_ies)
    return r


def _derive_message_ie_container_name(message_name: str) -> str:
    """
    "UE CONTEXT SETUP REQUEST" -> "UEContextSetupRequestIEs"
    """
    words = re.findall(r"[A-Za-z0-9]+", message_name or "")
    camel = "".join(w.capitalize() for w in words if w)
    return f"{camel}IEs" if camel else ""


def _score_rule(intent: str, rule: Dict[str, Any]) -> int:
    intent_n = _norm(intent)
    desc = _norm(str(rule.get("intent_description") or rule.get("description") or ""))
    score = 0
    if desc and (desc in intent_n or intent_n in desc):
        score += 20

    flow = rule.get("message_flow") or rule.get("messages", []) or []
    for mf in flow:
        if not isinstance(mf, dict):
            continue
        msg = _norm(mf.get("msg_name", ""))
        if msg and msg in intent_n:
            score += 10
    return score


def _select_best_rule(intent: str, mapping: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    best_key: Optional[str] = None
    best_rule: Optional[Dict[str, Any]] = None
    best_score = -1
    for key, value in mapping.items():
        if key == "additional_info" or not isinstance(value, dict):
            continue
        score = _score_rule(intent, value)
        if score > best_score:
            best_score = score
            best_key = key
            best_rule = value
    if best_score <= 0:
        return None, None
    return best_key, best_rule


def _build_ambiguities(intent: str, template: Dict[str, Any], rule_key: str, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rule → template (minimum coverage): validate only what the mapping requires.
    Extra messages or IEs in the template never produce ambiguities.
    """
    ambiguities: List[Dict[str, Any]] = []
    msg_map = _extract_template_message_map(template)
    ie_map = _extract_template_ie_map(template)

    for idx, msg_rule in enumerate(rule.get("message_flow", []) or [], start=1):
        if not isinstance(msg_rule, dict):
            continue
        msg_name = str(msg_rule.get("msg_name", "")).strip()
        if not msg_name:
            continue
        msg_key = _norm_token(msg_name)
        tmpl_msg = msg_map.get(msg_key)
        if tmpl_msg is None:
            ambiguities.append(
                {
                    "id": f"missing_message_{idx}",
                    "type": "missing_message",
                    "intent_rule": rule_key,
                    "message_name": msg_name,
                    "question": (
                        f"Message '{msg_name}' required by rule '{rule_key}' is missing from Call_Flow. "
                        "Add it? (yes/no). Extra messages already in the template are fine and are not errors."
                    ),
                    "default_value": "yes",
                    "expected": msg_rule,
                    "expected_constraint": {
                        "kind": "call_flow_message",
                        "message_name": msg_name,
                        "sender": msg_rule.get("sender"),
                        "receiver": msg_rule.get("receiver"),
                    },
                }
            )
            continue

        exp_sender = str(msg_rule.get("sender", "")).strip()
        exp_receiver = str(msg_rule.get("receiver", "")).strip()
        detected_from = str(tmpl_msg.get("From", "") or "").strip()
        detected_to = str(tmpl_msg.get("To", "") or "").strip()
        # Fall back for legacy schema: parse Direction if From/To absent.
        if not detected_from and not detected_to:
            direction = str(tmpl_msg.get("Direction", "") or tmpl_msg.get("direction", "")).strip()
            direction_n = _norm(direction)
            detected_from = direction.split("->", 1)[0].strip() if "->" in direction else detected_from
            detected_to = direction.split("->", 1)[1].strip() if "->" in direction else detected_to
        if exp_sender and _norm(exp_sender) != _norm(detected_from):
            ambiguities.append(
                {
                    "id": f"sender_mismatch_{idx}",
                    "type": "sender_mismatch",
                    "intent_rule": rule_key,
                    "message_name": msg_name,
                    "question": f"Sender mismatch for '{msg_name}'. Provide sender value.",
                    "default_value": exp_sender,
                    "expected": exp_sender,
                    "detected": detected_from,
                }
            )
        if exp_receiver and _norm(exp_receiver) != _norm(detected_to):
            ambiguities.append(
                {
                    "id": f"receiver_mismatch_{idx}",
                    "type": "receiver_mismatch",
                    "intent_rule": rule_key,
                    "message_name": msg_name,
                    "question": f"Receiver mismatch for '{msg_name}'. Provide receiver value.",
                    "default_value": exp_receiver,
                    "expected": exp_receiver,
                    "detected": detected_to,
                }
            )

        ies = msg_rule.get("ies", {}) or {}
        opt_fields_by_ie = msg_rule.get("_optional_fields_by_ie", {})
        if not isinstance(opt_fields_by_ie, dict):
            opt_fields_by_ie = {}
        # Rule-required IEs must appear in the message IE container ASN.1 when we have container text.
        # Template IEs not listed in the rule are never flagged; extra container members are allowed.
        container_name = _derive_message_ie_container_name(msg_name)
        container_entry = ie_map.get(_norm_token(container_name)) or ie_map.get(_norm_token(msg_name.replace(" ", "")))
        container_def = _extract_ie_definition_text(container_entry) if isinstance(container_entry, dict) else ""

        for ie_name, expected_ie in (ies.items() if isinstance(ies, dict) else []):
            ie_key = _norm_token(ie_name)
            if ie_key not in ie_map:
                ambiguities.append(
                    {
                        "id": f"missing_ie_{idx}_{ie_key}",
                        "type": "missing_ie",
                        "intent_rule": rule_key,
                        "message_name": msg_name,
                        "ie_name": ie_name,
                        "question": (
                            f"Required IE '{ie_name}' for message '{msg_name}' (rule '{rule_key}') is missing from "
                            "Information_Elements. Add IE_Name and IE_Definition whose ASN.1 matches the mapping "
                            "(see expected_constraint; mapping types are structural, not PDU values)."
                        ),
                        "default_value": "",
                        "value_semantics": "Paste or author full ASN.1 IE_Definition text for this IE.",
                        "expected": expected_ie,
                        "expected_constraint": {
                            "kind": "asn1_ie_shape",
                            "ie_name": ie_name,
                            "message_name": msg_name,
                            "mapping_field_types": expected_ie,
                            "note": (
                                "mapping_field_types reflects ASN.1 constructors and type references from the rule, "
                                "not concrete protocol values to copy into the template."
                            ),
                        },
                    }
                )
                continue

            # Required rule IE must be referenced in message container ASN.1 when container text exists.
            if (
                container_def
                and not _definition_has_token(container_def, ie_name)
            ):
                ambiguities.append(
                    {
                        "id": f"ie_not_in_message_{idx}_{ie_key}",
                        "type": "ie_not_in_message",
                        "intent_rule": rule_key,
                        "message_name": msg_name,
                        "ie_name": ie_name,
                        "question": (
                            f"Rule requires IE '{ie_name}' to appear in the ASN.1 container for '{msg_name}' "
                            f"({container_name}), but it was not found there. Update the container IE_Definition "
                            "to include this member. Extra IEs elsewhere in the template are not an error."
                        ),
                        "default_value": "yes",
                        "expected": {"message": msg_name, "ie": ie_name, "container": container_name},
                        "expected_constraint": {
                            "kind": "ie_in_message_container",
                            "message_name": msg_name,
                            "ie_name": ie_name,
                            "container_ie": container_name,
                        },
                    }
                )

            # Deep validation: fields/types inside the IE definition
            ie_entry = ie_map.get(ie_key, {})
            ie_def = _extract_ie_definition_text(ie_entry) if isinstance(ie_entry, dict) else ""
            optional_rel_paths = {
                str(p) for p in (opt_fields_by_ie.get(ie_name) or [])
            }
            if isinstance(expected_ie, dict):
                for field_path, expected_type in _flatten_expected_ie_fields(expected_ie):
                    if _field_path_under_optional_prefix(field_path, optional_rel_paths):
                        continue
                    # Only validate leaf fields (string expected) and top-level direct fields.
                    field_name = field_path.split(".")[-1]
                    ok, reason = _definition_has_field_and_type(ie_def, field_name, expected_type, ie_map)
                    if not ok:
                        ambiguities.append(
                            {
                                "id": f"ie_field_mismatch_{idx}_{ie_key}_{_norm_token(field_path)}",
                                "type": "ie_field_mismatch",
                                "intent_rule": rule_key,
                                "message_name": msg_name,
                                "ie_name": ie_name,
                                "field": field_path,
                                "question": (
                                    f"IE '{ie_name}' IE_Definition does not match mapping structure for '{field_path}': {reason}. "
                                    "Revise ASN.1 so field names and constructors align with the rule (mapping types are shapes, not values)."
                                ),
                                "default_value": ie_def[:8000],
                                "value_semantics": "Starting point: current IE_Definition excerpt; replace with corrected ASN.1.",
                                "expected": {field_path: expected_type},
                                "expected_constraint": {
                                    "kind": "asn1_field_shape",
                                    "field_path": field_path,
                                    "expected_type": expected_type,
                                    "note": "expected_type is an ASN.1 shape hint from the mapping, not a value to paste as-is.",
                                },
                                "detected_ie_definition_preview": ie_def[:8000],
                            }
                        )
            elif isinstance(expected_ie, str):
                # Expected a simple type: ensure tokens exist in IE definition
                tokens = _expected_type_tokens(expected_ie)
                for tok in tokens:
                    if not _definition_has_token(ie_def, tok):
                        ambiguities.append(
                            {
                                "id": f"ie_type_mismatch_{idx}_{ie_key}_{_norm_token(tok)}",
                                "type": "ie_type_mismatch",
                                "intent_rule": rule_key,
                                "message_name": msg_name,
                                "ie_name": ie_name,
                                "question": (
                                    f"IE '{ie_name}' IE_Definition is missing ASN.1 token '{tok}' required by the mapping "
                                    f"type '{expected_ie}'. Update the definition to include that constructor or type (structural check, not a value)."
                                ),
                                "default_value": ie_def[:8000],
                                "value_semantics": "Starting point: current IE_Definition excerpt; edit ASN.1 to satisfy mapping shape.",
                                "expected": expected_ie,
                                "expected_constraint": {
                                    "kind": "asn1_ie_type_tokens",
                                    "mapping_type_string": expected_ie,
                                    "missing_token": tok,
                                },
                                "detected_ie_definition_preview": ie_def[:8000],
                            }
                        )
    for amb in ambiguities:
        amb["requires_human_resolution"] = True
    return ambiguities


def _coerce_resolution_value(value: Any) -> str:
    """Turn user/MCP resolution payloads into a string for template fields (ASN.1 text or JSON)."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normalize_resolutions(user_resolutions: Any) -> Dict[str, str]:
    if user_resolutions is None:
        return {}
    if isinstance(user_resolutions, dict):
        return {str(k): _coerce_resolution_value(v) for k, v in user_resolutions.items()}
    if isinstance(user_resolutions, list):
        out: Dict[str, str] = {}
        for item in user_resolutions:
            if not isinstance(item, dict):
                continue
            k = str(item.get("id", "")).strip()
            if not k:
                continue
            out[k] = _coerce_resolution_value(item.get("value"))
        return out
    return {}


def _apply_resolutions(
    template: Dict[str, Any],
    ambiguities: List[Dict[str, Any]],
    resolutions: Dict[str, str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    call_flow = template.get("Call_Flow")
    if not isinstance(call_flow, list):
        template["Call_Flow"] = []
        call_flow = template["Call_Flow"]
    info_elements = template.get("Information_Elements")
    if not isinstance(info_elements, list):
        template["Information_Elements"] = []
        info_elements = template["Information_Elements"]

    unresolved: List[Dict[str, Any]] = []
    msg_map = _extract_template_message_map(template)
    for amb in ambiguities:
        amb_id = str(amb.get("id", ""))
        value = resolutions.get(amb_id)
        if value is None or str(value).strip() == "":
            unresolved.append(amb)
            continue

        amb_type = amb.get("type")
        msg_name = str(amb.get("message_name", "")).strip()
        msg_key = _norm_token(msg_name)
        if amb_type == "missing_message":
            if str(value).strip().lower() in {"yes", "y", "true", "1"}:
                expected = amb.get("expected", {}) if isinstance(amb.get("expected"), dict) else {}
                sender = str(expected.get("sender", "")).strip()
                receiver = str(expected.get("receiver", "")).strip()
                call_flow.append(
                    {
                        "Message": msg_name,
                        "From": sender,
                        "To": receiver,
                    }
                )
                msg_map = _extract_template_message_map(template)
            continue

        if amb_type in {"sender_mismatch", "receiver_mismatch"}:
            msg = msg_map.get(msg_key)
            if not msg:
                unresolved.append(amb)
                continue
            if amb_type == "sender_mismatch":
                msg["From"] = str(value).strip()
            else:
                msg["To"] = str(value).strip()
            continue

        if amb_type == "missing_ie":
            ie_name = str(amb.get("ie_name", "")).strip()
            info_elements.append(
                {
                    "IE_Name": ie_name,
                    "IE_Definition": str(value).strip(),
                }
            )
            continue

        if amb_type in {"ie_field_mismatch", "ie_type_mismatch"}:
            ie_name = str(amb.get("ie_name", "")).strip()
            ie_key = _norm_token(ie_name)
            ie_map = _extract_template_ie_map(template)
            entry = ie_map.get(ie_key)
            if not isinstance(entry, dict):
                # If we can't find it, add as new.
                info_elements.append({"IE_Name": ie_name, "IE_Definition": str(value).strip()})
            else:
                entry["IE_Definition"] = str(value).strip()
            continue

        if amb_type == "ie_not_in_message":
            # This one is policy-only; no template edit needed.
            continue

    return template, unresolved


def _want_llm_ambiguity_review(use_llm_review: Optional[bool]) -> bool:
    if use_llm_review is False:
        return False
    if use_llm_review is True:
        return True
    from .llm_self_learning_review import default_use_llm_review_flag

    return default_use_llm_review_flag()


def validate_template_with_mapping_rules(
    *,
    intent: str,
    final_template_path: str,
    mapping_rules_path: str,
    user_resolutions: Any = None,
    output_dir: Optional[str] = None,
    use_llm_review: Optional[bool] = None,
) -> Dict[str, Any]:
    template_path = Path(final_template_path)
    rules_path = Path(mapping_rules_path)
    template = _load_json(template_path)
    mapping = _load_json(rules_path)

    rule_key, rule = _select_best_rule(intent, mapping)
    if not rule_key or not isinstance(rule, dict):
        return {
            "matched_rule": None,
            "has_ambiguities": False,
            "ambiguities": [],
            "resolved_template_path": str(template_path),
            "resolution_applied": False,
            "validation_policy": VALIDATION_POLICY,
            "llm_ambiguity_review": {"ran": False, "skipped_reason": "no_rule_matched"},
            "deterministic_ambiguity_count": 0,
            "human_resolution_required": False,
            "next_step_if_ambiguous": "",
            "message": "No mapping rule matched; skipped Self Learning validation.",
        }

    rule_work = _prepare_rule_for_validation(rule)
    ambiguities = _build_ambiguities(intent, template, rule_key, rule_work)

    working_ambiguities = ambiguities
    llm_amb_meta: Dict[str, Any] = {"ran": False}
    if not ambiguities:
        llm_amb_meta["skipped_reason"] = "no_ambiguities"
    elif not _want_llm_ambiguity_review(use_llm_review):
        llm_amb_meta["skipped_reason"] = "disabled_or_env"
    else:
        from .llm_self_learning_review import build_self_learning_llm, llm_review_self_learning_ambiguities

        _llm = build_self_learning_llm()
        if _llm is None:
            llm_amb_meta = {
                "ran": False,
                "skipped_reason": "azure_openai_not_configured",
            }
        else:
            working_ambiguities, llm_amb_meta = llm_review_self_learning_ambiguities(
                llm=_llm,
                intent=intent,
                rule_key=rule_key,
                rule=rule_work,
                template=template,
                ambiguities=ambiguities,
                validation_policy=VALIDATION_POLICY,
            )

    resolutions = _normalize_resolutions(user_resolutions)
    resolution_applied = bool(resolutions)
    unresolved = working_ambiguities
    resolved_template = template

    if resolutions:
        resolved_template, unresolved = _apply_resolutions(template, working_ambiguities, resolutions)

    resolved_template_path = str(template_path)
    if resolution_applied:
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir) if output_dir else (template_path.parent.parent / "self_learning_templates")
        out_path = out_dir / f"{template_path.stem}_self_learning_{ts}.json"
        _save_json(out_path, resolved_template)
        resolved_template_path = str(out_path)

    return {
        "matched_rule": rule_key,
        "has_ambiguities": len(unresolved) > 0,
        "ambiguities": unresolved,
        "resolution_applied": resolution_applied,
        "resolved_template_path": resolved_template_path,
        "validation_policy": VALIDATION_POLICY,
        "llm_ambiguity_review": llm_amb_meta,
        "deterministic_ambiguity_count": len(ambiguities),
        "human_resolution_required": len(unresolved) > 0,
        "next_step_if_ambiguous": (
            "Collect answers keyed by each ambiguity id; call validate_template_with_mapping_rules "
            "or pipeline resolve path with user_resolutions to write the template, then regenerate the code prompt."
        ),
    }
