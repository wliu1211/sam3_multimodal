"""
backend/services/activity.py

Phase 1.5 — Activity grouping and NLP label generation.

Takes a verified scene graph and produces a list of human-readable
ActivityGroup objects, each with a natural language label and the
set of node IDs that participate in that activity.

Two-pass grouping:
  Pass 1 — bucket edges by (predicate, object_label)
  Pass 2 — merge buckets that share the same subject node IDs
            (same people doing multiple related things = one activity)

Label generation:
  Template first — fast, deterministic, handles most cases
  Claude fallback — only for templates that produce awkward output
                    (near predicate, missing object, unusual combos)

Only activities involving at least one verified subject→predicate→object
relationship are returned; lone detections with no relationships are
intentionally suppressed to avoid noise in the activity list.
"""

import os
import re
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic
from services.graph import SceneGraph, Node

# ── Client ────────────────────────────────────────────────────────────────────

_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ActivityGroup:
    id:           str
    label:        str
    node_ids:     list[str]
    predicate:    str | None          # None for presence-only groups
    confidence:   float
    subject_count: int
    color:        str = "#4ADE80"     # assigned by renderer


# ── Predicates that always go through Claude label generation ─────────────────

# Templates produce unnatural output for these predicates — always use Claude.
ALWAYS_USE_CLAUDE = {"near", "parked_next_to"}

# ── Main entry point ──────────────────────────────────────────────────────────

async def build_activity_groups(
    scene_graph:  SceneGraph,
    context_tags: list[str],          # verbs/attrs from RAM++ for label context
    noun_tags:    list[str],           # noun concepts detected in image
) -> list[ActivityGroup]:
    """
    Converts a verified scene graph into a list of ActivityGroup objects
    with human-readable NLP labels.

    Returns groups sorted by confidence descending.
    """
    if not scene_graph.nodes:
        return []

    # ── Pass 1: bucket verified edges by (predicate, object_label) ────────────
    buckets: dict[str, dict] = {}   # key → {subjects, objects, edges, pred}

    for edge in scene_graph.verified_edges():
        subject = scene_graph.get_node(edge.subject_id)
        obj     = scene_graph.get_node(edge.object_id)
        if not subject or not obj:
            continue

        key = f"{edge.predicate}::{obj.label}"
        if key not in buckets:
            buckets[key] = {
                "predicate":    edge.predicate,
                "object_label": obj.label,
                "subjects":     [],
                "objects":      [],
                "confidences":  [],
            }
        if subject.id not in [n.id for n in buckets[key]["subjects"]]:
            buckets[key]["subjects"].append(subject)
        if obj.id not in [n.id for n in buckets[key]["objects"]]:
            buckets[key]["objects"].append(obj)
        buckets[key]["confidences"].append(edge.confidence)

    # ── Pass 2: merge buckets sharing identical subject sets ──────────────────
    merged = _merge_by_subjects(list(buckets.values()))

    # ── Pass 3: assign colors from palette ────────────────────────────────────
    # Standalone (presence-only) groups are intentionally excluded — only
    # activities involving a subject + predicate + object are surfaced.
    PALETTE = [
        "#4ADE80", "#F87171", "#60A5FA", "#FBBF24", "#E879F9",
        "#34D399", "#FB923C", "#A78BFA", "#F472B6", "#38BDF8",
    ]
    all_raw = merged
    groups: list[ActivityGroup] = []

    for i, raw in enumerate(all_raw):
        node_ids = list({n.id for n in raw["subjects"] + raw["objects"]})
        avg_conf = (sum(raw["confidences"]) / len(raw["confidences"])
                    if raw["confidences"] else 0.5)

        label = await _generate_label(raw, context_tags)

        groups.append(ActivityGroup(
            id            = f"group_{str(i).zfill(3)}",
            label         = label,
            node_ids      = node_ids,
            predicate     = raw.get("predicate"),
            confidence    = round(avg_conf, 4),
            subject_count = len(raw["subjects"]),
            color         = PALETTE[i % len(PALETTE)],
        ))

    return sorted(groups, key=lambda g: g.confidence, reverse=True)


def groups_to_dict(groups: list[ActivityGroup]) -> list[dict]:
    """Serialises groups for the API response."""
    return [
        {
            "id":           g.id,
            "label":        g.label,
            "node_ids":     g.node_ids,
            "predicate":    g.predicate,
            "confidence":   g.confidence,
            "subject_count": g.subject_count,
            "color":        g.color,
        }
        for g in groups
    ]


# ── Pass 2: subject merging ───────────────────────────────────────────────────

def _merge_by_subjects(buckets: list[dict]) -> list[dict]:
    """
    Merges buckets whose subject sets are identical or subsets of each other.

    Example: person_001 + person_002 appear in both:
      "holding::racket"    → {person_001, person_002} + {racket_001, racket_002}
      "near::pickleball"   → {person_001, person_002} + {pickleball_001}

    These merge into one group covering all four objects, allowing Claude to
    generate "2 people playing pickleball with rackets" instead of two
    separate items.
    """
    if not buckets:
        return []

    merged:   list[dict]      = []
    used:     set[int]        = set()

    for i, a in enumerate(buckets):
        if i in used:
            continue
        combined = dict(a)
        combined["subjects"]    = list(a["subjects"])
        combined["objects"]     = list(a["objects"])
        combined["confidences"] = list(a["confidences"])
        combined["predicates"]  = [a["predicate"]]
        combined["object_labels"] = [a["object_label"]]

        subject_ids_a = {n.id for n in a["subjects"]}

        for j, b in enumerate(buckets):
            if j <= i or j in used:
                continue
            subject_ids_b = {n.id for n in b["subjects"]}

            # Merge if subject sets overlap significantly (>50%)
            overlap = len(subject_ids_a & subject_ids_b)
            if overlap / max(len(subject_ids_a), len(subject_ids_b), 1) >= 0.5:
                used.add(j)
                # Add any new subjects/objects not already present
                existing_ids = {n.id for n in combined["subjects"]}
                for n in b["subjects"]:
                    if n.id not in existing_ids:
                        combined["subjects"].append(n)
                existing_obj_ids = {n.id for n in combined["objects"]}
                for n in b["objects"]:
                    if n.id not in existing_obj_ids:
                        combined["objects"].append(n)
                combined["confidences"].extend(b["confidences"])
                if b["predicate"] not in combined["predicates"]:
                    combined["predicates"].append(b["predicate"])
                if b["object_label"] not in combined["object_labels"]:
                    combined["object_labels"].append(b["object_label"])

        used.add(i)
        merged.append(combined)

    return merged


# ── Label generation ──────────────────────────────────────────────────────────

async def _generate_label(raw: dict, context_tags: list[str]) -> str:
    """
    Generates a natural language label for an activity group.
    Tries template first; calls Claude only if the result is awkward.
    """
    label = _template_label(raw)

    if _needs_claude(label, raw):
        label = await _claude_label(raw, context_tags, label)

    return label


def _template_label(raw: dict) -> str:
    """
    Builds a label from the group structure using simple rules.

    Examples:
      2 subjects, predicate=holding, objects=[racket, racket]
        → "2 people holding rackets"
      3 subjects, predicates=[holding, near], objects=[racket, pickleball]
        → "3 people holding rackets near a pickleball"
    """
    subjects      = raw["subjects"]
    objects       = raw["objects"]
    predicates    = raw.get("predicates", [raw.get("predicate")] if raw.get("predicate") else [])
    object_labels = raw.get("object_labels", [raw.get("object_label")] if raw.get("object_label") else [])

    n_sub = len(subjects)
    sub_label = subjects[0].label if subjects else "object"
    sub_plural = _pluralise(sub_label, n_sub)
    sub_prefix = f"{n_sub} " if n_sub > 1 else "1 "

    # Build predicate + object phrase
    rel_parts = []
    for pred, obj_label in zip(predicates, object_labels):
        if pred and obj_label:
            n_obj = sum(1 for o in objects if o.label == obj_label)
            obj_plural = _pluralise(obj_label, n_obj)
            rel_parts.append(f"{pred.replace('_', ' ')} {obj_plural}")

    if not rel_parts:
        return f"{sub_prefix}{sub_plural}"

    relation = " and ".join(rel_parts)
    return f"{sub_prefix}{sub_plural} {relation}"


def _needs_claude(label: str, raw: dict) -> bool:
    """
    Returns True when the template label is awkward and Claude should refine it.
    """
    predicates = raw.get("predicates", [])

    # Always use Claude for "near" and "parked_next_to" — templates sound flat
    if any(p in ALWAYS_USE_CLAUDE for p in predicates if p):
        return True

    # Template produced placeholder text
    if "None" in label or "unknown" in label:
        return True

    # Merged group with multiple predicates — Claude can write a better sentence
    if len([p for p in predicates if p]) >= 2:
        return True

    # Awkward pluralisation fallback triggered
    if "persons" in label:
        return True

    return False


async def _claude_label(raw: dict, context_tags: list[str], template: str) -> str:
    """
    Calls Claude to produce a natural one-line activity description.
    Provides the template as a hint so Claude can improve rather than invent.
    """
    subjects      = raw["subjects"]
    predicates    = [p for p in raw.get("predicates", []) if p]
    object_labels = [o for o in raw.get("object_labels", []) if o]

    data_summary = {
        "subject_count": len(subjects),
        "subject_label": subjects[0].label if subjects else "object",
        "predicates":    predicates,
        "object_labels": object_labels,
        "context_words": context_tags[:5],   # limit to avoid prompt bloat
        "template_draft": template,
    }

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=40,
            messages=[{
                "role": "user",
                "content": f"""Rewrite this activity description in natural English.
Max 10 words. Be specific and grammatically natural. Output only the description.

Data: {data_summary}
Draft: {template}"""
            }]
        )
        result = response.content[0].text.strip().strip('"').strip("'")
        # Sanity check — if Claude returns something very long, fall back to template
        if len(result.split()) > 12:
            return template
        return result
    except Exception as e:
        print(f"[activity] Claude label generation failed: {e}")
        return template


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pluralise(word: str, count: int) -> str:
    """Simple English pluralisation for common object labels."""
    if count <= 1:
        return word

    IRREGULARS = {
        "person": "people", "man": "men", "woman": "women",
        "child": "children", "mouse": "mice", "goose": "geese",
        "tooth": "teeth", "foot": "feet", "ox": "oxen",
        "leaf": "leaves", "knife": "knives", "life": "lives",
        "half": "halves", "shelf": "shelves", "wolf": "wolves",
    }
    if word in IRREGULARS:
        return IRREGULARS[word]

    # Words ending in s, x, z, ch, sh → add "es"
    if re.search(r"(s|x|z|ch|sh)$", word):
        return word + "es"
    # Words ending in consonant + y → replace y with ies
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"

    return word + "s"