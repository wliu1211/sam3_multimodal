import os
import json
from anthropic import AsyncAnthropic

# ── Client ────────────────────────────────────────────────────────────────────

_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── System prompt ─────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """
You are a vision query parser. Extract ALL target concepts from the prompt.

Return a JSON object with EXACTLY this schema:
{
  "targets": [
    {
      "base_class": string,      // simple noun: "dog", "ball", "bird", "house"
      "attributes": {
        "color":  string | null,
        "make":   string | null,
        "type":   string | null,
        "other":  array of string
      },
      "confidence_threshold": float,
      "role": string             // "primary" | "related"
                                 // primary = what the user mainly wants highlighted
                                 // related = context objects for relationship edges
    }
  ],
  "relational_hints": array of string  // e.g. ["holding", "on", "near", "with"]
}

Rules:
- Extract EVERY distinct PHYSICAL OBJECT mentioned, not just the most prominent.
- For "dog and ball": targets = [dog (primary), ball (related)]
- For "monkey with banana": targets = [monkey (primary), banana (related)]
- For "bird on tree": targets = [bird (primary), tree (related)]
- For "house with car": targets = [house (primary), car (related)]
- For "blue Honda": targets = [car (primary)] with color=blue, make=Honda

COMPOSITIONAL QUERIES — when the prompt names a PRIMARY subject together with
one or more COMPANION objects, ALL companion objects become "related" targets.
The downstream system will only highlight primary instances that have EVERY
related class spatially nearby. This means:
- "man with paddle and umbrella" → man (primary), paddle (related), umbrella (related)
  Only men who have both a paddle AND an umbrella nearby will be shown.
- "person holding ball near bench" → person (primary), ball (related), bench (related)
- "cyclist with helmet and bike" → cyclist (primary), helmet (related), bike (related)

PERSON QUERIES — when the user only asks for people/persons/men/women with NO
companion objects (only clothing/appearance attributes), produce a SINGLE primary
target using base_class "person" regardless of the synonym used:
- "persons"          → [person (primary)]  — shows ALL people in the image
- "people"           → [person (primary)]  — shows ALL people in the image
- "men"              → [person (primary)]  — shows ALL men (use "man" as base_class)
- "show all people"  → [person (primary)]  — shows ALL people in the image

CRITICAL — clothing, colors, and appearance worn BY a person/subject are ATTRIBUTES
of that subject, NOT separate targets. Never create a target whose base_class is a
clothing item or body-part description when it is describing the main subject:
- "person with blue shirt"   → [person (primary)] with other=["wearing blue shirt"]
- "man in red jacket"        → [man (primary)] with other=["wearing red jacket"]
- "woman wearing glasses"    → [woman (primary)] with other=["wearing glasses"]
- "cyclist in yellow vest"   → [cyclist (primary)] with other=["wearing yellow vest"]
A new target is ONLY created when the second noun is an independent, separately
detectable physical object (banana, tree, car, ball, paddle, umbrella, etc.).

- Extract relational hints from prepositions and verbs between SEPARATE objects:
  "with", "on", "holding", "walking", "next to", "carrying", "sitting on" etc.
- Output ONLY valid JSON.

Examples:
  Input:  "find the monkey holding a banana"
  Output: {"targets":[{"base_class":"monkey","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"},{"base_class":"banana","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.50,"role":"related"}],"relational_hints":["holding"]}

  Input:  "man with paddle and umbrella"
  Output: {"targets":[{"base_class":"man","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"},{"base_class":"paddle","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.50,"role":"related"},{"base_class":"umbrella","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.50,"role":"related"}],"relational_hints":["with"]}

  Input:  "persons"
  Output: {"targets":[{"base_class":"person","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"}],"relational_hints":[]}

  Input:  "people"
  Output: {"targets":[{"base_class":"person","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"}],"relational_hints":[]}

  Input:  "find blue Honda cars"
  Output: {"targets":[{"base_class":"car","attributes":{"color":"blue","make":"Honda","type":null,"other":[]},"confidence_threshold":0.70,"role":"primary"}],"relational_hints":[]}

  Input:  "find birds on a tree"
  Output: {"targets":[{"base_class":"bird","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"},{"base_class":"tree","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.45,"role":"related"}],"relational_hints":["on"]}

  Input:  "find person with blue shirt"
  Output: {"targets":[{"base_class":"person","attributes":{"color":null,"make":null,"type":null,"other":["wearing blue shirt"]},"confidence_threshold":0.65,"role":"primary"}],"relational_hints":[]}

  Input:  "find the woman in a red jacket"
  Output: {"targets":[{"base_class":"woman","attributes":{"color":null,"make":null,"type":null,"other":["wearing red jacket"]},"confidence_threshold":0.65,"role":"primary"}],"relational_hints":[]}
"""

# ── Main parse function ───────────────────────────────────────────────────────

async def parse_prompt(prompt: str) -> dict:
    """
    Sends the raw user prompt to Claude and returns a structured dict.

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    Raises ValueError if Claude returns malformed JSON.

    Returns a dict matching this shape:
    {
        "targets": [
            {
                "base_class":           "monkey",
                "attributes":           {"color": None, "make": None, "type": None, "other": ["holding banana"]},
                "confidence_threshold": 0.65,
                "role":                 "primary",
            },
            {
                "base_class":           "banana",
                "attributes":           {"color": None, "make": None, "type": None, "other": []},
                "confidence_threshold": 0.50,
                "role":                 "related",
            },
        ],
        "relational_hints": ["holding"],
    }
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    raw = await _call_claude(prompt)
    return _validate_and_normalise(raw, prompt)


# ── Claude call ───────────────────────────────────────────────────────────────

async def _call_claude(prompt: str) -> str:
    """
    Calls claude-sonnet-4-6 with the parser system prompt.
    max_tokens raised to 512 — multi-target queries produce larger JSON.
    """
    response = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=PARSER_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.content[0].text.strip()


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_and_normalise(raw_json: str, original_prompt: str) -> dict:
    """
    Parses Claude's raw string output and validates the multi-target schema.
    Fills in safe defaults for any missing fields so the rest of the
    pipeline never has to handle KeyErrors.
    """
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude returned invalid JSON for prompt '{original_prompt}': {e}\n"
            f"Raw output was: {raw_json}"
        )

    raw_targets = data.get("targets", [])

    # Graceful fallback: if the model returned the old single-target shape, wrap it.
    if not raw_targets and "base_class" in data:
        raw_targets = [{
            "base_class":           data.get("base_class", "object"),
            "attributes":           data.get("attributes", {}),
            "confidence_threshold": data.get("confidence_threshold", 0.65),
            "role":                 "primary",
        }]

    if not raw_targets:
        raw_targets = [{"base_class": "object", "attributes": {}, "confidence_threshold": 0.65, "role": "primary"}]

    targets: list[dict] = []
    seen_primary = False
    for t in raw_targets:
        base_class = str(t.get("base_class", "object")).strip().lower() or "object"

        raw_attrs = t.get("attributes", {}) or {}
        attributes = {
            "color": _str_or_none(raw_attrs.get("color")),
            "make":  _str_or_none(raw_attrs.get("make")),
            "type":  _str_or_none(raw_attrs.get("type")),
            "other": [str(o) for o in (raw_attrs.get("other") or []) if o],
        }

        threshold = float(t.get("confidence_threshold") or 0.65)
        threshold = max(0.30, min(0.95, threshold))

        role = str(t.get("role") or "primary").lower()
        if role not in ("primary", "related"):
            role = "primary"
        # Ensure at least one primary target
        if role == "primary":
            seen_primary = True

        targets.append({
            "base_class":           base_class,
            "attributes":           attributes,
            "confidence_threshold": threshold,
            "role":                 role,
        })

    if not seen_primary and targets:
        targets[0]["role"] = "primary"

    relational_hints = [
        str(h).lower().strip()
        for h in (data.get("relational_hints") or [])
        if h
    ]

    return {
        "targets":          targets,
        "relational_hints": relational_hints,
    }


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None