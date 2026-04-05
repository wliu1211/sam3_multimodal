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
- Extract EVERY distinct object type mentioned, not just the most prominent.
- For "dog and ball": targets = [dog (primary), ball (related)]
- For "monkey with banana": targets = [monkey (primary), banana (related)]
- For "bird on tree": targets = [bird (primary), tree (related)]
- For "house with car": targets = [house (primary), car (related)]
- For "blue Honda": targets = [car (primary)] with color=blue, make=Honda
- Extract relational hints from prepositions and verbs: "with", "on", "holding",
  "walking", "next to", "carrying", "sitting on" etc.
- Output ONLY valid JSON.

Examples:
  Input:  "find the monkey holding a banana"
  Output: {"targets":[{"base_class":"monkey","attributes":{"color":null,"make":null,"type":null,"other":["holding banana"]},"confidence_threshold":0.65,"role":"primary"},{"base_class":"banana","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.50,"role":"related"}],"relational_hints":["holding"]}

  Input:  "find blue Honda cars"
  Output: {"targets":[{"base_class":"car","attributes":{"color":"blue","make":"Honda","type":null,"other":[]},"confidence_threshold":0.70,"role":"primary"}],"relational_hints":[]}

  Input:  "find birds on a tree"
  Output: {"targets":[{"base_class":"bird","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65,"role":"primary"},{"base_class":"tree","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.45,"role":"related"}],"relational_hints":["on"]}
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