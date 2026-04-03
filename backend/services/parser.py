import os
import json
from anthropic import AsyncAnthropic

# ── Client ────────────────────────────────────────────────────────────────────

_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── System prompt ─────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """
You are a vision query parser. Your only job is to extract structured search
parameters from a natural language object detection prompt.

Given a user's description, return a JSON object with EXACTLY this schema:
{
  "base_class": string,        // The general object category to detect.
                               // Must be a simple noun: "car", "person", "dog".
                               // Never a compound phrase.
  "attributes": {
    "color":  string | null,   // Dominant color if mentioned, else null.
    "make":   string | null,   // Brand or manufacturer if mentioned, else null.
    "type":   string | null,   // Sub-type (sedan, SUV, retriever...), else null.
    "other":  array of string  // Any other visual attributes mentioned.
                               // Empty array [] if none.
  },
  "confidence_threshold": float  // How strict matching should be.
                                 // Use 0.70 for very specific prompts
                                 // (e.g. "blue Honda Civic").
                                 // Use 0.55 for vague prompts
                                 // (e.g. "big dogs").
                                 // Default 0.65.
}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- base_class must always be a single general noun.
- If the prompt mentions multiple object types, use the most prominent one.
- Normalise colors to lowercase single words: "blue", "red", "dark green".
- Normalise make/brand names to their standard form: "Honda", "Toyota", "Nike".

Examples:
  Input:  "find cars that are blue and a Honda"
  Output: {"base_class":"car","attributes":{"color":"blue","make":"Honda","type":null,"other":[]},"confidence_threshold":0.70}

  Input:  "show me people wearing red jackets"
  Output: {"base_class":"person","attributes":{"color":"red","make":null,"type":null,"other":["wearing jacket"]},"confidence_threshold":0.65}

  Input:  "find large brown dogs"
  Output: {"base_class":"dog","attributes":{"color":"brown","make":null,"type":null,"other":["large"]},"confidence_threshold":0.60}

  Input:  "find stop signs"
  Output: {"base_class":"stop sign","attributes":{"color":null,"make":null,"type":null,"other":[]},"confidence_threshold":0.65}
"""

# ── Main parse function ───────────────────────────────────────────────────────

async def parse_prompt(prompt: str) -> dict:
    """
    Sends the raw user prompt to Claude and returns a structured dict.

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    Raises ValueError if Claude returns malformed JSON.

    Returns a dict matching this shape:
    {
        "base_class": "car",
        "attributes": {
            "color": "blue",
            "make":  "Honda",
            "type":  None,
            "other": []
        },
        "confidence_threshold": 0.70
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
    Low max_tokens since output is always a small JSON object.
    """
    response = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=PARSER_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.content[0].text.strip()


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_and_normalise(raw_json: str, original_prompt: str) -> dict:
    """
    Parses Claude's raw string output and validates the schema.
    Fills in safe defaults for any missing fields so the rest of the
    pipeline never has to handle KeyErrors.
    """
    # Strip any accidental markdown fences just in case
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

    base_class = str(data.get("base_class", "object")).strip().lower()
    if not base_class:
        base_class = "object"

    raw_attrs = data.get("attributes", {})
    attributes = {
        "color": _str_or_none(raw_attrs.get("color")),
        "make":  _str_or_none(raw_attrs.get("make")),
        "type":  _str_or_none(raw_attrs.get("type")),
        "other": [str(o) for o in raw_attrs.get("other", []) if o],
    }

    threshold = float(data.get("confidence_threshold", 0.65))
    threshold = max(0.30, min(0.95, threshold))

    return {
        "base_class":           base_class,
        "attributes":           attributes,
        "confidence_threshold": threshold,
    }


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None