import json
import aiosqlite
from datetime import datetime, timezone
from typing import Optional

# ── Write ─────────────────────────────────────────────────────────────────────

async def insert_query_log(
    db:            aiosqlite.Connection,
    id:            str,
    prompt:        str,
    base_class:    str,
    parsed_attrs:  dict,
    match_count:   int,
    inference_ms:  int,
    result_image:  Optional[str],
) -> None:
    """
    Inserts one completed query into the query_log table.
    Called by the route handler after the full pipeline finishes.

    parsed_attrs is stored as a JSON string since SQLite has no native
    dict/object type — deserialised back to a dict on read in row_to_dict().
    """
    await db.execute(
        """
        INSERT INTO query_log
            (id, prompt, base_class, parsed_attrs,
             match_count, inference_ms, result_image, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            id,
            prompt,
            base_class,
            json.dumps(parsed_attrs),
            match_count,
            inference_ms,
            result_image,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    await db.commit()


# ── Read ──────────────────────────────────────────────────────────────────────

async def fetch_query_by_id(
    db: aiosqlite.Connection,
    id: str,
) -> Optional[dict]:
    """
    Fetches a single query log entry by its UUID.
    Returns None if not found.
    """
    async with db.execute(
        "SELECT * FROM query_log WHERE id = ?", (id,)
    ) as cursor:
        row = await cursor.fetchone()
        return row_to_dict(row) if row else None


async def fetch_query_history(
    db:        aiosqlite.Connection,
    page:      int = 1,
    page_size: int = 20,
) -> dict:
    """
    Returns a paginated list of query log entries, newest first.
    Also returns the total row count so the frontend can render pagination.

    page is 1-indexed.
    """
    offset = (page - 1) * page_size

    # Total count for pagination metadata
    async with db.execute("SELECT COUNT(*) FROM query_log") as cursor:
        count_row = await cursor.fetchone()
        total = count_row[0] if count_row else 0

    # Paginated results
    async with db.execute(
        """
        SELECT * FROM query_log
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (page_size, offset),
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        "entries":   [row_to_dict(r) for r in rows],
        "total":     total,
        "page":      page,
        "page_size": page_size,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def row_to_dict(row: aiosqlite.Row) -> dict:
    """
    Converts an aiosqlite.Row into a plain dict and deserialises
    the parsed_attrs JSON string back into a Python dict.
    """
    d = dict(row)
    if d.get("parsed_attrs"):
        try:
            d["parsed_attrs"] = json.loads(d["parsed_attrs"])
        except (json.JSONDecodeError, TypeError):
            d["parsed_attrs"] = {}
    return d