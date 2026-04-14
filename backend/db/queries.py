import json
import aiosqlite
from datetime import datetime, timezone
from typing import Optional

                                                                                

async def insert_query_log(
    db:            aiosqlite.Connection,
    id:            str,
    prompt:        str,
    primary_class: str,
    target_count:  int,
    parsed_query:  dict,
    match_count:   int,
    inference_ms:  int,
    result_image:  Optional[str],
) -> None:
    """
    Inserts one completed query into the query_log table.

    primary_class — the base_class of the first "primary" target.
    target_count  — total number of targets extracted by the parser.
    parsed_query  — full parsed_query dict (targets + relational_hints),
                    stored as JSON in the parsed_attrs column.
    """
    await db.execute(
        """
        INSERT INTO query_log
            (id, prompt, base_class, target_count, parsed_attrs,
             match_count, inference_ms, result_image, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            id,
            prompt,
            primary_class,
            target_count,
            json.dumps(parsed_query),
            match_count,
            inference_ms,
            result_image,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    await db.commit()


                                                                                

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

                                         
    async with db.execute("SELECT COUNT(*) FROM query_log") as cursor:
        count_row = await cursor.fetchone()
        total = count_row[0] if count_row else 0

                       
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


                                                                                

def row_to_dict(row: aiosqlite.Row) -> dict:
    """
    Converts an aiosqlite.Row into a plain dict and deserialises
    the parsed_attrs JSON string (full parsed_query) back into a Python dict.
    """
    d = dict(row)
    if d.get("parsed_attrs"):
        try:
            d["parsed_attrs"] = json.loads(d["parsed_attrs"])
        except (json.JSONDecodeError, TypeError):
            d["parsed_attrs"] = {}
                                                                               
    d.setdefault("target_count", 1)
    return d