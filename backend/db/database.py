import aiosqlite
import os
from pathlib import Path

# ── Path config ───────────────────────────────────────────────────────────────

# Resolve db file path relative to project root, not this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "db" / "queries.db"

# Ensure the db/ directory exists before trying to create the file
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────

# Single table that logs every query the system processes.
# Keep it flat — no foreign keys, no joins — SQLite + one table is plenty here.
CREATE_QUERY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS query_log (
    id              TEXT      PRIMARY KEY,           -- UUID string
    prompt          TEXT      NOT NULL,              -- Raw user prompt
    base_class      TEXT      NOT NULL,              -- LLM-parsed base class e.g. "car"
    parsed_attrs    TEXT,                            -- JSON string of parsed attributes
    match_count     INTEGER   NOT NULL DEFAULT 0,    -- Total matched instances
    inference_ms    INTEGER   NOT NULL DEFAULT 0,    -- End-to-end inference time
    result_image    TEXT,                            -- Filename in storage/results/
    created_at      DATETIME  NOT NULL               -- ISO 8601 UTC timestamp
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_created_at ON query_log (created_at DESC);",
]

# ── Connection helper ─────────────────────────────────────────────────────────

async def get_db() -> aiosqlite.Connection:
    """
    Opens and returns a single aiosqlite connection.
    Used as a FastAPI dependency — the route receives the connection,
    uses it, and the context manager closes it automatically.

    Usage in a route:
        async def my_route(db: aiosqlite.Connection = Depends(get_db)):
            ...
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # Return rows as dict-like objects so columns are accessible by name
        db.row_factory = aiosqlite.Row
        yield db

# ── Initialisation ────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Creates the database file and all tables if they don't already exist.
    Called once at server startup from main.py's lifespan handler.
    Safe to call multiple times — all statements use IF NOT EXISTS.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(CREATE_QUERY_LOG_TABLE)
        for index_sql in CREATE_INDEXES:
            await db.execute(index_sql)
        await db.commit()

    print(f"[db] SQLite ready at {DB_PATH}")