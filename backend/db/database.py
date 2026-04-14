import aiosqlite
import os
from pathlib import Path

                                                                                

                                                                         
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "db" / "queries.db"

                                                                  
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

                                                                                

                                                          
                                                                               
CREATE_QUERY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS query_log (
    id              TEXT      PRIMARY KEY,           -- UUID string
    prompt          TEXT      NOT NULL,              -- Raw user prompt
    base_class      TEXT      NOT NULL,              -- Primary target base class e.g. "monkey"
    target_count    INTEGER   NOT NULL DEFAULT 1,    -- Total number of targets in parsed_query
    parsed_attrs    TEXT,                            -- Full parsed_query JSON (targets + hints)
    match_count     INTEGER   NOT NULL DEFAULT 0,    -- Total matched instances across all targets
    inference_ms    INTEGER   NOT NULL DEFAULT 0,    -- End-to-end inference time
    result_image    TEXT,                            -- Filename in storage/results/
    created_at      DATETIME  NOT NULL               -- ISO 8601 UTC timestamp
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_created_at ON query_log (created_at DESC);",
]

                                                                           
MIGRATIONS = [
    "ALTER TABLE query_log ADD COLUMN target_count INTEGER NOT NULL DEFAULT 1;",
]

                                                                                

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
                                                                            
        db.row_factory = aiosqlite.Row
        yield db

                                                                                

async def init_db() -> None:
    """
    Creates the database file and all tables if they don't already exist.
    Called once at server startup from main.py's lifespan handler.
    Safe to call multiple times — all statements use IF NOT EXISTS.
    Also applies forward migrations for new columns (errors silently ignored
    if the column already exists, which is standard SQLite practice).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(CREATE_QUERY_LOG_TABLE)
        for index_sql in CREATE_INDEXES:
            await db.execute(index_sql)
        for migration_sql in MIGRATIONS:
            try:
                await db.execute(migration_sql)
            except Exception:
                pass                                          
        await db.commit()

    print(f"[db] SQLite ready at {DB_PATH}")