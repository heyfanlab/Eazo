"""Small adapter for the application's existing parameterized SQL statements."""
import re
import atexit
import threading
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import config

_pool = None
_guard = threading.Lock()
_ID_TABLES = {"users", "daily_records", "meal_records", "free_inputs", "corpus_chunks", "user_notes"}


def pool():
    global _pool
    with _guard:
        if _pool is None:
            _pool = ConnectionPool(
                config.DATABASE_URL, min_size=0, max_size=8, timeout=15,
                kwargs={"row_factory": dict_row, "connect_timeout": 10, "prepare_threshold": None},
                check=ConnectionPool.check_connection, open=True,
            )
            atexit.register(_pool.close, timeout=0.3)
    return _pool


class Cursor:
    def __init__(self, cursor, lastrowid=None):
        self.cursor = cursor
        self.lastrowid = lastrowid

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def __iter__(self):
        return iter(self.cursor)


class Connection:
    def __init__(self):
        self.raw = pool().getconn()
        self.closed = False

    def execute(self, sql, params=()):
        # Only internal SQL is passed here. Values always remain bound parameters.
        match = re.match(r"\s*INSERT INTO\s+(\w+)", sql, re.I)
        returns_id = bool(match and match[1] in _ID_TABLES and "RETURNING" not in sql.upper())
        if returns_id:
            sql = sql.rstrip().rstrip(";") + " RETURNING id"
        cursor = self.raw.execute(sql.replace("?", "%s"), params)
        return Cursor(cursor, cursor.fetchone()["id"] if returns_id else None)

    def executescript(self, sql):
        self.raw.execute(sql, prepare=False)

    def commit(self):
        self.raw.commit()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.raw.rollback()
            finally:
                pool().putconn(self.raw)


@contextmanager
def user_lock(user_id):
    # Transaction-scoped locks work with Neon's pooled connection endpoint.
    with pool().connection() as conn:
        conn.execute("SET LOCAL lock_timeout = '15s'")
        conn.execute("SELECT pg_advisory_xact_lock(1280134477, %s)", (user_id,))
        yield
