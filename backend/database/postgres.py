import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

_pool: Optional[ThreadedConnectionPool] = None

POOL_MIN = int(os.getenv("POSTGRES_POOL_MIN", "2"))
POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "10"))


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=POOL_MIN,
            maxconn=POOL_MAX,
            dbname=os.getenv("POSTGRES_DB", "solver_agent"),
            user=os.getenv("POSTGRES_USER", "shihuimei"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )
        logger.info("PostgreSQL pool created: min=%d max=%d", POOL_MIN, POOL_MAX)
    return _pool


def _get_conn():
    return _get_pool().getconn()


def _put_conn(conn):
    _get_pool().putconn(conn)


def fetch_one(query: str, params: tuple = None):
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            return cur.fetchone()
    finally:
        _put_conn(conn)


def fetch_all(query: str, params: tuple = None):
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            return cur.fetchall()
    finally:
        _put_conn(conn)


def execute(query: str, params: tuple = None):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
        conn.commit()
    finally:
        _put_conn(conn)
