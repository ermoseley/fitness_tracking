#!/usr/bin/env python3

"""
SQLite-backed storage utilities for the Streamlit Fitness Tracker.

This module centralizes persistence for:
- weights (per user)
- lbm (per user)
- user height (per user)

It is designed to require minimal external dependencies (sqlite3 from stdlib)
and can be used both locally and in hosted environments. For Streamlit, the
connection is cached to reuse across reruns.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Any, Sequence

import pandas as pd

# Optional: cache connection in Streamlit environments
try:
    import streamlit as _st  # type: ignore
except Exception:  # pragma: no cover - not running in streamlit
    _st = None  # type: ignore


DB_FILENAME = "fitness_tracker.db"

# Optional cloud database (Postgres) via DATABASE_URL env var or Streamlit secrets
_DATABASE_URL: Optional[str] = None
if _st is not None:
    try:
        # Prefer secrets if available
        _DATABASE_URL = _st.secrets.get("DATABASE_URLPOR", None)  # type: ignore[attr-defined]
    except Exception:
        _DATABASE_URL = None
if not _DATABASE_URL:
    _DATABASE_URL = os.environ.get("DATABASE_URL", None)

_USE_SQLALCHEMY = False
_engine = None
if _DATABASE_URL:
    try:
        # Lazy import SQLAlchemy if a cloud URL is configured
        from sqlalchemy import create_engine, text  # type: ignore
        _engine = create_engine(_DATABASE_URL, pool_pre_ping=True)
        _USE_SQLALCHEMY = True
    except Exception:
        # Fallback to SQLite if SQLAlchemy or driver not available
        _engine = None
        _USE_SQLALCHEMY = False


def get_db_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, DB_FILENAME)


def _create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        get_db_path(),
        check_same_thread=False,
        isolation_level=None,  # autocommit; we manage transactions explicitly
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    # Some sensible pragmas for reliability and speed
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    return conn


if _USE_SQLALCHEMY:
    # Provide cached SQLAlchemy engine access
    if _st is not None:
        @_st.cache_resource(show_spinner=False)  # type: ignore[misc]
        def get_engine():
            return _engine
    else:
        def get_engine():
            return _engine
else:
    if _st is not None:
        @_st.cache_resource(show_spinner=False)  # cache one connection per process
        def get_connection() -> sqlite3.Connection:  # type: ignore[misc]
            return _create_connection()
    else:  # Fallback outside of Streamlit
        def get_connection() -> sqlite3.Connection:
            return _create_connection()


@contextmanager
def db_cursor() -> Iterable[Any]:
    if _USE_SQLALCHEMY:
        eng = get_engine()
        assert eng is not None
        with eng.begin() as conn:
            yield conn
    else:
        conn = get_connection()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()


def init_database() -> None:
    """Create tables and indices if they don't exist."""
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS weights (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    weight REAL NOT NULL
                );
                """
            ))
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS lbm (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    lbm REAL NOT NULL
                );
                """
            ))
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    height REAL NOT NULL
                );
                """
            ))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_weights_user_date ON weights(user_id, date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_lbm_user_date ON lbm(user_id, date);"))
            # user preferences table
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    confidence_interval TEXT NOT NULL DEFAULT '1σ',
                    enable_forecast INTEGER NOT NULL DEFAULT 1,
                    forecast_days INTEGER NOT NULL DEFAULT 30,
                    residuals_bins INTEGER NOT NULL DEFAULT 15
                );
                """
            ))
    else:
        with db_cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,  -- ISO8601 string
                    weight REAL NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS lbm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,  -- ISO8601 string
                    lbm REAL NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    height REAL NOT NULL
                );
                """
            )
            # Indices for query performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_weights_user_date ON weights(user_id, date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_lbm_user_date ON lbm(user_id, date);")
            # user preferences table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    confidence_interval TEXT NOT NULL DEFAULT '1σ',
                    enable_forecast INTEGER NOT NULL DEFAULT 1,
                    forecast_days INTEGER NOT NULL DEFAULT 30,
                    residuals_bins INTEGER NOT NULL DEFAULT 15
                );
                """
            )


# -------------------------
# Helpers
# -------------------------

def _to_iso(dt: datetime) -> str:
    # Normalize to naive ISO string (no timezone info)
    # Datetime inputs from Streamlit are naive already
    return dt.isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


# -------------------------
# Weights API
# -------------------------

def get_weights_for_user(user_id: str) -> List[Tuple[datetime, float]]:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            rows = list(conn.execute(text("SELECT date, weight FROM weights WHERE user_id=:uid ORDER BY date ASC;"), {"uid": user_id}))
        return [(_from_iso(d), float(w)) for (d, w) in rows]
    else:
        with db_cursor() as cur:
            cur.execute(
                "SELECT date, weight FROM weights WHERE user_id=? ORDER BY date ASC;",
                (user_id,),
            )
            rows = cur.fetchall()
        return [(_from_iso(d), float(w)) for (d, w) in rows]


def insert_weight_for_user(user_id: str, entry_datetime: datetime, weight: float) -> None:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text("INSERT INTO weights(user_id, date, weight) VALUES (:uid, :date, :w);"), {"uid": user_id, "date": _to_iso(entry_datetime), "w": float(weight)})
    else:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO weights(user_id, date, weight) VALUES (?, ?, ?);",
                (user_id, _to_iso(entry_datetime), float(weight)),
            )


def replace_weights_for_user(user_id: str, df: pd.DataFrame) -> None:
    required = {"date", "weight"}
    if not required.issubset(set(df.columns)):
        raise ValueError("weights DataFrame must contain 'date' and 'weight' columns")
    # Normalize types
    df_local = df.copy()
    df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
    df_local = df_local.dropna(subset=["date", "weight"])  # drop invalid rows
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text("DELETE FROM weights WHERE user_id=:uid;"), {"uid": user_id})
            if len(df_local) > 0:
                rows = [
                    {"uid": user_id, "date": _to_iso(dt.to_pydatetime()), "w": float(w)}
                    for dt, w in zip(df_local["date"], df_local["weight"])  # type: ignore[arg-type]
                ]
                conn.execute(text("INSERT INTO weights(user_id, date, weight) VALUES (:uid, :date, :w);"), rows)
    else:
        with db_cursor() as cur:
            cur.execute("BEGIN;")
            try:
                cur.execute("DELETE FROM weights WHERE user_id=?;", (user_id,))
                cur.executemany(
                    "INSERT INTO weights(user_id, date, weight) VALUES (?, ?, ?);",
                    [
                        (user_id, _to_iso(dt.to_pydatetime()), float(w))
                        for dt, w in zip(df_local["date"], df_local["weight"])  # type: ignore[arg-type]
                    ],
                )
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise


def get_weights_df_for_user(user_id: str) -> pd.DataFrame:
    data = get_weights_for_user(user_id)
    if not data:
        return pd.DataFrame(columns=["date", "weight"])
    return pd.DataFrame({
        "date": [dt.isoformat() for dt, _ in data],
        "weight": [w for _, w in data],
    })


# -------------------------
# LBM API
# -------------------------

def get_lbm_df_for_user(user_id: str) -> pd.DataFrame:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            rows = list(conn.execute(text("SELECT date, lbm FROM lbm WHERE user_id=:uid ORDER BY date ASC;"), {"uid": user_id}))
    else:
        with db_cursor() as cur:
            cur.execute(
                "SELECT date, lbm FROM lbm WHERE user_id=? ORDER BY date ASC;",
                (user_id,),
            )
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "lbm"])
    return pd.DataFrame({
        "date": [d for (d, _) in rows],
        "lbm": [float(v) for (_, v) in rows],
    })


def insert_lbm_for_user(user_id: str, entry_datetime: datetime, lbm_value: float) -> None:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text("INSERT INTO lbm(user_id, date, lbm) VALUES (:uid, :date, :v);"), {"uid": user_id, "date": _to_iso(entry_datetime), "v": float(lbm_value)})
    else:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO lbm(user_id, date, lbm) VALUES (?, ?, ?);",
                (user_id, _to_iso(entry_datetime), float(lbm_value)),
            )


def replace_lbm_for_user(user_id: str, df: pd.DataFrame) -> None:
    required = {"date", "lbm"}
    if not required.issubset(set(df.columns)):
        raise ValueError("lbm DataFrame must contain 'date' and 'lbm' columns")
    df_local = df.copy()
    df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
    df_local["lbm"] = pd.to_numeric(df_local["lbm"], errors="coerce")
    df_local = df_local.dropna(subset=["date", "lbm"])  # drop invalid rows
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text("DELETE FROM lbm WHERE user_id=:uid;"), {"uid": user_id})
            if len(df_local) > 0:
                rows = [
                    {"uid": user_id, "date": _to_iso(dt.to_pydatetime()), "v": float(v)}
                    for dt, v in zip(df_local["date"], df_local["lbm"])  # type: ignore[arg-type]
                ]
                conn.execute(text("INSERT INTO lbm(user_id, date, lbm) VALUES (:uid, :date, :v);"), rows)
    else:
        with db_cursor() as cur:
            cur.execute("BEGIN;")
            try:
                cur.execute("DELETE FROM lbm WHERE user_id=?;", (user_id,))
                cur.executemany(
                    "INSERT INTO lbm(user_id, date, lbm) VALUES (?, ?, ?);",
                    [
                        (user_id, _to_iso(dt.to_pydatetime()), float(v))
                        for dt, v in zip(df_local["date"], df_local["lbm"])  # type: ignore[arg-type]
                    ],
                )
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise


# -------------------------
# User settings API
# -------------------------

def get_height_for_user(user_id: str) -> Optional[float]:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            row = conn.execute(text("SELECT height FROM user_settings WHERE user_id=:uid;"), {"uid": user_id}).fetchone()
        return float(row[0]) if row is not None else None
    else:
        with db_cursor() as cur:
            cur.execute("SELECT height FROM user_settings WHERE user_id=?;", (user_id,))
            row = cur.fetchone()
        return float(row[0]) if row is not None else None


def set_height_for_user(user_id: str, height_inches: float) -> None:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text(
                """
                INSERT INTO user_settings (user_id, height) VALUES (:uid, :h)
                ON CONFLICT (user_id) DO UPDATE SET height=excluded.height;
                """
            ), {"uid": user_id, "h": float(height_inches)})
    else:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO user_settings (user_id, height) VALUES (?, ?) \n"
                "ON CONFLICT(user_id) DO UPDATE SET height=excluded.height;",
                (user_id, float(height_inches)),
            )


# -------------------------
# User preferences API
# -------------------------

from typing import Dict

def get_preferences_for_user(user_id: str) -> Dict[str, object]:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            row = conn.execute(text("SELECT confidence_interval, enable_forecast, forecast_days, residuals_bins FROM user_preferences WHERE user_id=:uid;"), {"uid": user_id}).fetchone()
        if not row:
            return {}
        return {
            "confidence_interval": row[0],
            "enable_forecast": bool(row[1]),
            "forecast_days": int(row[2]),
            "residuals_bins": int(row[3]),
        }
    else:
        with db_cursor() as cur:
            cur.execute("SELECT confidence_interval, enable_forecast, forecast_days, residuals_bins FROM user_preferences WHERE user_id=?;", (user_id,))
            row = cur.fetchone()
        if not row:
            return {}
        return {
            "confidence_interval": row[0],
            "enable_forecast": bool(row[1]),
            "forecast_days": int(row[2]),
            "residuals_bins": int(row[3]),
        }


def set_preferences_for_user(user_id: str, confidence_interval: str, enable_forecast: bool, forecast_days: int, residuals_bins: int) -> None:
    if _USE_SQLALCHEMY:
        from sqlalchemy import text  # type: ignore
        with db_cursor() as conn:
            conn.execute(text(
                """
                INSERT INTO user_preferences(user_id, confidence_interval, enable_forecast, forecast_days, residuals_bins)
                VALUES (:uid, :ci, :ef, :fd, :rb)
                ON CONFLICT (user_id) DO UPDATE SET
                    confidence_interval=excluded.confidence_interval,
                    enable_forecast=excluded.enable_forecast,
                    forecast_days=excluded.forecast_days,
                    residuals_bins=excluded.residuals_bins;
                """
            ), {"uid": user_id, "ci": confidence_interval, "ef": 1 if enable_forecast else 0, "fd": int(forecast_days), "rb": int(residuals_bins)})
    else:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_preferences(user_id, confidence_interval, enable_forecast, forecast_days, residuals_bins)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    confidence_interval=excluded.confidence_interval,
                    enable_forecast=excluded.enable_forecast,
                    forecast_days=excluded.forecast_days,
                    residuals_bins=excluded.residuals_bins;
                """,
                (user_id, confidence_interval, 1 if enable_forecast else 0, int(forecast_days), int(residuals_bins))
            )


# -------------------------
# Migration helpers (optional)
# -------------------------

def migrate_weights_csv_to_db(user_id: str, csv_df: pd.DataFrame) -> None:
    """Convenience to bulk-load from an existing CSV-style DataFrame."""
    replace_weights_for_user(user_id, csv_df)


def migrate_lbm_csv_to_db(user_id: str, csv_df: pd.DataFrame) -> None:
    replace_lbm_for_user(user_id, csv_df)


def delete_weight_entry(user_id: str, entry_datetime: datetime) -> bool:
    """Delete a specific weight entry by datetime. Returns True if deleted."""
    try:
        with db_cursor() as conn:
            if _USE_SQLALCHEMY:
                from sqlalchemy import text
                result = conn.execute(
                    text("DELETE FROM weights WHERE user_id=:uid AND date=:dt;"),
                    {"uid": user_id, "dt": entry_datetime.isoformat()}
                )
                return result.rowcount > 0
            else:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM weights WHERE user_id=? AND date=?;",
                    (user_id, entry_datetime.isoformat())
                )
                return cur.rowcount > 0
    except Exception:
        return False


def delete_lbm_entry(user_id: str, entry_datetime: datetime) -> bool:
    """Delete a specific LBM entry by datetime. Returns True if deleted."""
    try:
        with db_cursor() as conn:
            if _USE_SQLALCHEMY:
                from sqlalchemy import text
                result = conn.execute(
                    text("DELETE FROM lbm WHERE user_id=:uid AND date=:dt;"),
                    {"uid": user_id, "dt": entry_datetime.isoformat()}
                )
                return result.rowcount > 0
            else:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM lbm WHERE user_id=? AND date=?;",
                    (user_id, entry_datetime.isoformat())
                )
                return cur.rowcount > 0
    except Exception:
        return False


