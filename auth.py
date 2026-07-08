"""
Authentication & user-management module for the Digital Watermarking System.

Uses a local SQLite database (created automatically on first run) to store
user accounts (username / salted-hashed password / role) and an activity
log used by the admin panel.

NOTE on Streamlit Community Cloud: the filesystem is ephemeral — the
database persists across reruns and user sessions while the app instance
stays alive, but is wiped on redeploy / reboot / sleep-wake in some cases.
For a class project this is normally fine. If you need accounts to survive
redeploys, swap DB_PATH for a persistent store (e.g. a mounted volume or an
external DB) later — the rest of the app doesn't need to change.
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data.db")

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet, and seed a default admin account."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        admin_count = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role = 'admin'"
        ).fetchone()["c"]
        if admin_count == 0:
            _insert_user(conn, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, role="admin")


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return pw_hash, salt


def _insert_user(conn, username, password, role="user"):
    pw_hash, salt = _hash_password(password)
    conn.execute(
        "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, pw_hash, salt, role, datetime.now().isoformat(timespec="seconds")),
    )


def create_user(username, password, role="user"):
    """Self-service / admin account creation. Returns (ok: bool, message: str)."""
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return False, "That username is already taken."
        _insert_user(conn, username, password, role)
    return True, "Account created successfully."


def verify_user(username, password):
    """Returns {'username':..., 'role':...} on success, else None."""
    username = (username or "").strip()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None
    pw_hash, _ = _hash_password(password, row["salt"])
    if secrets.compare_digest(pw_hash, row["password_hash"]):
        return {"username": row["username"], "role": row["role"]}
    return None


def get_all_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user(username):
    """Returns (ok: bool, message: str). Refuses to delete the last admin."""
    with get_conn() as conn:
        target = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if target is None:
            return False, "User not found."
        if target["role"] == "admin":
            admin_count = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE role = 'admin'"
            ).fetchone()["c"]
            if admin_count <= 1:
                return False, "Cannot delete the last remaining admin account."
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
    return True, f"User '{username}' deleted."


def reset_password(username, new_password):
    if len(new_password or "") < 6:
        return False, "Password must be at least 6 characters."
    pw_hash, salt = _hash_password(new_password)
    with get_conn() as conn:
        result = conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (pw_hash, salt, username),
        )
        if result.rowcount == 0:
            return False, "User not found."
    return True, "Password updated."


def set_role(username, role):
    with get_conn() as conn:
        if role != "admin":
            admin_count = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE role = 'admin'"
            ).fetchone()["c"]
            target = conn.execute(
                "SELECT role FROM users WHERE username = ?", (username,)
            ).fetchone()
            if target and target["role"] == "admin" and admin_count <= 1:
                return False, "Cannot demote the last remaining admin account."
        conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
    return True, "Role updated."


def log_activity(username, action, details=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO activity_log (username, action, details, timestamp) VALUES (?, ?, ?, ?)",
            (username, action, details, datetime.now().isoformat(timespec="seconds")),
        )


def get_logs(limit=300, username_filter=None):
    with get_conn() as conn:
        if username_filter:
            rows = conn.execute(
                "SELECT * FROM activity_log WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]
