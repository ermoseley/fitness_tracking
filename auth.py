#!/usr/bin/env python3
"""
Simple authentication module for the Streamlit Fitness Tracker.
Supports user-based authentication with password hashing.
"""

import hashlib
import secrets
import sqlite3
from typing import Optional, Tuple
import os

# Reuse storage's database selection (SQLite vs Postgres via SQLAlchemy)
try:
    # These imports are lightweight when SQLite; SQLAlchemy engine is created in storage
    from storage import _USE_SQLALCHEMY as _AUTH_USE_SQLALCHEMY  # type: ignore
    from storage import get_engine, get_db_path  # type: ignore
except Exception:
    _AUTH_USE_SQLALCHEMY = False  # fallback to SQLite-only auth
    def get_db_path() -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "fitness_tracker.db")

# Optional: cache connection in Streamlit environments
try:
    import streamlit as st  # type: ignore
    _st = st  # alias used elsewhere in this module
    # Optional cookie manager for persistence across refresh
    try:
        from streamlit_cookies_manager import EncryptedCookieManager  # type: ignore
    except Exception:
        EncryptedCookieManager = None  # type: ignore
except Exception:  # pragma: no cover - not running in streamlit
    st = None  # type: ignore
    _st = None  # type: ignore
    EncryptedCookieManager = None  # type: ignore

def _get_sqlite_conn():
    return sqlite3.connect(get_db_path())

def _hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash a password with a salt. Returns (hashed_password, salt)"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use PBKDF2 for secure password hashing
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return hashed.hex(), salt

def _verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify a password against its hash and salt"""
    test_hash, _ = _hash_password(password, salt)
    return test_hash == hashed_password

# -------------------- Cookie helpers --------------------
_COOKIE_NAME = "bm_auth_sid"

def _get_cookie_manager():
    """Return an initialized cookie manager if available in Streamlit context."""
    if _st is None or EncryptedCookieManager is None:
        return None
    cm = EncryptedCookieManager(
        prefix="bodymetrics",
        password=os.environ.get("COOKIE_PASSWORD", "change-me-please"),
    )
    if not cm.ready():
        _st.stop()
    return cm

def _set_session_cookie(session_id: str):
    cm = _get_cookie_manager()
    if cm is None:
        return
    cm[_COOKIE_NAME] = session_id
    cm.save()

def _get_session_cookie() -> Optional[str]:
    cm = _get_cookie_manager()
    if cm is None:
        return None
    return cm.get(_COOKIE_NAME)

def _clear_session_cookie():
    cm = _get_cookie_manager()
    if cm is None:
        return
    try:
        del cm[_COOKIE_NAME]
        cm.save()
    except Exception:
        pass

def init_auth_tables():
    """Initialize authentication tables in the database (SQLite or Postgres)."""
    if _AUTH_USE_SQLALCHEMY:
        from sqlalchemy import text
        eng = get_engine()
        assert eng is not None
        with eng.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                );
                """
            ))
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                );
                """
            ))
    else:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        # Create users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        # Create sessions table for tracking active sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES auth_users (user_id)
            )
        """)
        conn.commit()
        conn.close()

def register_user(user_id: str, password: str) -> bool:
    """Register a new user with a password. Returns True if successful."""
    try:
        if _AUTH_USE_SQLALCHEMY:
            from sqlalchemy import text
            eng = get_engine()
            assert eng is not None
            with eng.begin() as conn:
                row = conn.execute(text("SELECT user_id FROM auth_users WHERE user_id = :u"), {"u": user_id}).fetchone()
                if row:
                    return False
                password_hash, salt = _hash_password(password)
                conn.execute(text("""
                    INSERT INTO auth_users (user_id, password_hash, salt)
                    VALUES (:u, :p, :s)
                """), {"u": user_id, "p": password_hash, "s": salt})
            return True
        else:
            conn = _get_sqlite_conn()
            cursor = conn.cursor()
            # Check if user already exists
            cursor.execute("SELECT user_id FROM auth_users WHERE user_id = ?", (user_id,))
            if cursor.fetchone():
                conn.close()
                return False
            # Hash the password
            password_hash, salt = _hash_password(password)
            # Insert new user
            cursor.execute("""
                INSERT INTO auth_users (user_id, password_hash, salt) 
                VALUES (?, ?, ?)
            """, (user_id, password_hash, salt))
            conn.commit()
            conn.close()
            return True
    except Exception:
        return False

def authenticate_user(user_id: str, password: str) -> bool:
    """Authenticate a user with their password. Returns True if successful."""
    try:
        if _AUTH_USE_SQLALCHEMY:
            from sqlalchemy import text
            eng = get_engine()
            assert eng is not None
            with eng.begin() as conn:
                result = conn.execute(text(
                    "SELECT password_hash, salt FROM auth_users WHERE user_id = :u"
                ), {"u": user_id}).fetchone()
                if not result:
                    return False
                password_hash, salt = result
                is_valid = _verify_password(password, password_hash, salt)
                if is_valid:
                    conn.execute(text("UPDATE auth_users SET last_login = CURRENT_TIMESTAMP WHERE user_id = :u"), {"u": user_id})
                return is_valid
        else:
            conn = _get_sqlite_conn()
            cursor = conn.cursor()
            # Get user's password hash and salt
            cursor.execute("""
                SELECT password_hash, salt FROM auth_users WHERE user_id = ?
            """, (user_id,))
            result = cursor.fetchone()
            if not result:
                conn.close()
                return False
            password_hash, salt = result
            # Verify password
            is_valid = _verify_password(password, password_hash, salt)
            if is_valid:
                cursor.execute("""
                    UPDATE auth_users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
                """, (user_id,))
                conn.commit()
            conn.close()
            return is_valid
    except Exception:
        return False

def create_session(user_id: str) -> str:
    """Create a new session for a user. Returns session ID."""
    session_id = secrets.token_urlsafe(32)
    if _AUTH_USE_SQLALCHEMY:
        from sqlalchemy import text
        eng = get_engine()
        assert eng is not None
        with eng.begin() as conn:
            conn.execute(text(
                """
                INSERT INTO auth_sessions (session_id, user_id, created_at, expires_at)
                VALUES (:sid, :uid, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP + INTERVAL '24 HOURS')
                """
            ), {"sid": session_id, "uid": user_id})
        return session_id
    else:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        # Set session to expire in 24 hours
        cursor.execute("""
            INSERT INTO auth_sessions (session_id, user_id, expires_at) 
            VALUES (?, ?, datetime('now', '+24 hours'))
        """, (session_id, user_id))
        conn.commit()
        conn.close()
        return session_id

def validate_session(session_id: str) -> Optional[str]:
    """Validate a session and return the user_id if valid, None if invalid/expired."""
    try:
        if _AUTH_USE_SQLALCHEMY:
            from sqlalchemy import text
            eng = get_engine()
            assert eng is not None
            with eng.begin() as conn:
                result = conn.execute(text(
                    """
                    SELECT user_id FROM auth_sessions 
                    WHERE session_id = :sid AND expires_at > CURRENT_TIMESTAMP
                    """
                ), {"sid": session_id}).fetchone()
                return result[0] if result else None
        else:
            conn = _get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id FROM auth_sessions 
                WHERE session_id = ? AND expires_at > datetime('now')
            """, (session_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
    except Exception:
        return None

def logout_session(session_id: str):
    """Logout by deleting the session."""
    try:
        if _AUTH_USE_SQLALCHEMY:
            from sqlalchemy import text
            eng = get_engine()
            assert eng is not None
            with eng.begin() as conn:
                conn.execute(text("DELETE FROM auth_sessions WHERE session_id = :sid"), {"sid": session_id})
        else:
            conn = _get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM auth_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
    except Exception:
        pass

def cleanup_expired_sessions():
    """Clean up expired sessions from the database."""
    try:
        if _AUTH_USE_SQLALCHEMY:
            from sqlalchemy import text
            eng = get_engine()
            assert eng is not None
            with eng.begin() as conn:
                conn.execute(text("DELETE FROM auth_sessions WHERE expires_at <= CURRENT_TIMESTAMP"))
        else:
            conn = _get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM auth_sessions WHERE expires_at <= datetime('now')")
            conn.commit()
            conn.close()
    except Exception:
        pass

# Streamlit-specific authentication functions
def get_current_user() -> Optional[str]:
    """Get the currently authenticated user from Streamlit session state."""
    if _st is None:
        return None
    
    # Try session_state first
    session_id = _st.session_state.get("auth_session_id")
    # If not present, attempt to restore from cookie
    if not session_id:
        cookie_sid = _get_session_cookie()
        if cookie_sid:
            _st.session_state.auth_session_id = cookie_sid
            session_id = cookie_sid
    if not session_id:
        return None
    
    user_id = validate_session(session_id)
    
    # If session is invalid/expired, clean up session state
    if user_id is None and "auth_session_id" in _st.session_state:
        del _st.session_state.auth_session_id
        if "user_id" in _st.session_state:
            del _st.session_state.user_id
        _clear_session_cookie()
    
    return user_id

def require_auth():
    """Decorator/function to require authentication. Call this at the start of your app."""
    if _st is None:
        return True
    
    # Check if user is already authenticated
    current_user = get_current_user()
    if current_user:
        return True
    
    # Show login/register form
    show_auth_form()
    return False

def show_auth_form():
    """Show the authentication form in Streamlit."""
    if _st is None:
        return
    
    st.title("🔐 BodyMetrics Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            user_id = st.text_input("User ID", placeholder="Enter your user ID")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Login", type="primary")
            
            if login_button:
                if not user_id or not password:
                    st.error("Please enter both User ID and Password")
                elif authenticate_user(user_id, password):
                    # Create session and store in session state
                    session_id = create_session(user_id)
                    _st.session_state.auth_session_id = session_id
                    _st.session_state.user_id = user_id
                    # Persist in cookie for refresh persistence
                    _set_session_cookie(session_id)
                    _st.rerun()
                else:
                    st.error("Invalid User ID or Password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            new_user_id = st.text_input("User ID", placeholder="Choose a unique user ID", key="reg_user_id")
            new_password = st.text_input("Password", type="password", placeholder="Choose a secure password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            register_button = st.form_submit_button("Register", type="primary")
            
            if register_button:
                if not new_user_id or not new_password or not confirm_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif register_user(new_user_id, new_password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("User ID already exists. Please choose a different one.")
    
    st.info("💡 **Tip**: Your User ID will be used to store your fitness data separately from other users.")

def logout():
    """Logout the current user."""
    if _st is None:
        return
    
    session_id = _st.session_state.get("auth_session_id")
    if session_id:
        logout_session(session_id)
    
    # Clear session state
    if "auth_session_id" in _st.session_state:
        del _st.session_state.auth_session_id
    if "user_id" in _st.session_state:
        del _st.session_state.user_id
    _clear_session_cookie()
    
    _st.rerun()
