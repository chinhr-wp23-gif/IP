"""
Firebase-backed authentication & user-management module for the Digital
Watermarking System.

- Firebase Authentication (Email/Password) handles credential storage and
  verification — Google manages the password hashing, not us.
- Cloud Firestore stores each user's role ('user' / 'admin') and the
  activity log used by the admin panel.

Since Firebase Auth requires an email, usernames are mapped internally to
a synthetic address "<username>@wm-app.local" — users never see this, they
only ever type a username.

REQUIRED Streamlit secrets (Settings -> Secrets on Streamlit Cloud, or
.streamlit/secrets.toml locally):

    [firebase]
    api_key = "your-web-api-key"
    default_admin_password = "SomeStrongPassword123!"   # optional, used once

    [firebase_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "..."
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."

(Copy these fields straight out of the service-account JSON you downloaded
from Firebase; keep the \n's in private_key literal, in quotes, on one line.)
"""

import requests
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
from google.cloud.firestore_v1.base_query import FieldFilter

FAKE_EMAIL_DOMAIN = "wm-app.local"
DEFAULT_ADMIN_USERNAME = "admin"

try:
    DEFAULT_ADMIN_PASSWORD = st.secrets["firebase"].get(
        "default_admin_password", "ChangeMe123!"
    )
except Exception:
    DEFAULT_ADMIN_PASSWORD = "ChangeMe123!"

_db = None


def _username_to_email(username):
    return f"{username.strip().lower()}@{FAKE_EMAIL_DOMAIN}"


def _init_firebase():
    global _db
    if _db is not None:
        return
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_service_account"]))
        firebase_admin.initialize_app(cred)
    _db = firestore.client()


def _api_key():
    return st.secrets["firebase"]["api_key"]


def _find_uid(username):
    docs = (
        _db.collection("users")
        .where(filter=FieldFilter("username", "==", username))
        .limit(1)
        .stream()
    )
    for d in docs:
        return d.id
    return None


def _admin_count():
    return sum(
        1 for _ in _db.collection("users").where(filter=FieldFilter("role", "==", "admin")).stream()
    )


def get_db():
    """Shared Firestore client, for other modules (exhibitions.py, etc.) to reuse
    the same Firebase connection instead of initializing their own."""
    _init_firebase()
    return _db


def init_db():
    """Connect to Firebase and seed a default admin account if none exists."""
    _init_firebase()
    if _admin_count() == 0:
        create_user(DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, role="admin")


def create_user(username, password, role="user"):
    """Self-service / admin account creation. Returns (ok: bool, message: str)."""
    _init_firebase()
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    email = _username_to_email(username)
    try:
        user_record = fb_auth.create_user(
            email=email, password=password, display_name=username
        )
    except fb_auth.EmailAlreadyExistsError:
        return False, "That username is already taken."
    except Exception as e:
        return False, f"Could not create account ({e})."

    _db.collection("users").document(user_record.uid).set(
        {
            "username": username,
            "role": role,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
    )
    return True, "Account created successfully."


def verify_user(username, password):
    """Verifies credentials against Firebase Auth. Returns {'username','role'} or None."""
    _init_firebase()
    username = (username or "").strip()
    if not username or not password:
        return None
    email = _username_to_email(username)
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={_api_key()}"
    )
    try:
        resp = requests.post(
            url,
            json={"email": email, "password": password, "returnSecureToken": True},
            timeout=10,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    uid = resp.json()["localId"]
    doc = _db.collection("users").document(uid).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    return {"username": data.get("username", username), "role": data.get("role", "user")}


def get_all_users():
    _init_firebase()
    users = []
    for d in _db.collection("users").stream():
        data = d.to_dict()
        created = data.get("created_at")
        users.append(
            {
                "username": data.get("username"),
                "role": data.get("role"),
                "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
            }
        )
    return sorted(users, key=lambda u: u["username"] or "")


def delete_user(username):
    """Returns (ok: bool, message: str). Refuses to delete the last admin."""
    _init_firebase()
    uid = _find_uid(username)
    if not uid:
        return False, "User not found."
    data = _db.collection("users").document(uid).get().to_dict() or {}
    if data.get("role") == "admin" and _admin_count() <= 1:
        return False, "Cannot delete the last remaining admin account."
    fb_auth.delete_user(uid)
    _db.collection("users").document(uid).delete()
    return True, f"User '{username}' deleted."


def reset_password(username, new_password):
    _init_firebase()
    if len(new_password or "") < 6:
        return False, "Password must be at least 6 characters."
    uid = _find_uid(username)
    if not uid:
        return False, "User not found."
    fb_auth.update_user(uid, password=new_password)
    return True, "Password updated."


def set_role(username, role):
    _init_firebase()
    uid = _find_uid(username)
    if not uid:
        return False, "User not found."
    data = _db.collection("users").document(uid).get().to_dict() or {}
    if role != "admin" and data.get("role") == "admin" and _admin_count() <= 1:
        return False, "Cannot demote the last remaining admin account."
    _db.collection("users").document(uid).update({"role": role})
    return True, "Role updated."


def log_activity(username, action, details=""):
    _init_firebase()
    _db.collection("activity_log").add(
        {
            "username": username,
            "action": action,
            "details": details,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )


def get_logs(limit=300, username_filter=None):
    _init_firebase()
    query = _db.collection("activity_log")
    if username_filter:
        query = query.where(filter=FieldFilter("username", "==", username_filter))
    query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)

    logs = []
    for d in query.stream():
        data = d.to_dict()
        ts = data.get("timestamp")
        logs.append(
            {
                "username": data.get("username"),
                "action": data.get("action"),
                "details": data.get("details"),
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            }
        )
    return logs
