"""
Events & submissions module for the Exhibition/Gallery feature.

Firestore collections:
- 'events': admin-created exhibitions (title, description, date range, status)
- 'submissions': a user's watermarked image submitted to an event, pending
  admin review. Holds Cloudinary URLs for the watermarked image plus
  whatever auxiliary files (original cover image, embedding keys, reference
  watermark) are needed to re-run extraction for verification later.
"""

import uuid
from firebase_admin import firestore

import auth
import storage


def _db():
    return auth.get_db()


# ---------------------------------------------------------------- Events ---

def create_event(title, description, start_date, end_date, created_by):
    title = (title or "").strip()
    if not title:
        return False, "Event title cannot be empty.", None
    doc_ref = _db().collection("events").document()
    doc_ref.set(
        {
            "title": title,
            "description": (description or "").strip(),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "created_by": created_by,
            "created_at": firestore.SERVER_TIMESTAMP,
            "status": "active",
        }
    )
    return True, "Event created.", doc_ref.id


def get_events(status=None):
    q = _db().collection("events")
    if status:
        q = q.where("status", "==", status)
    events = []
    for d in q.stream():
        data = d.to_dict()
        data["id"] = d.id
        events.append(data)
    events.sort(key=lambda e: e.get("start_date") or "", reverse=True)
    return events


def get_event(event_id):
    doc = _db().collection("events").document(event_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def close_event(event_id):
    _db().collection("events").document(event_id).update({"status": "closed"})
    return True, "Event closed."


def reopen_event(event_id):
    _db().collection("events").document(event_id).update({"status": "active"})
    return True, "Event reopened."


def delete_event(event_id):
    # Also remove submissions tied to this event.
    subs = get_submissions(event_id=event_id)
    for s in subs:
        delete_submission(s["id"])
    _db().collection("events").document(event_id).delete()
    return True, "Event and its submissions deleted."


# ----------------------------------------------------------- Submissions ---

def submit_to_event(
    event_id,
    username,
    watermarked_bytes,
    method,
    alpha,
    block_size=None,
    cover_bytes=None,
    keys_bytes=None,
    wm_ref_bytes=None,
):
    """Uploads the submission's image files to Cloudinary and records the
    submission (status='pending') in Firestore. Returns (ok, message, submission_id)."""
    if not event_id:
        return False, "No event selected.", None
    if watermarked_bytes is None:
        return False, "Missing watermarked image.", None

    submission_id = str(uuid.uuid4())
    folder = f"submissions/{submission_id}"

    watermarked_url = storage.upload_bytes(watermarked_bytes, folder, "watermarked")
    cover_url = storage.upload_bytes(cover_bytes, folder, "cover") if cover_bytes else None
    wm_ref_url = storage.upload_bytes(wm_ref_bytes, folder, "wm_ref") if wm_ref_bytes else None
    keys_url = (
        storage.upload_bytes(keys_bytes, folder, "keys", resource_type="raw")
        if keys_bytes
        else None
    )

    _db().collection("submissions").document(submission_id).set(
        {
            "event_id": event_id,
            "username": username,
            "status": "pending",
            "method": method,
            "alpha": alpha,
            "block_size": block_size,
            "watermarked_url": watermarked_url,
            "cover_url": cover_url,
            "wm_ref_url": wm_ref_url,
            "keys_url": keys_url,
            "submitted_at": firestore.SERVER_TIMESTAMP,
            "reviewed_at": None,
            "reviewed_by": None,
            "nc_score": None,
            "ber_score": None,
        }
    )
    return True, "Submitted for review.", submission_id


def get_submissions(event_id=None, status=None, username=None):
    q = _db().collection("submissions")
    if event_id:
        q = q.where("event_id", "==", event_id)
    if status:
        q = q.where("status", "==", status)
    if username:
        q = q.where("username", "==", username)
    subs = []
    for d in q.stream():
        data = d.to_dict()
        data["id"] = d.id
        subs.append(data)
    return subs


def get_submission(submission_id):
    doc = _db().collection("submissions").document(submission_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def review_submission(submission_id, approve, reviewer, nc_score=None, ber_score=None):
    _db().collection("submissions").document(submission_id).update(
        {
            "status": "approved" if approve else "rejected",
            "reviewed_by": reviewer,
            "reviewed_at": firestore.SERVER_TIMESTAMP,
            "nc_score": nc_score,
            "ber_score": ber_score,
        }
    )
    return True, "Submission approved." if approve else "Submission rejected."


def delete_submission(submission_id):
    storage.delete_folder(f"submissions/{submission_id}")
    _db().collection("submissions").document(submission_id).delete()
    return True, "Submission deleted."


def get_gallery(event_id=None):
    """Approved submissions only — what the public gallery displays."""
    return get_submissions(event_id=event_id, status="approved")
