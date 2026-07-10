"""
Cloudinary storage wrapper for the Exhibition/Gallery feature.

Used to store the actual image files (watermarked submissions, the original
cover images, and reference watermark images/keys needed later for admin
verification) — Firestore holds metadata only, never large binary blobs.

REQUIRED Streamlit secret:

    [cloudinary]
    cloud_name = "your-cloud-name"
    api_key = "your-api-key"
    api_secret = "your-api-secret"

Get these from the Cloudinary dashboard (cloudinary.com) after signing up
for a free account: Dashboard -> Product Environment Credentials.
"""

import io
import cloudinary
import cloudinary.uploader
import streamlit as st

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    cloudinary.config(
        cloud_name=st.secrets["cloudinary"]["cloud_name"],
        api_key=st.secrets["cloudinary"]["api_key"],
        api_secret=st.secrets["cloudinary"]["api_secret"],
        secure=True,
    )
    _configured = True


def upload_bytes(data, folder, public_id, resource_type="image"):
    """Uploads raw bytes to Cloudinary and returns the public HTTPS URL.
    resource_type='raw' should be used for non-image binary files (e.g. .pkl keys)."""
    if data is None:
        return None
    _configure()
    result = cloudinary.uploader.upload(
        io.BytesIO(data),
        folder=folder,
        public_id=public_id,
        resource_type=resource_type,
        overwrite=True,
    )
    return result["secure_url"]


def delete_folder(folder):
    """Deletes all assets under a folder (used when an event/submission is removed)."""
    _configure()
    try:
        cloudinary.api.delete_resources_by_prefix(folder)
        cloudinary.api.delete_resources_by_prefix(folder, resource_type="raw")
        cloudinary.api.delete_folder(folder)
    except Exception:
        pass  # best-effort cleanup, not critical if it fails
