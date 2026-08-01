import os
import json
import firebase_admin
from firebase_admin import credentials

# Same pattern as the working Zindryx/Gemini app's firebase_config.py.
# Reads a full service-account JSON blob from an env var in production
# (Render), falls back to a local key file for local/dev testing.

firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

if firebase_creds_json:
    cred_dict = json.loads(firebase_creds_json)
    cred = credentials.Certificate(cred_dict)
else:
    cred = credentials.Certificate("service-account-key.json")

# Guard against "app already initialized" if this module gets imported
# more than once (e.g. by both memory_service and a future auth module).
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
