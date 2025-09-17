import os, json, tempfile, streamlit as st

def activate_adc_from_secrets() -> str:
    info = dict(st.secrets["firebase"])

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", info["project_id"])
    os.environ.setdefault("GCLOUD_PROJECT", info["project_id"])
    return path