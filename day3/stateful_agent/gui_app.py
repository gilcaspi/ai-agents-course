import uuid
import requests
import streamlit as st

# Base URL of your ADK API server
API_BASE_URL = "http://127.0.0.1:8000"

# App name as exposed by `adk api_server .`
APP_NAME = "StatefulApp"

API_RUN_URL = f"{API_BASE_URL}/run"
API_RESUME_URL = f"{API_BASE_URL}/resume"

USER_ID = "local-user-1"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    # {"invocationId": "...", "hint": "...", "payload": {...}}
    st.session_state.pending = None

st.title("My ADK Agent (stateful)")

# ---------------- Chat history ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- Pending approval UI (HITL placeholder) ----------------

if st.session_state.pending:
    p = st.session_state.pending
    st.warning(p.get("hint", "Pending approval"))

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Approve"):
            resp = requests.post(
                API_RESUME_URL,
                json={
                    "appName": APP_NAME,
                    "userId": USER_ID,
                    "sessionId": st.session_state.session_id,
                    "invocationId": p["invocationId"],
                    # TODO: adapt to your actual resume schema if different
                    "confirmation": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            st.session_state.pending = None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"RESUME RESPONSE:\n```json\n{data}\n```",
                }
            )
            st.rerun()

    with col2:
        if st.button("Reject"):
            resp = requests.post(
                API_RESUME_URL,
                json={
                    "appName": APP_NAME,
                    "userId": USER_ID,
                    "sessionId": st.session_state.session_id,
                    "invocationId": p["invocationId"],
                    "confirmation": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            st.session_state.pending = None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"RESUME RESPONSE:\n```json\n{data}\n```",
                }
            )
            st.rerun()


# ---------------- Helper: extract assistant text from /run response ----------------

def extract_assistant_text(run_response_json) -> str:
    data = run_response_json

    # ADK usually returns a list of events, or a dict with "events"
    if isinstance(data, dict) and "events" in data and isinstance(data["events"], list):
        events = data["events"]
    elif isinstance(data, list):
        events = data
    else:
        return f"Raw response:\n```json\n{data}\n```"

    last_text = None

    for ev in events:
        if not isinstance(ev, dict):
            continue

        ev_type = (
            ev.get("eventType")
            or ev.get("event_type")
            or ev.get("type")
        )
        ev_data = ev.get("data", {})
        content = ev.get("content") or ev_data.get("content") or {}

        # Common event types for model/agent output
        if ev_type in ("agent_output", "model_output", "assistant_message"):
            parts = content.get("parts", [])
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    last_text = p["text"]

    if last_text is not None:
        return last_text

    return f"Raw response (no explicit agent text found):\n```json\n{data}\n```"


# ---------------- Chat input ----------------

if user_input := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    payload = {
        "appName": APP_NAME,
        "userId": USER_ID,
        "sessionId": st.session_state.session_id,
        "newMessage": {
            "role": "user",
            "parts": [
                {"text": user_input},
            ],
        },
    }

    resp = requests.post(API_RUN_URL, json=payload)
    resp.raise_for_status()
    run_json = resp.json()

    assistant_text = extract_assistant_text(run_json)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )

    st.rerun()
