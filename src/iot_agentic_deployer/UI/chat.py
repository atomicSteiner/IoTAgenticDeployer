import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

from iot_agentic_deployer.workflow import IoTAgenticWorkflow

CHATS_INDEX_PATH = Path("chats_index.json")


def _load_chats_index() -> dict:
    if CHATS_INDEX_PATH.exists():
        return json.loads(CHATS_INDEX_PATH.read_text())
    return {}


def _save_chats_index(index: dict):
    CHATS_INDEX_PATH.write_text(json.dumps(index, indent=2))


def _dt_data_to_dataframe(dt_data: list) -> pd.DataFrame:
    """Flattens the nested device/attributes/telemetry structure from
    ThingsBoard into one row per device, one column per attribute/telemetry
    key - the shape a table actually needs, instead of asking the LLM to
    reformat raw JSON into markdown by hand."""
    rows = []
    for device in dt_data:
        row = {
            "Name": device.get("name"),
            "Type": device.get("type"),
            "Label": device.get("label"),
        }
        for attr in device.get("attributes", []):
            row[attr.get("key")] = attr.get("value")
        for key, values in device.get("telemetry", {}).items():
            if values:
                row[key] = values[0].get("value")
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_resource
def get_workflow() -> IoTAgenticWorkflow:
    """Builds the LangGraph workflow once and reuses it across all sessions."""
    return IoTAgenticWorkflow()


class IoTDeploymentUI:
    """Manages the Streamlit GUI for the Agentic Workflow, with multi-chat support."""

    def __init__(self, page_title: str = "IoT Agentic Deployer", icon: str = "⚙️"):
        self.page_title = page_title
        self.icon = icon
        st.set_page_config(page_title=self.page_title, page_icon=self.icon)
        self.workflow = get_workflow()

    def _initialize_state(self):
        if "chats_index" not in st.session_state:
            st.session_state.chats_index = _load_chats_index()

        if "active_chat_id" not in st.session_state:
            if st.session_state.chats_index:
                most_recent = max(
                    st.session_state.chats_index,
                    key=lambda cid: st.session_state.chats_index[cid]["created_at"],
                )
                st.session_state.active_chat_id = most_recent
            else:
                self._start_new_chat()

    def _start_new_chat(self):
        chat_id = str(uuid.uuid4())
        st.session_state.chats_index[chat_id] = {
            "title": "New chat",
            "created_at": datetime.utcnow().isoformat(),
        }
        st.session_state.active_chat_id = chat_id
        _save_chats_index(st.session_state.chats_index)

    def _maybe_set_title(self, chat_id: str, first_message: str):
        meta = st.session_state.chats_index[chat_id]
        if meta["title"] == "New chat":
            meta["title"] = first_message[:40] + ("…" if len(first_message) > 40 else "")
            _save_chats_index(st.session_state.chats_index)

    def _render_sidebar(self):
        with st.sidebar:
            st.caption(f"ThingsBoard: {self.workflow.tb_client.base_url}")

            if st.button("➕ New chat", use_container_width=True):
                self._start_new_chat()
                st.rerun()

            st.divider()
            st.caption("Previous chats")

            ordered_chats = sorted(
                st.session_state.chats_index.items(),
                key=lambda kv: kv[1]["created_at"],
                reverse=True,
            )
            for chat_id, meta in ordered_chats:
                is_active = chat_id == st.session_state.active_chat_id
                label = ("👉 " if is_active else "") + meta["title"]
                if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()

    def _render_chat_history(self):
        messages = self.workflow.get_history(st.session_state.active_chat_id)
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    def _render_devices_table(self):
        """Shows the latest retrieved devices as a real table, deterministically
        built from state - not something the LLM has to format itself. Kept
        outside the chat flow so it stays visible/up to date across reruns."""
        dt_data = self.workflow.get_dt_data(st.session_state.active_chat_id)
        if not dt_data:
            return

        with st.expander(f"📋 Devices ({len(dt_data)})", expanded=True):
            st.dataframe(_dt_data_to_dataframe(dt_data), use_container_width=True)

    def _handle_user_interaction(self):
        user_input = st.chat_input("Enter your operational intent (e.g., deploy 5 sensors)...")
        if not user_input:
            return

        chat_id = st.session_state.active_chat_id
        self._maybe_set_title(chat_id, user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("The Orchestrator is processing..."):
                try:
                    reply = self.workflow.run_turn(chat_id, user_input)
                except Exception as e:
                    st.error(f"Workflow error: {e}")
                    return
            st.markdown(reply)

    def render(self):
        """The main execution method that builds the UI."""
        st.title(f"{self.icon} IoT Agentic Deployment Interface")

        self._initialize_state()
        self._render_sidebar()
        self._render_devices_table()
        self._render_chat_history()
        self._handle_user_interaction()