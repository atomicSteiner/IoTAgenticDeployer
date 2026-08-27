"""The chat interface (thesis 5.5).

Where the architect describes the building, answers questions, looks at
summaries, reads through the planned operations and confirms the deployment -
and where earlier configurations can be reopened and carried on with.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.rendering.formatting import (
    installation_to_markdown,
    installation_to_mermaid,
    validation_to_markdown, trace_to_markdown,
)
from iot_agentic_deployer.workflow import IoTAgenticWorkflow

SESSIONS_INDEX_PATH = Path("sessions_index.json")

_UNREAD = object()      # 'not looked up yet', which None cannot mean here

WELCOME = """
👋 **Welcome to the IoT Deployment Assistant.**

I help you conceptualise a building and configure an IoT installation on it.

**To start, describe your building in natural language** — its name, location,
floors and rooms. For example:

> *The Computer Science Building in Camerino has two floors. The ground floor has
> 4 classrooms, 2 laboratories and a technical room. The first floor has 6 offices,
> a meeting room and a corridor.*

Then we choose a use case (wellness / seismic safety / general installation),
select devices from the catalogue, check the configuration, and deploy it.
"""

def _load_sessions_index() -> dict:
    if SESSIONS_INDEX_PATH.exists():
        return json.loads(SESSIONS_INDEX_PATH.read_text())
    return {}


def _save_sessions_index(index: dict):
    SESSIONS_INDEX_PATH.write_text(json.dumps(index, indent=2))


@st.cache_resource
def get_workflow() -> IoTAgenticWorkflow:
    """Builds the graph once and reuses it for every session. The only
    per-session thing is the configuration itself, and that lives in the checkpointer"""
    return IoTAgenticWorkflow()


class IoTDeploymentUI:
    """Streamlit front end for the workflow, with more than one session.

    A session is just a `thread_id`. The checkpointer holds its whole
    configuration, so all this class keeps is which session is open and a
    small index of titles for the sidebar.
    """

    def __init__(self, page_title: str = "IoT Agentic Deployer", icon: str = "⚙️"):
        self.page_title = page_title
        self.icon = icon
        st.set_page_config(page_title=self.page_title, page_icon=self.icon, layout="wide")
        self.workflow = get_workflow()
        self._cached_values = None
        self._cached_approval = _UNREAD
        self._cached_times = None

    # -- session handling -------------------------------------------------

    #select the most recent session or create a new one if there is no one
    def _initialize_state(self):
        if "sessions_index" not in st.session_state:
            st.session_state.sessions_index = _load_sessions_index()

        if "active_session_id" not in st.session_state:
            if st.session_state.sessions_index:
                st.session_state.active_session_id = max(
                    st.session_state.sessions_index,
                    key=lambda sid: st.session_state.sessions_index[sid]["created_at"],
                )
            else:
                self._start_new_session()

    def _start_new_session(self):
        session_id = str(uuid.uuid4())
        st.session_state.sessions_index[session_id] = {
            "title": "New configuration",
            "created_at": datetime.utcnow().isoformat(),
        }
        st.session_state.active_session_id = session_id
        _save_sessions_index(st.session_state.sessions_index)

    def _delete_session(self, session_id: str):
        self.workflow.delete_session(session_id)
        del st.session_state.sessions_index[session_id]
        _save_sessions_index(st.session_state.sessions_index)

        if st.session_state.active_session_id == session_id:
            if st.session_state.sessions_index:
                st.session_state.active_session_id = max(
                    st.session_state.sessions_index,
                    key=lambda sid: st.session_state.sessions_index[sid]["created_at"],
                )
            else:
                self._start_new_session()

    #set the new title of the session
    def _maybe_set_title(self, session_id: str, first_message: str):
        meta = st.session_state.sessions_index[session_id]
        if meta["title"] == "New configuration":
            meta["title"] = first_message[:40] + ("…" if len(first_message) > 40 else "")
            _save_sessions_index(st.session_state.sessions_index)

    @property
    def _session(self) -> str:
        return st.session_state.active_session_id

    def _values(self) -> dict:
        # Streamlit rebuilds this class on every rerun, so the cache lasts
        # all six panels see the same state,
        if self._cached_values is None:
            self._cached_values = self.workflow.get_state_values(self._session)
        return self._cached_values

    #the plan waiting on the architect, read once per rerun like the state is
    def _approval(self) -> dict | None:
        if self._cached_approval is _UNREAD:
            self._cached_approval = self.workflow.pending_approval(self._session)
        return self._cached_approval

    #when each message arrived, read once per rerun like the state is
    def _times(self) -> dict:
        if self._cached_times is None:
            self._cached_times = self.workflow.message_times(self._session)
        return self._cached_times

    @staticmethod
    def _stamp(iso: str | None) -> str:
        """The hour on its own for today, the date as well for anything
        older: what a phone does, so a long session does not repeat the
        same date under every line"""
        when = datetime.fromisoformat(iso).astimezone() if iso else datetime.now()
        return when.strftime("%H:%M" if when.date() == datetime.now().date()
                             else "%d %b, %H:%M")

    def _installation(self) -> Installation:
        return Installation(**self._values().get("installation", {}))

    # -- sidebar ----------------------------------------------------------

    def _render_sidebar(self):
        with st.sidebar:
            if st.button("➕ New configuration", use_container_width=True):
                self._start_new_session()
                st.rerun()

            st.divider()
            st.caption("Previous configurations")
            if "confirm_delete_id" not in st.session_state:
                st.session_state.confirm_delete_id = None

            #sort the sessions
            ordered = sorted(
                st.session_state.sessions_index.items(),
                key=lambda kv: kv[1]["created_at"], reverse=True,
            )
            #process every session to open them or delete if necessary
            for session_id, meta in ordered:
                label = ("👉 " if session_id == self._session else "") + meta["title"]
                col_open, col_delete = st.columns([5, 1])
                with col_open:
                    if st.button(label, key=f"session_{session_id}", use_container_width=True):
                        st.session_state.active_session_id = session_id
                        st.rerun()
                with col_delete:
                    if st.button("🗑️", key=f"delete_{session_id}", use_container_width=True):
                        st.session_state.confirm_delete_id = session_id
                        st.rerun()
                #if there is an id that is selected to be deleted
                if st.session_state.confirm_delete_id == session_id:
                    st.caption(f"Delete '{meta['title']}'? This cannot be undone.")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Confirm", key=f"confirm_delete_{session_id}",
                                     use_container_width=True):
                            self._delete_session(session_id)
                            st.session_state.confirm_delete_id = None
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_delete_{session_id}",
                                     use_container_width=True):
                            st.session_state.confirm_delete_id = None
                            st.rerun()

            self._render_status()

    def _render_status(self):
        """Which platform, and whether it is ready - so the architect can see
        where things stand without running the check again"""
        values = self._values()
        inst = Installation(**values.get("installation", {}))

        st.divider()
        st.caption("Installation")
        st.markdown(
            f"**Use case:** {inst.use_case or '—'}  \n"
            f"**Target platform:** {inst.target.platform}"
        )

        report = values.get("validation_report")
        if report:
            st.caption("Validation")
            if report["deployable"]:
                st.success(f"Deployable · {report['warnings']} warning(s)")
            else:
                st.error(f"{report['errors']} error(s) · {report['warnings']} warning(s)")
            with st.expander("Findings"):
                st.markdown(validation_to_markdown(report))

        trace = values.get("trace")
        if trace:
            with st.expander(f"Activity trace ({len(trace)})"):
                st.markdown(trace_to_markdown(trace))

    # -- main panels ------------------------------------------------------

    def _render_welcome(self):
        if not self._values().get("messages"):
            st.info(WELCOME)

    def _render_configuration(self):
        """The table and the topology diagram. Both are drawn from the model,
        so there is nothing for the architect to decipher."""
        inst = self._installation()
        if not inst.building:
            return

        # Something for the jump button at the bottom of the page to aim at
        st.markdown('<div id="configuration-top"></div>', unsafe_allow_html=True)

        spaces = sum(1 for _ in inst.iter_spaces())
        header = (f"🏢 {inst.building.name} — {spaces} space(s), "
                  f"{len(inst.all_devices())} device(s)")

        with st.expander(header, expanded=True):
            table_tab, diagram_tab = st.tabs(["Configuration", "Topology"])
            with table_tab:
                st.markdown(installation_to_markdown(inst))
            with diagram_tab:
                diagram = installation_to_mermaid(inst)
                if diagram:
                    components.html(
                        f"""
                        <div class="mermaid">{diagram}</div>
                        <script type="module">
                          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
                          mermaid.initialize({{
                            startOnLoad: false,
                            flowchart: {{ htmlLabels: true, useMaxWidth: true }},
                          }});
                          const el = document.querySelector('.mermaid');
                          const timer = setInterval(async () => {{
                            if (!el.offsetWidth) return;      // still hidden
                            clearInterval(timer);
                            await mermaid.run({{ nodes: [el] }});
                          }}, 100);
                        </script>
                        """,
                        height=460, scrolling=True,
                    )

    def _render_deployment_plan(self):
        """Going from planning to actually doing it is a deliberate step: the
        plan is shown, and nothing runs until it is confirmed."""
        # While the graph is suspended the plan lives in the interrupt payload,
        # not in the state: nothing written before an interrupt is kept
        approval = self._approval()
        plan = approval["operations"] if approval else self._values().get("deployment_plan")
        if not plan:
            return

        completed = sum(1 for op in plan if op["status"] == "completed")
        failed = [op for op in plan if op["status"] == "failed"]
        pending = [op for op in plan if op["status"] == "pending"]

        if not pending and not failed:
            st.success(f"✅ Deployment completed — {completed} operation(s) performed.")
            return

        title = ("⚠️ Deployment interrupted" if failed
                 else f"📋 Planned operations ({len(plan)})")
        with st.expander(title, expanded=True):
            if failed:
                st.error(f"Rejected: {failed[0]['description']}\n\n{failed[0]['error']}")
                st.caption(
                    f"{completed} of {len(plan)} operation(s) completed. The entities "
                    "already created were left untouched; confirming again resumes from "
                    "the operation that failed."
                )

            icons = {"pending": "·", "completed": "✅", "failed": "❌", "skipped": "⏭️"}
            st.markdown("\n".join(
                f"{i}. {icons[op['status']]} {op['description']}"
                for i, op in enumerate(plan, 1)
            ))

            if approval:
                st.info("Reply **confirm** in the chat to execute, or ask for changes.")
            elif failed:
                st.info("Ask to deploy again to resume from the operation that failed.")

    # -- interaction ------------------------------------------------------

    def _run_turn(self, user_input: str) -> str | None:
        import traceback
        try:
            # A plan waiting on the architect turns the next message into the
            # answer to it. Reading that answer is the deployment agent's job,
            # so it goes through untouched
            if self.workflow.pending_approval(self._session):
                return self.workflow.resume(self._session, user_input)
            return self.workflow.run_turn(self._session, user_input)
        except BaseExceptionGroup as eg:
            st.error("Workflow error: " + "; ".join(
                f"{type(sub).__name__}: {sub}" for sub in eg.exceptions))
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Workflow error: {e}")
            st.code(traceback.format_exc())
        return None

    #render each history messages of the workflow
    def _render_history(self):
        times = self._times()
        for msg in self._values().get("messages", []):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)
                st.caption(self._stamp(times.get(msg.id)))

    def _render_pending_approval(self):
        """The request to confirm, at the foot of the conversation.

        It cannot come from the history: the graph is suspended inside the
        deployment agent, which has not returned and so has written nothing.
        Drawn live instead, it is there for exactly as long as the plan is
        waiting and goes as soon as the architect answers."""
        approval = self._approval()
        if not approval:
            return
        with st.chat_message("assistant"):
            st.markdown(
                f"[Deployment] {len(approval['operations'])} operation(s) planned on "
                f"**{approval['platform']}**. Nothing has been written yet.\n\n"
                "Reply **confirm** to execute, or ask for changes.")
            st.caption(self._stamp(None))

    def _render_jump_to_configuration(self):
        """A way back up to the configuration panel without scrolling."""
        if not self._installation().building:
            return

        st.markdown(
            """
            <style>
              .jump-to-configuration {
                /* clear of the chat input, which is pinned to the bottom */
                position: fixed; right: 2.5rem; bottom: 9.5rem; z-index: 1000;
                display: inline-flex; align-items: center; gap: .4rem;
                padding: .45rem .9rem; border-radius: 2rem;
                font-size: .85rem; font-weight: 600; text-decoration: none;
                color: inherit;
                background: rgba(128, 128, 128, .18);
                border: 1px solid rgba(128, 128, 128, .35);
                backdrop-filter: blur(6px);
              }
              .jump-to-configuration:hover {
                background: rgba(128, 128, 128, .32);
                text-decoration: none;
              }
            </style>
            <a class="jump-to-configuration" href="#configuration-top">
              ⬆️ Configuration
            </a>
            """,
            unsafe_allow_html=True,
        )

    #this is the core of the interface interaction
    def _handle_user_interaction(self):
        user_input = st.chat_input(
            "Describe the building, choose devices, or ask for a summary…")
        if not user_input:
            return

        self._maybe_set_title(self._session, user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("The supervisor is delegating…"):
                reply = self._run_turn(user_input)
            if reply is None:
                return
            st.markdown(reply)

        # The turn may have produced a plan or changed the model, so redraw
        # and let the panels above catch up.
        st.rerun()

    def render(self):
        st.title(f"{self.icon} IoT Agentic Deployment")

        self._initialize_state()
        self._render_sidebar()
        self._render_welcome()
        self._render_configuration()
        self._render_deployment_plan()
        self._render_history()
        self._render_pending_approval()
        self._render_jump_to_configuration()
        #the input is the last one after all the graphical rendering
        self._handle_user_interaction()