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
    per-session thing is the configuration itself, and that lives in the
    checkpointer, not here."""
    return IoTAgenticWorkflow()


class IoTDeploymentUI:
    """Streamlit front end for the workflow, with more than one session.

    A session is just a `thread_id`. The checkpointer holds its whole
    configuration, so all this class keeps is which session is open and a
    small index of titles for the sidebar (R7).
    """

    def __init__(self, page_title: str = "IoT Agentic Deployer", icon: str = "⚙️"):
        self.page_title = page_title
        self.icon = icon
        st.set_page_config(page_title=self.page_title, page_icon=self.icon, layout="wide")
        self.workflow = get_workflow()

    # -- session handling -------------------------------------------------

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

    def _maybe_set_title(self, session_id: str, first_message: str):
        meta = st.session_state.sessions_index[session_id]
        if meta["title"] == "New configuration":
            meta["title"] = first_message[:40] + ("…" if len(first_message) > 40 else "")
            _save_sessions_index(st.session_state.sessions_index)

    @property
    def _session(self) -> str:
        return st.session_state.active_session_id

    def _values(self) -> dict:
        return self.workflow.get_state_values(self._session)

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

            ordered = sorted(
                st.session_state.sessions_index.items(),
                key=lambda kv: kv[1]["created_at"], reverse=True,
            )
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
        where things stand without running the check again."""
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
        if not self.workflow.get_history(self._session):
            st.info(WELCOME)

    def _render_configuration(self):
        """The table and the topology diagram. Both are drawn from the model,
        so there is nothing for the architect to decipher."""
        inst = self._installation()
        if not inst.building:
            return

        # Something for the jump button at the bottom of the page to aim at.
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
                    # This tab is the one that isn't showing when Streamlit
                    # first draws the page, so its iframe is 0x0. Mermaid
                    # measures as it renders, and rendering into nothing bakes
                    # a degenerate viewBox - which is why the diagram came up
                    # blank once you clicked over to it. Waiting until the
                    # element has a real width means it draws at the size it
                    # will actually be seen at.
                    #
                    # htmlLabels lets a label hold the two lines we build for
                    # it; without it the <br/> is dropped and the name and the
                    # type run together as "Classroom 1classroom".
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
        plan is shown, and nothing runs until it is confirmed (thesis 5.4)."""
        values = self._values()
        plan = values.get("deployment_plan")
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

            label = "↻ Retry from the failed operation" if failed else "▶️ Confirm and execute"
            if st.button(label, type="primary", use_container_width=True):
                self._execute_plan()

    def _execute_plan(self):
        """Notes the confirmation and re-enters the graph so the deployment
        agent can get on with it."""
        with st.spinner("Executing the planned operations…"):
            self.workflow.confirm_plan(self._session)
            self._run_turn("I confirm the deployment plan. Execute it.")
        st.rerun()

    # -- interaction ------------------------------------------------------

    def _run_turn(self, user_input: str) -> str | None:
        import traceback
        try:
            return self.workflow.run_turn(self._session, user_input)
        except BaseExceptionGroup as eg:
            # The MCP client runs on anyio task groups, which wrap the real
            # error - on its own, str(eg) only says "unhandled errors in a
            # TaskGroup", which helps nobody.
            st.error("Workflow error: " + "; ".join(
                f"{type(sub).__name__}: {sub}" for sub in eg.exceptions))
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Workflow error: {e}")
            st.code(traceback.format_exc())
        return None

    def _render_history(self):
        for msg in self.workflow.get_history(self._session):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    def _render_jump_to_configuration(self):
        """A way back up to the configuration panel without scrolling.

        The table and the diagram sit above the whole conversation, so once a
        few turns have gone by they are a long way off. This is a plain anchor
        link rather than a Streamlit button: a button would have to rerun the
        script to do anything, and a rerun cannot scroll the page."""
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
        self._render_jump_to_configuration()
        self._handle_user_interaction()