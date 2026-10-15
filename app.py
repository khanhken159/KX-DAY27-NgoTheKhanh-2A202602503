"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()

st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def get_recent_sessions() -> list[dict]:
    """Return recent sessions with worst risk and latest audit timestamp."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level
                       WHEN 'low' THEN 1 WHEN 'med' THEN 2 WHEN 'high' THEN 3
                       ELSE 0 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' WHEN 1 THEN 'low'
                       ELSE 'unknown' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 10
            """
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]


async def restore_session(thread_id: str) -> tuple[dict, dict | None]:
    """Read the checkpoint so a selected sidebar session can be reopened."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(cfg)
        values = dict(snapshot.values or {})
        payload = None
        for task in snapshot.tasks:
            for paused in task.interrupts:
                payload = paused.value
                break
            if payload is not None:
                break
        return values, payload


with st.sidebar:
    st.header("Recent sessions")
    sessions = asyncio.run(get_recent_sessions())
    if not sessions:
        st.caption("No review sessions yet.")
    for session in sessions:
        thread_id = session["thread_id"]
        with st.expander(f"{session['pr_url']}", expanded=False):
            st.caption(
                f"Risk: {session['worst_risk']} · Events: {session['events']}\n\n"
                f"Last event: {session['last_event']}\n\nThread: {thread_id}"
            )
            if st.button("Open session", key=f"open-{thread_id}"):
                st.session_state.thread_id = thread_id
                st.session_state.pr_url = session["pr_url"]
                values, payload = asyncio.run(restore_session(thread_id))
                st.session_state.interrupt_payload = payload
                st.session_state.final = values if values.get("final_action") else None
                st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """Show a proposed review with three choices; return the resume value on click."""
    conf = payload["confidence"]
    if payload.get("stage") == "refined_review_confirmation":
        st.subheader(f"Confirm refined review — confidence {conf:.0%}")
    else:
        st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input(
        "Feedback (optional)",
        key=f"approval-feedback-{st.session_state.thread_id}",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        if not feedback.strip():
            st.warning("Add feedback to guide the edit before selecting Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for index, question in enumerate(payload.get("questions", [])):
            answers[question] = st.text_area(
                question,
                key=f"escalation-answer-{st.session_state.thread_id}-{index}",
            )
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await graph.ainvoke(
                {
                    "pr_url": pr_url,
                    "thread_id": thread_id,
                    "confirm_escalated_review": True,
                },
                cfg,
            )
        return await graph.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    for key in list(st.session_state):
        if key.startswith("approval-feedback-") or key.startswith("escalation-answer-"):
            del st.session_state[key]

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action in {"auto_committed", "committed"}:
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        analysis = final.get("analysis")
        if analysis is not None:
            st.metric("Review confidence", f"{analysis.confidence:.0%}")
            st.markdown(analysis.summary)
        if final.get("posted_comment_url"):
            st.link_button("View comment on GitHub", final["posted_comment_url"])
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif action == "edit_requires_feedback":
        st.warning("The edit was not submitted. Start a new review and provide edit feedback.")
    elif action.endswith("failed"):
        st.error(f"Could not post the review ({action}). Check the GitHub token and try again.")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
