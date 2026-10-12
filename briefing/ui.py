"""Streamlit UI: start a run, review every step, read the report, talk to the agent."""

import pandas as pd
import streamlit as st

from briefing import pipeline, report
from briefing.agent import Agent
from briefing.pipeline import STEP_DOCS, STEPS

st.set_page_config(page_title="Condition Briefing", layout="wide")
ss = st.session_state
ss.setdefault("run_id", None)
ss.setdefault("history", [])  # Responses API items for the agent
ss.setdefault("chat", [])  # what the chat panel shows: {"role", "content", "tools", "run_id"}


SUGGESTIONS = [
    "Why is S2 in the report? Show the quotes.",
    "Which sources were excluded, and why?",
    "Re-run using only studies and trials from 2024 onward.",
]


def progress_writer(box):
    def on_progress(step: str, msg: str) -> None:
        n = f"{STEPS.index(step) + 1}/{len(STEPS)} " if step in STEPS else ""
        box.write(f"**{n}{step}**: {msg}")
        if step in STEPS and msg == "started":
            box.update(label=f"Step {n}{step}...")
    return on_progress


def api_error(e: Exception) -> str:
    """Short, readable error text for the user."""
    text = str(e)
    if "insufficient_quota" in text or "credit_balance_exhausted" in text:
        return "the OpenAI account has no credits left (429 insufficient_quota). Add credits and try again."
    return text.splitlines()[0][:300]


def select_run(rid: str) -> None:
    """Switch the current run on the next script run (the selectbox owns its value)."""
    ss.pending_run = rid


# ------------------------------------------------------------------ sidebar

welcome = st.empty()  # main-area message while there are no runs
with st.sidebar:
    st.header("New research")
    with st.form("new"):
        condition = st.text_input("Condition", placeholder="e.g. type 2 diabetes")
        focus = st.text_input("Focus", value="US health system strategy team")
        years = st.slider("Publication / trial start years", 2010, 2026, (2021, 2026))
        go = st.form_submit_button("Start research", type="primary")
    if go and condition.strip():
        rid = pipeline.create_run(condition.strip(), focus, {"search_plan": {"year_from": years[0], "year_to": years[1]}})
        with st.status(f"Running {rid}", expanded=True) as box:
            try:
                pipeline.execute(rid, on_progress=progress_writer(box))
                box.update(label="Done", state="complete")
            except Exception as e:
                box.update(label=f"Failed: {api_error(e)}", state="error")
        select_run(rid)
        st.rerun()

    runs = pipeline.list_runs()
    if not runs:
        welcome.info("Enter a condition in the sidebar and press Start research. A run takes about 2 minutes "
                     "and produces a cited briefing you can question and re-run step by step.")
        st.stop()
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    if "pending_run" in ss:
        ss.run_select = ss.pop("pending_run")
    if ss.get("run_select") not in by_id:  # open on the newest finished run
        ss.run_select = next((r["id"] for r in runs if r["status"] == "done"), ids[0])

    def label(rid: str) -> str:
        r = by_id[rid]
        when = r["created_at"][5:16].replace("T", " ")
        text = f"{r['title']} · {when}"
        if r["parent"]:
            note = (r["change_note"] or "")[:40]
            text += f" · re-run from {r['parent']['fork_step']}" + (f": {note}" if note else "")
        if r["status"] != "done":
            text += f" [{r['status']}]"
        return text

    st.header("Runs")
    st.selectbox("Current run", ids, format_func=label, key="run_select")
    ss.run_id = ss.run_select

    st.caption("Lineage")
    chain, cur = [], by_id[ss.run_id]
    while cur:
        chain.append(cur)
        cur = by_id.get(cur["parent"]["run_id"]) if cur["parent"] else None
    for r in reversed(chain):
        line = f"`{r['id']}`"
        if r["parent"]:
            line += f"  \nre-run from **{r['parent']['fork_step']}**: {r['change_note'] or ''}"
        st.markdown(line)

run = pipeline.load_run(ss.run_id)
rid = run["id"]

# --------------------------------------------------------------- main panel

left, right = st.columns([3, 2], gap="large")

with left:
    t_report, t_plan, t_evidence = st.tabs(["Report", "Plan and steps", "Sources and evidence"])

    with t_report:
        if run["steps"]["verify"]["status"] in ("done", "inherited"):
            md = report.render_markdown(rid)
            c1, c2 = st.columns(2)
            c1.download_button("Download report (Markdown)", md, f"{rid}.md", "text/markdown")
            c2.download_button("Download provenance bundle (zip)", report.provenance_bundle(rid), f"{rid}.zip",
                               "application/zip")
            st.markdown(md, unsafe_allow_html=True)
        else:
            st.warning(f"Run status: {run['status']}. {run.get('error') or ''}")

    with t_plan:
        if run["parent"]:
            st.info(f"Derived from `{run['parent']['run_id']}`, re-run from **{run['parent']['fork_step']}**. "
                    f"{run['change_note'] or ''}")
        for i, s in enumerate(STEPS, 1):
            info = run["steps"][s]
            rec = pipeline.load_step(rid, s)
            status = info["status"]
            title = f"{i}. {s} · {status}"
            if info.get("duration_s") is not None:
                title += f" · {info['duration_s']}s"
            if info.get("summary"):
                title += f" · {info['summary']}"
            with st.expander(title):
                st.caption(STEP_DOCS[s])
                if info.get("inherited_from"):
                    st.caption(f"Copied unchanged from run `{info['inherited_from']}`.")
                if info.get("error"):
                    st.error(info["error"])
                p1, p2 = st.columns(2)
                p1.markdown("**Parameters**")
                p1.json(run["params"][s], expanded=True)
                if rec:
                    p2.markdown("**Execution**")
                    p2.json(rec["meta"], expanded=False)
                if not rec:
                    continue
                o = rec["outputs"]
                if s == "scope":
                    st.json(o)
                elif s == "search_plan":
                    st.write(o["rationale"])
                    st.dataframe(pd.DataFrame(o["pubmed"] + o["trials"]), hide_index=True)
                elif s == "retrieve":
                    q = pd.DataFrame([{k: x.get(k) for k in ("id", "kind", "purpose", "query", "total_hits",
                                                            "retrieved_at", "url", "error")} | {"used": len(x["returned"])}
                                      for x in o["queries"]])
                    st.dataframe(q, hide_index=True, column_config={"url": st.column_config.LinkColumn()})
                    st.dataframe(pd.DataFrame([{"id": x["id"], "type": x["type"], "year": x["year"], "title": x["title"],
                                                "found_by": ", ".join(x["found_by"]), "url": x["url"]}
                                               for x in o["sources"].values()]),
                                 hide_index=True, column_config={"url": st.column_config.LinkColumn()})
                elif s == "screen_extract":
                    df = pd.DataFrame([{"source": k, "decision": v["decision"], "by": v["decided_by"],
                                        "category": v["category"], "class": v["intervention_class"],
                                        "evidence": len(v["evidence_ids"]), "reason": v["reason"]}
                                       for k, v in o["assessments"].items()])
                    st.dataframe(df.sort_values(["decision", "source"]), hide_index=True)
                elif s == "aggregate":
                    a1, a2 = st.columns(2)
                    a1.markdown("Trials by phase")
                    a1.dataframe(pd.DataFrame([{"phase": k, "trials": len(v)} for k, v in o["trials_by_phase"].items()]),
                                 hide_index=True)
                    a2.markdown("Trials by intervention class")
                    a2.dataframe(pd.DataFrame([{"class": k, "trials": len(v)} for k, v in o["trials_by_class"].items()]),
                                 hide_index=True)
                    st.markdown("Top sponsors")
                    st.dataframe(pd.DataFrame([{"name": d["name"], "class": d["class"], "lead": len(d["lead"]),
                                                "collaborator": len(d["collaborator"]),
                                                "trials": ", ".join(d["lead"] + d["collaborator"])}
                                               for d in o["top_sponsors"]]), hide_index=True)
                elif s == "synthesize":
                    st.dataframe(pd.DataFrame([{"id": x["id"], "section": label, "basis": x["basis"],
                                                "evidence": ", ".join(x["evidence_ids"]), "text": x["text"]}
                                               for label, x in pipeline.all_statements(o)]), hide_index=True)
                elif s == "verify":
                    st.json(o["summary"])
                    flagged = {k: v for k, v in o["statement_checks"].items() if not v["ok"]}
                    flagged |= {k: v for k, v in o["player_checks"].items() if not v["ok"]}
                    if flagged:
                        st.json(flagged)
                    near = {k: v for k, v in o["evidence_checks"].items() if v["status"] != "exact"}
                    if near:
                        st.markdown("Quotes that are not exact matches")
                        st.json(near)

    with t_evidence:
        se, ret, ver = (pipeline.load_step(rid, s) for s in ("screen_extract", "retrieve", "verify"))
        if se and ret:
            srcs = ret["outputs"]["sources"]
            checks = ver["outputs"]["evidence_checks"] if ver else {}
            ev = pd.DataFrame([{"id": e["id"], "source": e["source_id"], "topic": e["topic"], "claim": e["claim"],
                                "quote": e["quote"], "quote check": checks.get(e["id"], {}).get("status")}
                               for e in se["outputs"]["evidence"].values()])
            st.dataframe(ev, hide_index=True, height=320)
            sid = st.selectbox("Inspect source", sorted(srcs))
            s = srcs[sid]
            a = se["outputs"]["assessments"].get(sid, {})
            st.markdown(f"**[{sid}]({s['url']})** {s['title']} ({s['year']})  \n"
                        f"Retrieved {s['retrieved_at']} by {', '.join(s['found_by'])}. "
                        f"Screening: **{a.get('decision')}** ({a.get('decided_by')}): {a.get('reason')}")
            st.text(s["text"])
            st.json(s["meta"], expanded=False)

with right:
    st.subheader("Research assistant")
    st.caption(f"Ask about sources and method, or ask for changes. Talking about `{rid}`.")
    box = st.container(height=640)
    suggested = None
    if not ss.chat:
        with box:
            st.caption("Try:")
            for s in SUGGESTIONS:
                if st.button(s, key=f"sugg-{s}"):
                    suggested = s
    for m in ss.chat:
        with box.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                with st.expander(f"{len(m['tools'])} tool calls"):
                    for t in m["tools"]:
                        st.code(f"{t['tool']}({t['args']})", language="python")
    q = st.chat_input("e.g. Why is S4 in the report? / Only use trials from 2023 on") or suggested
    if q:
        ss.chat.append({"role": "user", "content": q})
        ss.history.append({"role": "user", "content": q})
        with box.chat_message("user"):
            st.markdown(q)
        with box.chat_message("assistant"):
            with st.status("Working", expanded=False) as stat:
                agent = Agent(on_progress=progress_writer(stat))
                try:
                    ans, items, log = agent.reply(ss.history, rid)
                    ss.history += items
                    stat.update(label="Done", state="complete")
                except Exception as e:
                    ans, log = f"Error: {api_error(e)}", []
                    ss.history.pop()
                    stat.update(label="Failed", state="error")
        ss.chat.append({"role": "assistant", "content": ans, "tools": log})
        if agent.new_run_id:
            select_run(agent.new_run_id)
        st.rerun()
