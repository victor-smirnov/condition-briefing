"""Chat agent over runs: answers provenance questions and re-runs the plan from any step."""

import json
import re
from typing import Callable

from . import pipeline
from .config import settings
from .llm import client
from .pipeline import STEP_DOCS, STEPS, all_statements

MAX_TOOL_CHARS = 40_000
MAX_ROUNDS = 10

TOOLS = [
    {"type": "function",
        "name": "list_runs", "description": "List all runs with parent, status and change note.",
        "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function",
        "name": "get_plan", "description": "Run overview: parameters of every step, status, one-line summaries, parent run.",
        "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
    {"type": "function",
        "name": "get_step", "description": "Outputs of one step in compact form (source texts omitted; use get_source).",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "step": {"type": "string", "enum": STEPS}}, "required": ["run_id", "step"]}},
    {"type": "function",
        "name": "get_source", "description": "Full record of one source (PMID:... or NCT...): text, retrieval, screening decision, evidence, citing statements.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "source_id": {"type": "string"}}, "required": ["run_id", "source_id"]}},
    {"type": "function",
        "name": "trace", "description": "Provenance chain of a report statement (S#), evidence item (E#) or key player name: quotes, sources, queries, screening reasons, verification.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "item": {"type": "string"}}, "required": ["run_id", "item"]}},
    {"type": "function",
        "name": "search", "description": "Case-insensitive text search over source titles and texts and evidence claims in a run.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["run_id", "text"]}},
    {"type": "function",
        "name": "rerun", "description": "Create a new run derived from run_id, change parameters, and execute from the earliest affected step. Takes from a few seconds (synthesize) to 2 minutes (search_plan). Returns the new run id and a diff.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"},
            "from_step": {"type": "string", "enum": STEPS},
            "param_changes": {"type": "object", "description": "Keyed by step name, e.g. {\"search_plan\": {\"year_from\": 2023}}. Only changed params."},
            "note": {"type": "string", "description": "The user's request in one sentence, stored with the new run."}},
            "required": ["run_id", "from_step", "param_changes", "note"]}},
    {"type": "function",
        "name": "diff_runs", "description": "Compare two runs: sources added/removed, report sections changed, verification.",
        "parameters": {"type": "object", "properties": {
            "run_a": {"type": "string"}, "run_b": {"type": "string"}}, "required": ["run_a", "run_b"]}},
]

SYSTEM = """You are the research assistant inside a tool that produces condition briefings for a
health system strategy team. Each briefing is a run of a fixed research plan:
{steps}

The user is looking at run {run_id}. Its current parameters:
{params}

Rules:
- Answer questions about the report, sources, and method only from tool results. Cite IDs
  (S#, E#, PMID:..., NCT..., query IDs Q#/T#) and give the verbatim quote when asked why
  something is in the report. If the tools do not show it, say so. Do not use your own
  medical knowledge to fill gaps; you may point out that a gap exists.
- To change the research, call rerun with the earliest affected step and only the changed
  params. Mapping: date range, query wording, more or fewer results, new queries ->
  search_plan (year_from, year_to, pubmed_per_query, trials_per_query, instructions,
  extra_pubmed_queries as PubMed query strings, extra_trial_queries as
  {{"condition", "intervention"}} objects); condition interpretation or focus -> scope;
  include or exclude specific sources or change screening criteria -> screen_extract
  (include_ids, exclude_ids, instructions); report emphasis, structure, audience ->
  synthesize (instructions, audience). An "instructions" value replaces the old one, so carry
  over earlier instructions that still apply.
- If a change request is ambiguous, ask one short question before re-running.
- After a rerun, report the new run id and the diff: sources added and removed, sections
  changed, verification results.
- Write plain, short sentences."""


def _compact_step(run_id: str, step: str) -> dict:
    rec = pipeline.load_step(run_id, step)
    if rec is None:
        return {"error": f"step {step} has not run in {run_id}"}
    o = rec["outputs"]
    head = {"step": step, "summary": rec["summary"], "params": rec["params"], "meta": rec["meta"],
            "inherited_from": rec.get("inherited_from")}
    if step == "retrieve":
        body = {"queries": o["queries"],
                "sources": [{"id": s["id"], "type": s["type"], "year": s["year"], "title": s["title"],
                             "found_by": s["found_by"]} for s in o["sources"].values()]}
    elif step == "screen_extract":
        body = {"included": o["included"], "excluded": o["excluded"],
                "assessments": {k: {kk: v[kk] for kk in ("decision", "decided_by", "reason", "category",
                                                           "intervention_class", "evidence_ids")}
                                for k, v in o["assessments"].items()},
                "evidence_count": len(o["evidence"])}
    elif step == "verify":
        body = {"summary": o["summary"],
                "flagged_statements": {k: v for k, v in o["statement_checks"].items() if not v["ok"]},
                "flagged_players": {k: v for k, v in o["player_checks"].items() if not v["ok"]},
                "non_exact_quotes": {k: v for k, v in o["evidence_checks"].items() if v["status"] != "exact"}}
    else:
        body = o
    return {**head, "outputs": body}


def _query_info(queries: list[dict], ids: list[str]) -> list[dict]:
    return [{k: q.get(k) for k in ("id", "kind", "purpose", "query", "url", "retrieved_at", "total_hits")}
            for q in queries if q["id"] in ids]


def _evidence_chain(run_id: str, eid: str, d: dict) -> dict:
    e = d["screen_extract"]["evidence"].get(eid)
    if not e:
        return {"error": f"no evidence {eid}"}
    s = d["retrieve"]["sources"][e["source_id"]]
    a = d["screen_extract"]["assessments"][s["id"]]
    return {
        "evidence": e,
        "quote_check": (d["verify"] or {}).get("evidence_checks", {}).get(eid),
        "source": {"id": s["id"], "type": s["type"], "title": s["title"], "year": s["year"], "url": s["url"],
                   "retrieved_at": s["retrieved_at"], "meta": s["meta"]},
        "screening": {k: a[k] for k in ("decision", "decided_by", "reason", "category")},
        "found_by_queries": _query_info(d["retrieve"]["queries"], s["found_by"]),
    }


def _load(run_id: str) -> dict:
    from .report import load_all
    return load_all(run_id)


def tool_trace(run_id: str, item: str) -> dict:
    d = _load(run_id)
    item = item.strip()
    if re.fullmatch(r"[Ee]\d+", item):
        return _evidence_chain(run_id, item.upper(), d)
    if re.fullmatch(r"[Ss]\d+", item):
        for label, s in all_statements(d["synthesize"]):
            if s["id"] == item.upper():
                return {"statement": s, "section": label,
                        "verification": (d["verify"] or {}).get("statement_checks", {}).get(s["id"]),
                        "evidence": [_evidence_chain(run_id, e, d) for e in s["evidence_ids"]]}
        return {"error": f"no statement {item}"}
    for p in d["synthesize"]["key_players"]:
        if item.lower() in p["name"].lower():
            srcs = d["retrieve"]["sources"]
            return {"player": p, "verification": (d["verify"] or {}).get("player_checks", {}).get(p["name"]),
                    "trials": [{"id": t, "title": srcs[t]["title"], "lead_sponsor": srcs[t]["meta"]["lead_sponsor"],
                                "collaborators": srcs[t]["meta"]["collaborators"], "phases": srcs[t]["meta"]["phases"],
                                "status": srcs[t]["meta"]["status"]} for t in p["trial_ids"] if t in srcs]}
    return {"error": f"nothing called {item}"}


def tool_get_source(run_id: str, source_id: str) -> dict:
    d = _load(run_id)
    s = d["retrieve"]["sources"].get(source_id)
    if not s:
        return {"error": f"no source {source_id} in run {run_id}"}
    a = d["screen_extract"]["assessments"].get(source_id) if d["screen_extract"] else None
    ev = [e for e in (d["screen_extract"] or {}).get("evidence", {}).values() if e["source_id"] == source_id]
    eids = {e["id"] for e in ev}
    citing = [s2["id"] for _, s2 in all_statements(d["synthesize"]) if eids & set(s2["evidence_ids"])] \
        if d["synthesize"] else []
    return {"source": s, "found_by_queries": _query_info(d["retrieve"]["queries"], s["found_by"]),
            "screening": a, "evidence": ev, "cited_by_statements": citing}


def tool_search(run_id: str, text: str) -> dict:
    d = _load(run_id)
    t = text.lower()
    hits = []
    for s in d["retrieve"]["sources"].values():
        if t in s["text"].lower():
            a = (d["screen_extract"] or {}).get("assessments", {}).get(s["id"], {})
            hits.append({"source": s["id"], "title": s["title"], "year": s["year"], "decision": a.get("decision")})
    ev = [{"evidence": e["id"], "source": e["source_id"], "claim": e["claim"]}
          for e in (d["screen_extract"] or {}).get("evidence", {}).values() if t in e["claim"].lower()]
    return {"sources": hits[:20], "sources_total": len(hits), "evidence": ev[:20], "evidence_total": len(ev)}


def tool_get_plan(run_id: str) -> dict:
    run = pipeline.load_run(run_id)
    return {"run_id": run_id, "status": run["status"], "parent": run["parent"], "change_note": run["change_note"],
            "params": run["params"],
            "steps": {s: {k: v for k, v in i.items() if k in ("status", "summary", "duration_s", "inherited_from")}
                      for s, i in run["steps"].items()}}


class Agent:
    def __init__(self, on_progress: Callable[[str, str], None] = lambda s, m: None) -> None:
        self.on_progress = on_progress
        self.new_run_id: str | None = None

    def _call(self, name: str, args: dict) -> object:
        if name == "list_runs":
            return [{"id": r["id"], "created_at": r["created_at"], "status": r["status"],
                     "parent": r["parent"], "change_note": r["change_note"]} for r in pipeline.list_runs()]
        if name == "get_plan":
            return tool_get_plan(args["run_id"])
        if name == "get_step":
            return _compact_step(args["run_id"], args["step"])
        if name == "get_source":
            return tool_get_source(args["run_id"], args["source_id"])
        if name == "trace":
            return tool_trace(args["run_id"], args["item"])
        if name == "search":
            return tool_search(args["run_id"], args["text"])
        if name == "diff_runs":
            return pipeline.diff_runs(args["run_a"], args["run_b"])
        if name == "rerun":
            new_id, start = pipeline.fork(args["run_id"], args["from_step"], args["param_changes"], args.get("note", ""))
            self.on_progress("rerun", f"created {new_id}, executing from {start}")
            pipeline.execute(new_id, start, on_progress=self.on_progress)
            self.new_run_id = new_id
            return {"new_run_id": new_id, "started_from": start, "diff": pipeline.diff_runs(args["run_id"], new_id)}
        return {"error": f"unknown tool {name}"}

    def reply(self, history: list[dict], run_id: str) -> tuple[str, list[dict], list[dict]]:
        """history: Responses API input items (user messages and earlier output items).

        Returns (answer text, items to append to history, tool log).
        """
        steps = "\n".join(f"{i + 1}. {s}: {STEP_DOCS[s]}" for i, s in enumerate(STEPS))

        def instructions(rid: str) -> str:
            return SYSTEM.format(steps=steps, run_id=rid, params=json.dumps(pipeline.load_run(rid)["params"], indent=1))

        system = instructions(run_id)
        new_items: list[dict] = []
        log: list[dict] = []
        for _ in range(MAX_ROUNDS):
            # store=False keeps conversation state on our side; reasoning items come back encrypted.
            r = client().responses.create(
                model=settings().model_main, instructions=system, input=[*history, *new_items], tools=TOOLS,
                reasoning={"effort": "low"}, store=False, include=["reasoning.encrypted_content"],
            )
            new_items += [item.model_dump(exclude_none=True) for item in r.output]
            calls = [item for item in r.output if item.type == "function_call"]
            if not calls:
                return r.output_text, new_items, log
            for call in calls:
                args = json.loads(call.arguments or "{}")
                shown = ", ".join(f"{k}={v}" for k, v in args.items() if k != "run_id")
                self.on_progress("tool", f"{call.name}({shown[:120]})")
                try:
                    result = self._call(call.name, args)
                except Exception as e:
                    result = {"error": str(e)}
                text = json.dumps(result, ensure_ascii=False, default=str)
                if len(text) > MAX_TOOL_CHARS:
                    text = text[:MAX_TOOL_CHARS] + " ...[truncated]"
                log.append({"tool": call.name, "args": args, "result_chars": len(text)})
                new_items.append({"type": "function_call_output", "call_id": call.call_id, "output": text})
            if self.new_run_id:  # later calls in this turn should see the new run
                system = instructions(self.new_run_id)
        return "Stopped after too many tool calls.", new_items, log
