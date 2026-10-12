"""Render a finished run as a Markdown briefing and package its full provenance."""

import io
import json
import zipfile

from . import pipeline
from .pipeline import STEPS, all_statements


def load_all(run_id: str) -> dict:
    st = pipeline.store()
    out = {"run": pipeline.load_run(run_id, st)}
    for s in STEPS:
        rec = pipeline.load_step(run_id, s, st)
        out[s] = rec["outputs"] if rec else None
    return out


def source_numbers(rep: dict, evidence: dict) -> dict[str, int]:
    """Number cited sources in order of first citation."""
    nums: dict[str, int] = {}
    for _, s in all_statements(rep):
        for eid in s["evidence_ids"]:
            sid = evidence.get(eid, {}).get("source_id")
            if sid and sid not in nums:
                nums[sid] = len(nums) + 1
    return nums


def render_markdown(run_id: str) -> str:
    d = load_all(run_id)
    run, rep, se, srcs, ver = d["run"], d["synthesize"], d["screen_extract"], d["retrieve"]["sources"], d["verify"]
    ev = se["evidence"]
    nums = source_numbers(rep, ev)
    checks = ver["statement_checks"] if ver else {}
    ev_checks = ver["evidence_checks"] if ver else {}

    def cite(sid: str, eids: list[str]) -> str:
        """Link to the source itself; hovering shows the title and the quote this statement relies on."""
        src = srcs[sid]
        eid = next((e for e in eids if ev[e]["source_id"] == sid), None)
        # For near matches show the source's own wording, not the model's edited quote.
        quote = (ev_checks.get(eid, {}).get("source_span") or ev[eid]["quote"]) if eid else ""
        tip = f"{sid}: {src['title']} ({src['year']}). Quote: {quote}"
        tip = " ".join(tip.replace('"', "'").split())[:400]
        return f'[[{nums[sid]}]]({src["url"]} "{tip}")'

    def stmt(s: dict) -> str:
        eids = [e for e in s["evidence_ids"] if e in ev]
        cited = sorted({ev[e]["source_id"] for e in eids}, key=lambda sid: nums[sid])
        refs = "".join(cite(sid, eids) for sid in cited)
        tag = " *(analysis)*" if s["basis"] == "inference" else ""
        flag = ""
        sup = (checks.get(s["id"]) or {}).get("support")
        if sup and sup["verdict"] == "partial":
            flag = f" *(partly supported: {sup['problem']})*"
        if s["id"] in checks and not checks[s["id"]]["ok"]:
            flag = " **[unverified: " + "; ".join(checks[s["id"]]["issues"]) + "]**"
        return f"- {s['text']}{tag} {refs} `{s['id']}`{flag}"

    lines = [f"# {rep['title']}", "",
             f"Run `{run['id']}` · created {run['created_at']} · condition: {d['scope']['condition']}", ""]
    if ver:
        v = ver["summary"]
        lines += [f"> Verification: {v['statements_ok']} statements verified, {v['statements_flagged']} flagged, "
                  f"{v['inference_statements']} marked as analysis. Quotes: {v['evidence_exact']} exact, "
                  f"{v['evidence_near']} near, {v['evidence_missing']} not found."
                  + (f" Support check: {v['support_supported']} fully supported, {v['support_partial']} partly, "
                     f"{v['support_unsupported']} not supported." if "support_partial" in v else ""), ""]
    lines += ["## Executive summary", *map(stmt, rep["executive_summary"]), ""]
    for part, title in (("standard_of_care", "Standard of care"), ("emerging_treatments", "Emerging treatments")):
        lines += [f"## {title}", ""]
        for sec in rep[part]:
            lines += [f"### {sec['heading']}", *map(stmt, sec["statements"]), ""]
    lines += ["## Key companies and institutions", "",
              "| Organization | Type | Role | Trials |", "|---|---|---|---|"]
    for p in rep["key_players"]:
        trials = ", ".join(f"[{t}](https://clinicaltrials.gov/study/{t})" for t in p["trial_ids"])
        lines.append(f"| {p['name']} | {p['kind']} | {p['role']} | {trials} |")
    lines += ["", "## Implications for the health system", *map(stmt, rep["implications"]), "",
              "## Limitations", *[f"- {x}" for x in rep["limitations"]], ""]

    lines += ["## Sources", ""]
    for sid, n in sorted(nums.items(), key=lambda kv: kv[1]):
        s = srcs[sid]
        used = [e for e, x in ev.items() if x["source_id"] == sid]
        extra = s["meta"].get("journal") or f"{'/'.join(s['meta'].get('phases', []))}, {s['meta'].get('lead_sponsor', '')}"
        lines.append(f"{n}. <a id='src-{n}'></a>[{sid}]({s['url']}) {s['title']} {extra}, {s['year']}. "
                     f"Found by {', '.join(s['found_by'])}. Evidence {', '.join(used)}.")
    uncited = len(se["included"]) - len(nums)
    lines += ["", f"{uncited} more included sources were screened in but not cited; see the provenance bundle.", ""]

    plan = d["search_plan"]
    lines += ["## Method and provenance", ""]
    if run["parent"]:
        lines.append(f"- Derived from run `{run['parent']['run_id']}`, re-run from step `{run['parent']['fork_step']}`."
                     + (f" Change: {run['change_note']}" if run["change_note"] else ""))
    lines += [f"- Publication and trial start date range: {plan['year_from']} to {plan['year_to']}.",
              f"- Retrieved {len(srcs)} sources, included {len(se['included'])}, excluded {len(se['excluded'])}. "
              "Abstracts and trial registry records only, no full text.", "",
              "| Query | Source | Purpose | Query text | Hits | Used | Retrieved |", "|---|---|---|---|---|---|---|"]
    for q in d["retrieve"]["queries"]:
        hits = q.get("total_hits", "error")
        lines.append(f"| [{q['id']}]({q.get('url', '')}) | {q['kind']} | {q['purpose']} | `{q['query']}` | "
                     f"{hits} | {len(q['returned'])} | {q.get('retrieved_at', '')} |")
    lines += ["", "| Step | Status | Models | Prompt versions | LLM calls | Duration |", "|---|---|---|---|---|---|"]
    for s in STEPS:
        info = run["steps"][s]
        llm = info.get("llm", {})
        prompts = ", ".join(f"{k}@{v}" for k, v in llm.get("prompts", {}).items())
        status = info["status"] + (f" from `{info['inherited_from']}`" if info.get("inherited_from") else "")
        lines.append(f"| {s} | {status} | {', '.join(llm.get('models', []))} | {prompts} | "
                     f"{llm.get('calls', 0)} | {info.get('duration_s', '')}s |")
    return "\n".join(lines) + "\n"


def provenance_bundle(run_id: str) -> bytes:
    """Zip of run.json, every step record and the rendered report."""
    st = pipeline.store()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{run_id}/run.json", json.dumps(pipeline.load_run(run_id, st), indent=2, ensure_ascii=False))
        for s in STEPS:
            rec = pipeline.load_step(run_id, s, st)
            if rec:
                z.writestr(pipeline.step_key(run_id, s), json.dumps(rec, indent=2, ensure_ascii=False))
        z.writestr(f"{run_id}/report.md", render_markdown(run_id))
    return buf.getvalue()
