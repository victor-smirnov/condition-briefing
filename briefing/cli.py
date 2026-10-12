"""Command line entry points, used for testing and for batch runs (e.g. a Cloud Run job).

    python -m briefing.cli run "type 2 diabetes" [--focus "..."]
    python -m briefing.cli fork <run_id> <step> '{"search_plan": {"year_from": 2023}}'
    python -m briefing.cli list
"""

import argparse
import json

from . import pipeline


def progress(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="briefing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("condition")
    r.add_argument("--focus")
    r.add_argument("--params", default="{}", help="JSON param overrides keyed by step")
    f = sub.add_parser("fork")
    f.add_argument("run_id")
    f.add_argument("step", choices=pipeline.STEPS)
    f.add_argument("changes", nargs="?", default="{}")
    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "run":
        rid = pipeline.create_run(a.condition, a.focus, json.loads(a.params))
        print(f"run {rid}")
        pipeline.execute(rid, on_progress=progress)
    elif a.cmd == "fork":
        rid, start = pipeline.fork(a.run_id, a.step, json.loads(a.changes), note="cli")
        print(f"run {rid} from {start}")
        pipeline.execute(rid, start, on_progress=progress)
        print(json.dumps(pipeline.diff_runs(a.run_id, rid), indent=2))
    else:
        for run in pipeline.list_runs():
            parent = run["parent"]["run_id"] if run["parent"] else "-"
            print(f"{run['id']}  {run['status']:8} parent={parent}")


if __name__ == "__main__":
    main()
