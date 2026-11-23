"""Linha de comando do laboratório: python -m lab <comando>"""
import argparse
import json

from . import config as C
from . import data, paper, reports
from .judge import judge, save


def main():
    ap = argparse.ArgumentParser(prog="lab")
    ap.add_argument("cmd", choices=["data", "scout", "judge", "engineer", "sentinel", "brief", "weekly",
                                    "approve", "resume", "status", "nightly", "alerts"])
    ap.add_argument("--perms", type=int, default=C.IS_PERMUTATIONS)
    ap.add_argument("--wf-perms", type=int, default=C.WF_PERMUTATIONS)
    a = ap.parse_args()

    if a.cmd == "data":
        for m in C.MARKETS:
            data.update(m)
            print(json.dumps(data.meta(m)))
    if a.cmd in ("scout", "nightly"):
        for m in C.MARKETS:
            data.update(m)
        sc = reports.scout()
        print("\n".join(reports.regime_lines(sc)))
    if a.cmd in ("judge", "nightly"):
        res = judge(a.perms, a.wf_perms)
        save(res, C.REPORTS_DIR / "research" / f"judge-{reports.today()}.json")
        sc = reports.scout()
        md = reports.research_report(res, sc)
        (C.REPORTS_DIR / "research" / f"{reports.today()}.md").write_text(md)
        print(md.split("## Todas as ideias")[0])
    if a.cmd in ("engineer", "nightly"):
        latest = sorted((C.REPORTS_DIR / "research").glob("judge-*.json"))[-1]  # Judge roda na véspera
        res = json.loads(latest.read_text())
        print("Engineer:", reports.publish_champions(res))
    if a.cmd == "sentinel":
        st = paper.step()
        print(json.dumps({"equity": st["account"]["equity"], "halted": st["halted"],
                          "positions": {m: b["position"] for m, b in st["books"].items()}}, indent=1))
    if a.cmd == "brief":
        print(reports.brief())
    if a.cmd == "weekly":
        print(reports.weekly())
    if a.cmd == "approve":
        print(reports.approve())
    if a.cmd == "resume":
        st = paper.resume()
        print("retomado" if st else "sem estado")
    if a.cmd == "alerts":
        # imprime alertas ainda não enviados e marca como enviados (o workflow abre uma issue com isso)
        st = paper.load_json(paper.STATE) or {}
        new = [x for x in st.get("alerts", []) if not x.get("notified")]
        for x in new:
            x["notified"] = True
        if new:
            paper.save_json(paper.STATE, st)
            print("\n".join(f"- {x['time']} UTC: {x['message']}" for x in new))
    if a.cmd == "status":
        st = paper.load_json(paper.STATE) or {}
        print(json.dumps({k: st.get(k) for k in ["last_run_utc", "account", "halted"]}, indent=1))


if __name__ == "__main__":
    main()
