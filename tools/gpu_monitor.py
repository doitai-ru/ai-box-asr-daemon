# -*- coding: utf-8 -*-
"""
Standalone-монитор GPU-профиля.

Опрашивает /api/v1/admin/gpu-profile (URL+токен) раз в N сек, пишет CSV и
печатает таблицу; считает рост/плато по process_gpu_mib и корреляцию с
конкуренцией. Альтернатива — тейл logs/gpu_profile.jsonl (--jsonl).

Запуск:
  python tools/gpu_monitor.py --url http://127.0.0.1:49153 --token <JWT> --interval 2 --csv /tmp/gpu.csv
  python tools/gpu_monitor.py --jsonl logs/gpu_profile.jsonl
"""

import argparse
import json
import time
import urllib.request


def analyze(series) -> dict:
    """По ряду process_gpu_mib: max, дельта, вердикт plateau|growing|empty."""
    vals = [v for v in series if v is not None]
    if not vals:
        return {"max": None, "delta_tail": None, "verdict": "empty"}
    tail = vals[-min(4, len(vals)):]
    rising = all(b >= a for a, b in zip(tail, tail[1:])) and (tail[-1] - tail[0]) > 0
    verdict = "growing" if rising and len(tail) >= 3 else "plateau"
    return {"max": max(vals), "delta_tail": tail[-1] - tail[0], "verdict": verdict}


def _fetch(url: str, token: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/admin/gpu-profile",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--token", default="")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    series = []
    csv_f = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if csv_f:
        csv_f.write("ts,process_gpu_mib,gpu_used_mib,tone_backlog,conns_total\n")

    try:
        while True:
            if args.url:
                d = _fetch(args.url, args.token)
                if not d.get("enabled"):
                    print("GPU_PROFILE выключен на сервере"); return
                snap = d.get("snapshot", {})
                pm = snap.get("process_gpu_mib")
                used = snap.get("gpu_used_mib")
                backlog = d.get("tone_backlog")
                conns = (d.get("conns_by_path") or {}).get("total")
            else:
                # тейл JSONL: последняя sample-строка
                pm = used = backlog = conns = None
                try:
                    with open(args.jsonl, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-200:]
                    for ln in reversed(lines):
                        o = json.loads(ln)
                        if o.get("kind") == "sample":
                            pm = o.get("process_gpu_mib"); used = o.get("gpu_used_mib")
                            backlog = o.get("tone_backlog")
                            conns = (o.get("conns_by_path") or {}).get("total")
                            break
                except Exception:
                    pass
            series.append(pm)
            a = analyze(series)
            print(f"proc={pm} used={used} backlog={backlog} conns={conns} | max={a['max']} verdict={a['verdict']}")
            if csv_f:
                csv_f.write(f"{time.time()},{pm},{used},{backlog},{conns}\n"); csv_f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_f:
            csv_f.close()


if __name__ == "__main__":
    main()
