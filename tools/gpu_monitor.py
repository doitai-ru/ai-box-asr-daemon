# -*- coding: utf-8 -*-
"""
Standalone-монитор GPU-профиля с пороговой сигнализацией.

Тейлит logs/gpu_profile.jsonl (--jsonl) ИЛИ опрашивает /api/v1/admin/gpu-profile
(--url + --token) раз в N сек. Печатает статус-строку; при нарушении порогов —
строку 'ALERT! ...' (мало свободной видеопамяти / устойчиво высокий бэклог T-one).

Запуск:
  python tools/gpu_monitor.py --jsonl logs/gpu_profile.jsonl --interval 10 \
      --mem-free-min 1024 --backlog-max 40 --backlog-hold 6
  python tools/gpu_monitor.py --url http://127.0.0.1:49153 --token <JWT> --interval 10
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


def _last_sample_from_jsonl(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-400:]
    except Exception:
        return {}
    for ln in reversed(lines):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("kind") == "sample":
            return o
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--token", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--mem-free-min", type=int, default=1024, help="ALERT если свободно < N МиБ")
    ap.add_argument("--backlog-max", type=int, default=40, help="порог бэклога T-one")
    ap.add_argument("--backlog-hold", type=int, default=6, help="ALERT если бэклог > max подряд N раз")
    args = ap.parse_args()

    series = []
    backlog_streak = 0
    csv_f = open(args.csv, "a", encoding="utf-8") if args.csv else None
    if csv_f:
        csv_f.write("ts,process_gpu_mib,gpu_used_mib,gpu_free_mib,tone_backlog,conns_total,alert\n")

    try:
        while True:
            pm = used = free = backlog = conns = None
            if args.url:
                try:
                    d = _fetch(args.url, args.token)
                    if not d.get("enabled"):
                        print("GPU_PROFILE выключен на сервере"); return
                    s = d.get("snapshot", {}) or {}
                    pm = s.get("process_gpu_mib"); used = s.get("gpu_used_mib"); free = s.get("gpu_free_mib")
                    backlog = d.get("tone_backlog"); conns = (d.get("conns_by_path") or {}).get("total")
                except Exception as e:
                    print(f"{time.strftime('%H:%M:%S')} опрос не удался: {e}")
                    time.sleep(args.interval); continue
            else:
                o = _last_sample_from_jsonl(args.jsonl)
                pm = o.get("process_gpu_mib"); used = o.get("gpu_used_mib"); free = o.get("gpu_free_mib")
                backlog = o.get("tone_backlog"); conns = (o.get("conns_by_path") or {}).get("total")

            series.append(pm)
            a = analyze(series)

            alerts = []
            if free is not None and free < args.mem_free_min:
                alerts.append(f"МАЛО ВИДЕОПАМЯТИ: свободно {free} МиБ < {args.mem_free_min}")
            if backlog is not None and backlog > args.backlog_max:
                backlog_streak += 1
                if backlog_streak >= args.backlog_hold:
                    alerts.append(f"БЭКЛОГ T-one {backlog} > {args.backlog_max} ({backlog_streak} подряд)")
            else:
                backlog_streak = 0

            stamp = time.strftime("%H:%M:%S")
            status = f"{stamp} proc={pm} used={used} free={free} backlog={backlog} conns={conns} | max={a['max']} {a['verdict']}"
            line = ("ALERT! " + "; ".join(alerts) + " | " + status) if alerts else status
            print(line, flush=True)
            if csv_f:
                csv_f.write(f"{time.time()},{pm},{used},{free},{backlog},{conns},{int(bool(alerts))}\n"); csv_f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_f:
            csv_f.close()


if __name__ == "__main__":
    main()
