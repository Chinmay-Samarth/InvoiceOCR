"""Benchmark OCR configs for stability + accuracy on sample PDFs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
PORT = 8082
BASE = f"http://127.0.0.1:{PORT}"
SAMPLES = [
    Path(r"E:\das-cnc-files\sampleinvoice1.pdf"),
    Path(r"E:\das-cnc-files\sampleinvoice4.pdf"),
    Path(r"E:\das-cnc-files\salesinvoicesample1.pdf"),
]

CONFIGS = [
    {
        "name": "A_full_v4_t2_b4",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "150",
            "OCR_MAX_SIDE": "2400",
            "OCR_DET_LIMIT": "960",
            "OCR_CPU_THREADS": "2",
            "OCR_REC_BATCH_NUM": "4",
        },
    },
    {
        "name": "B_v4_side1920_t2_b4",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "150",
            "OCR_MAX_SIDE": "1920",
            "OCR_DET_LIMIT": "960",
            "OCR_CPU_THREADS": "2",
            "OCR_REC_BATCH_NUM": "4",
        },
    },
    {
        "name": "C_v4_dpi120_side1600_t2_b2",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "120",
            "OCR_MAX_SIDE": "1600",
            "OCR_DET_LIMIT": "960",
            "OCR_CPU_THREADS": "2",
            "OCR_REC_BATCH_NUM": "2",
        },
    },
    {
        "name": "D_v4_full_t1_b1",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "150",
            "OCR_MAX_SIDE": "2400",
            "OCR_DET_LIMIT": "960",
            "OCR_CPU_THREADS": "1",
            "OCR_REC_BATCH_NUM": "1",
        },
    },
    {
        "name": "E_v4_dpi150_side1600_t1_b2",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "150",
            "OCR_MAX_SIDE": "1600",
            "OCR_DET_LIMIT": "736",
            "OCR_CPU_THREADS": "1",
            "OCR_REC_BATCH_NUM": "2",
        },
    },
    {
        "name": "F_v4_dpi100_side1920_t2_b4",
        "env": {
            "OCR_LOW_MEMORY": "false",
            "OCR_FAST_MODE": "false",
            "OCR_ENABLE_MKLDNN": "false",
            "OCR_VERSION": "PP-OCRv4",
            "OCR_DPI": "100",
            "OCR_MAX_SIDE": "1920",
            "OCR_DET_LIMIT": "960",
            "OCR_CPU_THREADS": "2",
            "OCR_REC_BATCH_NUM": "4",
        },
    },
]


def score(data: dict) -> int:
    s = 0
    supplier = data.get("supplier") or {}
    invoice = data.get("invoice") or {}
    totals = data.get("totals") or {}
    if supplier.get("name"):
        s += 2
    if supplier.get("gstin"):
        s += 3
    if supplier.get("current_address") or supplier.get("billing_address"):
        s += 1
    if invoice.get("number"):
        s += 3
    if invoice.get("date"):
        s += 2
    if float(totals.get("total_amount") or 0) > 0:
        s += 3
    if float(totals.get("base_amount") or 0) > 0:
        s += 2
    if float(totals.get("tax_amount") or 0) > 0:
        s += 1
    lines = data.get("line_items") or []
    if lines:
        s += min(3, len(lines))
    conf = float(data.get("ocr_confidence") or 0)
    if conf >= 0.9:
        s += 2
    elif conf >= 0.75:
        s += 1
    return s


def http_get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_post_file(url: str, path: Path, timeout: float = 300.0):
    boundary = "----BenchBoundary7MA4YWxkTrZu0gW"
    file_bytes = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="document"; filename="{path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def wait_health(proc: subprocess.Popen, timeout: float = 180.0) -> dict | None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return None
        try:
            status, data = http_get(f"{BASE}/health", timeout=3)
            if status == 200:
                return data
        except Exception:
            pass
        time.sleep(1)
    return None


def kill_proc(proc: subprocess.Popen | None):
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_config(cfg: dict) -> dict:
    env = os.environ.copy()
    # Clear competing flags
    for k in list(env):
        if k.startswith("OCR_"):
            del env[k]
    env.update(cfg["env"])
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n=== {cfg['name']} ===", flush=True)
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = {
        "name": cfg["name"],
        "env": cfg["env"],
        "health": None,
        "samples": [],
        "crashes": 0,
        "ok": 0,
        "total_score": 0,
        "total_time": 0.0,
    }
    try:
        health = wait_health(proc)
        result["health"] = health
        if not health:
            print("  startup failed / process died", flush=True)
            result["crashes"] += 1
            return result
        print(f"  health={health}", flush=True)

        for sample in SAMPLES:
            if not sample.exists():
                result["samples"].append({"file": sample.name, "status": "missing"})
                continue
            t0 = time.time()
            try:
                status, data = http_post_file(f"{BASE}/parse", sample, timeout=300)
                elapsed = time.time() - t0
                sc = score(data)
                entry = {
                    "file": sample.name,
                    "status": "ok",
                    "http": status,
                    "seconds": round(elapsed, 1),
                    "score": sc,
                    "gstin": (data.get("supplier") or {}).get("gstin"),
                    "invoice_number": (data.get("invoice") or {}).get("number"),
                    "total_amount": (data.get("totals") or {}).get("total_amount"),
                    "confidence": data.get("ocr_confidence"),
                }
                result["ok"] += 1
                result["total_score"] += sc
                result["total_time"] += elapsed
                print(
                    f"  {sample.name}: ok {elapsed:.1f}s score={sc} "
                    f"gstin={entry['gstin']} inv={entry['invoice_number']} "
                    f"total={entry['total_amount']}",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                entry = {
                    "file": sample.name,
                    "status": "fail",
                    "seconds": round(elapsed, 1),
                    "error": str(exc),
                }
                result["crashes"] += 1
                print(f"  {sample.name}: FAIL {elapsed:.1f}s {exc}", flush=True)
                # If process died, stop remaining samples for this config
                if proc.poll() is not None:
                    print("  process exited after failure", flush=True)
                    result["samples"].append(entry)
                    break
                # Try health; if dead, stop
                try:
                    http_get(f"{BASE}/health", timeout=3)
                except Exception:
                    print("  health dead after failure", flush=True)
                    result["samples"].append(entry)
                    break
            result["samples"].append(entry)
    finally:
        kill_proc(proc)
        time.sleep(2)
    return result


def main():
    available = [p for p in SAMPLES if p.exists()]
    if not available:
        print("No sample PDFs found")
        sys.exit(1)

    report = []
    for cfg in CONFIGS:
        report.append(run_config(cfg))

    # Rank: fewest crashes, then highest score, then lower time
    ranked = sorted(
        report,
        key=lambda r: (
            r["crashes"],
            -r["total_score"],
            r["total_time"] if r["ok"] else 1e9,
        ),
    )
    out = ROOT / "bench_report.json"
    out.write_text(json.dumps({"ranked": ranked}, indent=2), encoding="utf-8")

    print("\n======== RANKING ========", flush=True)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['name']}: crashes={r['crashes']} ok={r['ok']} "
            f"score={r['total_score']} time={r['total_time']:.1f}s",
            flush=True,
        )
    best = ranked[0]
    print("\nBEST:", best["name"], flush=True)
    print(json.dumps(best["env"], indent=2), flush=True)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
