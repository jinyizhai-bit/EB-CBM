"""GPU job queue: keeps the given GPUs busy with the commands of a text file (one per line).

    python src/cbm/scripts/queue.py experiments/jobs_paper.txt [--gpus 0,1] [--per_gpu 1] [--dry]

* Lines are shell commands; blank lines and lines starting with '#' are ignored.  A line may start with
  ``[half]`` to mark a small job: when ``--per_gpu 2`` two jobs share a GPU only if both are ``[half]``
  (default: with --per_gpu 2 any two jobs may share).
* Each job gets ``CUDA_VISIBLE_DEVICES=<gpu>``; ``--gpu <gpu>`` is appended if the command has ``--run_id``
  but no ``--gpu`` (so the tracker records the physical GPU).  stdout+stderr go to
  experiments/runs/<run_id>/log.txt (run_id parsed from ``--run_id``; otherwise logs/queue_<n>.log), and
  ``CBM_STDOUT_IS_LOG=1`` tells the trainer not to duplicate its log lines.
* Jobs are pre-registered in the tracker as ``queued`` and marked ``failed`` if they exit non-zero before the
  trainer reports a final status.  The queue state is printed every minute; Ctrl-C terminates running jobs.
* Lines running ``cbm.eval.evaluate_run`` (with ``--run_id``) log into that run's log.txt but are not registered
  in / never change the tracker status of the run.
* A job that dies with CUDA out-of-memory at start-up (OOM signature in its log since launch and no new line in
  its metrics.jsonl) is re-queued once (``--oom_retries``, default 1) and relaunched after ``--retry_delay`` s
  (default 60); other jobs may use the GPU meanwhile.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

# this directory contains queue.py, which would shadow the stdlib ``queue`` module: drop it from sys.path
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # src/
from cbm import track as T  # noqa: E402
from cbm.paths import ROOT, SRC  # noqa: E402


def parse_jobs(path: str) -> list[dict]:
    jobs = []
    for n, line in enumerate(open(path)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        half = line.startswith("[half]")
        if half:
            line = line[len("[half]"):].strip()
        m = re.search(r"--run_id[ =](\S+)", line)
        rid = m.group(1) if m else None
        # evaluation-only jobs (cbm.eval.evaluate_run) log into the run folder but never touch its tracker status
        jobs.append({"n": n, "cmd": line, "run_id": rid, "half": half, "track": "evaluate_run" not in line})
    return jobs


OOM_SIGNS = ("out of memory", "outofmemoryerror", "cudaerrormemoryallocation")


def _metrics_lines(rid: str | None) -> int:
    if not rid:
        return 0
    p = T.RUNS_DIR / rid / "metrics.jsonl"
    try:
        with open(p) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _startup_oom(job: dict, logp: Path) -> bool:
    try:
        with open(logp, "rb") as f:
            f.seek(job.get("log_offset", 0))
            txt = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return False
    return any(sg in txt for sg in OOM_SIGNS) and _metrics_lines(job["run_id"]) <= job.get("metrics_at_launch", 0)


def run_status(rid: str) -> str | None:
    d = T._read("runs.json", {"runs": []})
    for r in d.get("runs", []):
        if r.get("id") == rid:
            return r.get("status")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--any_share", action="store_true", help="with --per_gpu 2, allow sharing for non-[half] jobs")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--oom_retries", type=int, default=1)
    ap.add_argument("--retry_delay", type=float, default=60.0)
    a = ap.parse_args(argv)
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    jobs = parse_jobs(a.jobs)
    print(f"[queue] {len(jobs)} jobs on GPUs {gpus} (per_gpu={a.per_gpu})", flush=True)
    for j in jobs:
        if j["run_id"] and j["track"] and not a.dry:
            T.update_run(j["run_id"], status="queued", name=j["run_id"], gpu=None, epoch=0)
    if a.dry:
        for j in jobs:
            print(j)
        return
    pending = list(jobs)
    running: dict[int, list] = {g: [] for g in gpus}   # gpu -> [(job, Popen, logfile)]
    T.log_event(f"queue started: {len(jobs)} jobs from {a.jobs} on GPUs {gpus}", source="queue")

    def stop_all(*_):
        for g in running:
            for _, p, _ in running[g]:
                p.terminate()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    last_print = 0.0
    while pending or any(running.values()):
        # reap
        for g in gpus:
            still = []
            for job, p, lf in running[g]:
                rc = p.poll()
                if rc is None:
                    still.append((job, p, lf))
                    continue
                lf.close()
                st = run_status(job["run_id"]) if job["run_id"] else None
                print(f"[queue] job {job['run_id'] or job['n']} on GPU{g} exited rc={rc} (tracker: {st})", flush=True)
                if rc != 0 and job.get("retries", 0) < a.oom_retries and _startup_oom(job, Path(lf.name)):
                    job["retries"] = job.get("retries", 0) + 1
                    job["not_before"] = time.time() + a.retry_delay
                    pending.insert(0, job)
                    print(f"[queue] start-up CUDA OOM: re-queued {job['run_id'] or job['n']} "
                          f"(retry {job['retries']}/{a.oom_retries} in {a.retry_delay:.0f}s)", flush=True)
                    if job["run_id"] and job["track"]:
                        T.update_run(job["run_id"], status="queued", error=None)
                    T.log_event(f"queue: {job['run_id'] or job['n']} hit CUDA OOM at start-up on GPU{g}; "
                                f"retrying in {a.retry_delay:.0f}s", "warn", source="queue")
                    continue
                if rc != 0 and job["run_id"] and job["track"] and st not in ("failed", "done"):
                    T.update_run(job["run_id"], status="failed", error=f"exit code {rc}")
            running[g] = still
        # launch
        for g in gpus:
            while pending:
                cur = running[g]
                if len(cur) >= a.per_gpu:
                    break
                ready = [i for i, j in enumerate(pending) if j.get("not_before", 0) <= time.time()]
                if not ready:
                    break
                cand = pending[ready[0]]
                if cur and not a.any_share and not (all(c[0]["half"] for c in cur) and cand["half"]):
                    break
                job = pending.pop(ready[0])
                cmd = job["cmd"]
                if job["run_id"] and "--gpu" not in cmd:
                    cmd += f" --gpu {g}"
                if job["run_id"]:
                    d = T.RUNS_DIR / job["run_id"]
                    d.mkdir(parents=True, exist_ok=True)
                    logp = d / "log.txt"
                else:
                    logp = ROOT / "logs" / f"queue_{job['n']}.log"
                logp.parent.mkdir(parents=True, exist_ok=True)
                lf = open(logp, "a")
                lf.write(f"# queue: {time.ctime()} GPU{g}: {cmd}\n")
                lf.flush()
                job["log_offset"] = lf.tell()
                job["metrics_at_launch"] = _metrics_lines(job["run_id"])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CBM_STDOUT_IS_LOG="1",
                           PYTHONPATH=f"{SRC}:" + os.environ.get("PYTHONPATH", ""), PYTHONUNBUFFERED="1")
                p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                                     executable="/bin/bash")
                running[g].append((job, p, lf))
                print(f"[queue] launched {job['run_id'] or job['n']} on GPU{g} (pid {p.pid}) -> {logp}", flush=True)
        if time.time() - last_print > 60:
            last_print = time.time()
            print(f"[queue] {time.strftime('%H:%M:%S')} pending={len(pending)} running="
                  + ", ".join(f"GPU{g}:{[j['run_id'] for j, _, _ in running[g]]}" for g in gpus), flush=True)
        time.sleep(a.poll)
    T.log_event(f"queue finished: {a.jobs}", "success", source="queue")
    print("[queue] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
